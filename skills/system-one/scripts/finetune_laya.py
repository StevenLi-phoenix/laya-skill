#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["laya==0.3.22"]
# ///
"""Fine-tune Laya on your own labelled decisions, calibrate, evaluate, and (optionally) push.

Single-process port of upstream's RLCD loop (notebooks/laya_finetune_typed_decisions_mps.py @
laya 0.3.22): proper-scoring-rule policy gradient + soft cross-entropy, per-type temperature
fit on a calibration slice held out from training, then a held-out evaluation of base vs
fine-tuned with the same metrics as compare.py.

Data: JSONL, one decision set per line, the `laya-evals` format plus an optional soft target:
  {"state": ..., "questions": {...}, "expected": {"qid": "label" | true/false | level},
   "gold": {"qid": {"probabilities": {"label": p, ...}}}}      # optional, overrides expected
Hard labels become smoothed one-hot targets (--label-smoothing).

Examples:
  uv run finetune_laya.py --smoke                                   # 1-2 min wiring check
  uv run finetune_laya.py --data train.jsonl --base english --epochs 4 --out runs/ft
  uv run finetune_laya.py --data train.jsonl --eval-data test.jsonl --freeze-encoder --out runs/ft-head
  uv run finetune_laya.py --data d.jsonl --out runs/ft --push-to-hub you/laya-myteam   # needs $HF_TOKEN
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systemone import read_jsonl, score_results, setup_logging, validate_questions

log = logging.getLogger("finetune_laya")
HERE = Path(__file__).resolve().parent
SMOKE_DATA = HERE.parent / "assets" / "examples" / "tickets.jsonl"
BUNDLE_REPO = "convaiinnovations/laya"
BASES = {"english": (BUNDLE_REPO, None), "multilingual": (BUNDLE_REPO, "multilingual"),
         "typed-decisions": (BUNDLE_REPO, "typed-decisions")}
CKPT_FILES = ("rl_agent_config.json", "model.safetensors", "config.json", "tokenizer.json", "tokenizer/*", "encoder/*")


# ---------------------------------------------------------------- data (no torch needed)

def target_for(question: dict[str, Any], expected: Any = None, gold: dict[str, Any] | None = None,
               smoothing: float = 0.05) -> list[float] | None:
    """Target distribution over the question's options, in the order the model sees them.

    choice: criteria key order; noul: [false, true]; score: level 0..n-1.
    `gold["probabilities"]` (soft) wins over `expected` (hard, smoothed). None = unusable.
    """
    t, crit = question["type"], question.get("criteria")
    if t == "choice":
        keys = [str(k) for k in (crit if isinstance(crit, dict) else crit or [])]
    elif t == "noul":
        keys = ["false", "true"]
    else:
        keys = [str(i) for i in range(len(crit or []))]
    k = len(keys)
    if k < 2:
        return None
    if gold and isinstance(gold.get("probabilities"), dict):
        p = [float(gold["probabilities"].get(key, 0.0)) for key in keys]
        s = sum(p)
        return [x / s for x in p] if s > 0 else None
    hard = [0.0] * k
    if t == "choice":
        if str(expected) not in keys:
            return None
        hard[keys.index(str(expected))] = 1.0
    elif t == "noul":
        if not isinstance(expected, bool):
            return None
        hard[int(expected)] = 1.0
    else:
        if not isinstance(expected, (int, float)) or isinstance(expected, bool) or not 0 <= expected <= k - 1:
            return None
        lo = math.floor(expected)
        frac = float(expected) - lo
        hard[lo] += 1.0 - frac
        if frac:
            hard[lo + 1] += frac
    return [(1.0 - smoothing) * h + smoothing / k for h in hard]


def split_rows(rows: list[dict[str, Any]], eval_frac: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train / held-out split at row level (every decision of a row stays together)."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, round(len(rows) * eval_frac)) if eval_frac > 0 and len(rows) > 1 else 0
    held = set(idx[:n_eval])
    return [r for i, r in enumerate(rows) if i not in held], [r for i, r in enumerate(rows) if i in held]


def data_fingerprint(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- model

def resolve_base(base: str) -> tuple[str, str]:
    """Local checkpoint dir for a base name, Hub id (optionally 'org/repo:subfolder') or path."""
    from huggingface_hub import snapshot_download

    if Path(base).is_dir():
        return str(Path(base).resolve()), base
    repo, sub = BASES.get(base, (base.split(":")[0], base.split(":")[1] if ":" in base else None))
    prefix = f"{sub}/" if sub else ""
    root = snapshot_download(repo, allow_patterns=[prefix + f for f in CKPT_FILES], token=os.environ.get("HF_TOKEN"))
    path = os.path.join(root, sub) if sub else root
    return path, f"{repo}/{sub}" if sub else repo


def choose_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(requested)


def build_items(tok: Any, cfg: dict[str, Any], rows: list[dict[str, Any]], smoothing: float) -> tuple[list[dict[str, Any]], int]:
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, render_options

    items, skipped = [], 0
    for row in rows:
        for qid, q in row["questions"].items():
            target = target_for(q, (row.get("expected") or {}).get(qid), (row.get("gold") or {}).get(qid), smoothing)
            if target is None:
                skipped += 1
                continue
            internal = Agent._to_internal(q)
            ids, markers = build_sequence(tok, row["state"], internal, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(internal)) or len(markers) != len(target):
                skipped += 1  # options did not fit the head budget
                continue
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]], "target": target})
    return items, skipped


def collate(items: list[dict[str, Any]], pad_id: int) -> tuple[Any, ...]:
    import torch

    b, seq, kmax = len(items), max(len(i["ids"]) for i in items), max(len(i["markers"]) for i in items)
    ids = torch.full((b, seq), pad_id, dtype=torch.long)
    att = torch.zeros((b, seq), dtype=torch.long)
    pos = torch.zeros((b, kmax), dtype=torch.long)
    mask = torch.zeros((b, kmax), dtype=torch.bool)
    tgt = torch.zeros((b, kmax), dtype=torch.float32)
    for n, it in enumerate(items):
        L, k = len(it["ids"]), len(it["markers"])
        ids[n, :L] = torch.tensor(it["ids"])
        att[n, :L] = 1
        pos[n, :k] = torch.tensor(it["markers"])
        mask[n, :k] = True
        tgt[n, :k] = torch.tensor(it["target"])
    return ids, att, pos, mask, tgt, torch.tensor([it["qtype"] for it in items], dtype=torch.long)


def save_checkpoint(model: Any, tok: Any, cfg: dict[str, Any], path: Path, meta: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    model.encoder.config.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    (path / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    (path / "finetune_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def train(args: argparse.Namespace, base_dir: str, train_rows: list[dict[str, Any]], out: Path, meta: dict[str, Any]) -> Path:
    """RLCD fine-tune (upstream recipe), then fit per-type temperatures. Returns the checkpoint dir."""
    import torch
    from laya.agent import _fix_tokenizer_config
    from laya.calibrate import MIN_TYPE_N, fit_temperature_map
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    device = choose_device(args.device)
    _fix_tokenizer_config(base_dir)
    cfg = json.loads((Path(base_dir) / "rl_agent_config.json").read_text())
    if args.max_len:
        cfg["max_len"] = args.max_len
    if args.head_max_len:
        cfg["head_max_len"] = args.head_max_len
    tok = AutoTokenizer.from_pretrained(Path(base_dir) / "tokenizer")
    model = build_model(cfg, encoder_dir=str(Path(base_dir) / "encoder"))
    model.load_state_dict(load_file(str(Path(base_dir) / "model.safetensors")), strict=True)
    model.float()

    items, skipped = build_items(tok, cfg, train_rows, args.label_smoothing)
    if len(items) < 2:
        raise SystemExit(f"only {len(items)} usable training decisions (skipped {skipped}); check 'expected' labels")
    order = list(range(len(items)))
    random.Random(args.seed).shuffle(order)
    # Calibration slice is taken out BEFORE training: fitting temperatures on trained-on items
    # measures the fit, not the calibration (upstream docs/finetune.md).
    n_cal = min(args.calib_max, len(items) // 10) if len(items) >= 20 else 0
    cal_items = [items[i] for i in order[:n_cal]]
    tr_items = [items[i] for i in order[n_cal:]]
    log.info("device=%s decisions: train=%d calib=%d skipped=%d | max_len=%d head_max_len=%d",
             device, len(tr_items), len(cal_items), skipped, cfg["max_len"], cfg["head_max_len"])

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    if args.freeze_encoder:
        for p in enc_params:
            p.requires_grad_(False)
        groups = [{"params": head_params, "lr": args.lr_head}]
    else:
        if not args.no_checkpointing:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.head_checkpointing = True
        groups = [{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}]
    model.to(device).train()
    if args.freeze_encoder:
        model.encoder.eval()  # no dropout in a frozen encoder
    opt =torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr_items) / args.micro_batch / args.grad_accum)
    total_updates = max(1, min(steps_per_epoch * args.epochs, args.max_steps or 10**9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    log.info("updates=%d (micro_batch=%d grad_accum=%d epochs=%d freeze_encoder=%s)",
             total_updates, args.micro_batch, args.grad_accum, args.epochs, args.freeze_encoder)

    updates, t0 = 0, time.perf_counter()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(tr_items)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)  # exploration noise anneal
        opt.zero_grad(set_to_none=True)
        run_loss, n_micro = 0.0, 0
        for start in range(0, len(tr_items), args.micro_batch):
            ids, att, pos, mask, tgt, qt = (x.to(device) for x in collate(tr_items[start:start + args.micro_batch], tok.pad_token_id))
            logits, act = model(ids, att, pos, mask, qt, detach_encoder=args.freeze_encoder)
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy = logits.detach().unsqueeze(0) + eps
            probs = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(probs, tgt.unsqueeze(0), qt, mask, w_sph=0.75, w_rps=1.0)
                adv = reward - reward.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(tgt * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce + 0.0 * act.sum()) / args.grad_accum
            loss.backward()
            n_micro += 1
            run_loss += loss.item() * args.grad_accum
            if n_micro % args.grad_accum == 0 or start + args.micro_batch >= len(tr_items):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
                if updates % args.log_every == 0 or updates == total_updates:
                    log.info("epoch %d update %d/%d loss=%.4f ce=%.4f %.1fs", epoch + 1, updates, total_updates,
                             run_loss / n_micro, loss_ce.item(), time.perf_counter() - t0)
                if updates >= total_updates:
                    break
        log.info("epoch %d done avg_loss=%.4f", epoch + 1, run_loss / max(1, n_micro))
        if updates >= total_updates:
            break

    model.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(cal_items), args.micro_batch):
            chunk = cal_items[start:start + args.micro_batch]
            ids, att, pos, mask, tgt, qt = (x.to(device) for x in collate(chunk, tok.pad_token_id))
            logits, _ = model(ids, att, pos, mask, qt)
            for n, it in enumerate(chunk):
                kk = len(it["markers"])
                records.append((it["qtype"], logits[n, :kk].float().cpu().numpy(), it["target"], kk))
    base_t = cfg.get("temperature")
    base_t = [float(x) for x in base_t] if isinstance(base_t, list) and len(base_t) == 3 else [1.0, 1.0, 1.0]
    counts = [sum(1 for r in records if r[0] == t) for t in range(3)]
    if records:
        fitted = fit_temperature_map(records)["temperature"]
        # A type below the fitter's floor comes back as 1.0; keep the base value for it instead.
        cfg["temperature"] = [float(fitted[t]) if counts[t] >= MIN_TYPE_N else base_t[t] for t in range(3)]
    else:
        cfg["temperature"] = base_t
    log.info("temperatures (choice, score, noul) = %s; calib decisions per type %s (fit needs >= %d, else base kept)",
             [round(x, 3) for x in cfg["temperature"]], counts, MIN_TYPE_N)
    # Inherited bucket temperatures take precedence at inference and would mask the new fit.
    cfg.pop("temperature_by_options", None)
    cfg["fine_tuned"] = True
    meta.update({"train_decisions": len(tr_items), "calib_decisions": len(cal_items), "updates": updates,
                 "train_seconds": round(time.perf_counter() - t0, 1), "temperature": cfg.get("temperature")})
    ckpt = out / "checkpoint"
    save_checkpoint(model, tok, cfg, ckpt, meta)
    log.info("checkpoint -> %s", ckpt)
    del model, opt
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return ckpt


def evaluate_checkpoint(model_ref: str, rows: list[dict[str, Any]], device: str | None) -> tuple[dict[str, Any], list]:
    import laya

    agent = laya.load(model_ref, device=None if device == "auto" else device)
    results = []
    for row in rows:
        t0 = time.perf_counter()
        r = agent.predict(row["state"], row["questions"])
        results.append({**r, "_meta": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)}})
    del agent
    gc.collect()
    return score_results(rows, results), results


def push_to_hub(ckpt: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--push-to-hub needs a write token in $HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), repo_id=repo_id, commit_message="laya fine-tune (system-one skill)")
    return f"https://huggingface.co/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="labelled JSONL (train; held-out split carved from it unless --eval-data)")
    ap.add_argument("--eval-data", help="separate held-out JSONL")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--base", default="english", help="english | multilingual | typed-decisions | <dir> | org/repo[:subfolder]")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer updates (0 = no cap)")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder", action="store_true", help="train only the decision head (fast, low memory)")
    ap.add_argument("--no-checkpointing", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--calib-max", type=int, default=400)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--skip-base-eval", action="store_true", help="do not evaluate the base checkpoint")
    ap.add_argument("--push-to-hub", metavar="ORG/REPO", help="upload the checkpoint (token from $HF_TOKEN)")
    ap.add_argument("--public", action="store_true", help="with --push-to-hub: create a public repo")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check: bundled examples, multilingual base, head-only, 8 updates, max_len 256")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    if args.smoke:
        args.data = args.data or str(SMOKE_DATA)
        args.base = "multilingual" if args.base == "english" else args.base
        args.epochs, args.max_steps, args.grad_accum, args.micro_batch = 1, args.max_steps or 8, 1, 4
        args.max_len, args.freeze_encoder, args.log_every = args.max_len or 256, True, 2
        args.out = args.out if args.out != "runs/finetune" else "runs/finetune-smoke"
    if not args.data:
        ap.error("--data is required (or use --smoke)")

    rows = read_jsonl(args.data)
    for i, r in enumerate(rows):
        validate_questions(r.get("questions"))
        if "expected" not in r and "gold" not in r:
            raise SystemExit(f"row {i}: needs 'expected' or 'gold'")
    if args.eval_data:
        train_rows, eval_rows = rows, read_jsonl(args.eval_data)
    else:
        train_rows, eval_rows = split_rows(rows, args.eval_frac, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_dir, base_ref = resolve_base(args.base)
    log.info("base %s (%s); rows train=%d eval=%d", args.base, base_dir, len(train_rows), len(eval_rows))
    meta: dict[str, Any] = {"base": base_ref, "data": args.data, "data_sha": data_fingerprint(rows),
                            "args": {k: v for k, v in vars(args).items() if k != "verbose"}}

    ckpt = train(args, base_dir, train_rows, out, meta)

    report: dict[str, Any] = {"base": base_ref, "checkpoint": str(ckpt), "eval_rows": len(eval_rows)}
    if eval_rows:
        if not args.skip_base_eval:
            report["base_metrics"], _ = evaluate_checkpoint(base_dir, eval_rows, args.device)
        report["finetuned_metrics"], results = evaluate_checkpoint(str(ckpt), eval_rows, args.device)
        report["sample_answer"] = results[0]["answers"]
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log.info("report -> %s", out / "report.json")
    for k in ("base_metrics", "finetuned_metrics"):
        if k in report:
            m = report[k]
            log.info("%-17s accuracy=%s choice=%s noul=%s score=%s ece=%s", k, m["accuracy"], m["choice_accuracy"],
                     m["noul_accuracy"], m["score_accuracy"], m["ece"])
    if args.push_to_hub:
        log.info("pushed -> %s", push_to_hub(ckpt, args.push_to_hub, private=not args.public))
    print(json.dumps({k: v for k, v in report.items() if k != "sample_answer"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
