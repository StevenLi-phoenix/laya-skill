// Evaluate one or more split-ONNX Laya exports (laya-ts, onnxruntime-node, CPU) on a labelled JSONL
// and compare them: accuracy per question, agreement with the first model, max probability drift.
// Usage: node eval_onnx.mjs <tickets.jsonl> <modelDirA> [modelDirB ...]  > report.json
import { readFileSync } from "node:fs";
import { Agent } from "../vendor/laya-ts/index.js";

const [, , dataPath, ...dirs] = process.argv;
if (!dataPath || dirs.length === 0) {
  console.error("usage: node eval_onnx.mjs <data.jsonl> <modelDir> [modelDir ...]");
  process.exit(2);
}
const log = (...a) => console.error("[eval_onnx]", ...a);
const rows = readFileSync(dataPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));

function correct(ans, expected) {
  if (expected === undefined || expected === null) return null;
  if (ans.type === "choice") return ans.choice === expected;
  if (ans.type === "score") {
    const top = Object.entries(ans.probabilities).sort((a, b) => b[1] - a[1])[0][0];
    return Number(top) === Number(expected);
  }
  return (ans.noul >= 0.5) === Boolean(expected);
}
const probs = (a) => (a.type === "noul" ? { yes: a.noul } : a.probabilities);

const runs = [];
for (const dir of dirs) {
  const t0 = performance.now();
  const agent = await Agent.load(dir, { device: "cpu" });
  const loadMs = performance.now() - t0;
  log(dir, `loaded in ${(loadMs / 1000).toFixed(1)}s`);
  const out = [];
  const lat = [];
  for (const r of rows) {
    const t = performance.now();
    const res = await agent.predict(r.state, r.questions);
    lat.push(performance.now() - t);
    out.push(res.answers);
  }
  lat.sort((a, b) => a - b);
  runs.push({ dir, loadMs, answers: out, p50: lat[Math.floor(lat.length / 2)] });
}

const report = { data: dataPath, n: rows.length, models: [] };
const base = runs[0];
for (const run of runs) {
  const acc = {};
  let agree = 0, total = 0, drift = 0;
  run.answers.forEach((answers, i) => {
    for (const [qid, a] of Object.entries(answers)) {
      const c = correct(a, rows[i].expected?.[qid]);
      if (c !== null) {
        acc[qid] ??= [0, 0];
        acc[qid][0] += c ? 1 : 0;
        acc[qid][1] += 1;
      }
      const b = base.answers[i][qid];
      total += 1;
      agree += correct(a, a.type === "choice" ? b.choice : a.type === "score"
        ? Number(Object.entries(b.probabilities).sort((x, y) => y[1] - x[1])[0][0]) : b.noul >= 0.5) ? 1 : 0;
      const pa = probs(a), pb = probs(b);
      for (const k of Object.keys(pa)) drift = Math.max(drift, Math.abs(pa[k] - pb[k]));
    }
  });
  report.models.push({
    dir: run.dir,
    load_s: +(run.loadMs / 1000).toFixed(1),
    cpu_p50_ms: +run.p50.toFixed(0),
    accuracy: Object.fromEntries(Object.entries(acc).map(([k, [c, n]]) => [k, `${c}/${n} = ${(c / n).toFixed(3)}`])),
    agreement_with_first: `${agree}/${total}`,
    max_prob_drift_vs_first: +drift.toFixed(4),
  });
}
console.log(JSON.stringify(report, null, 2));
