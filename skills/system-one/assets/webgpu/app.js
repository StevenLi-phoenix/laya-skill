// Laya WebGPU sample: loads a split-ONNX Laya export with laya-ts and times decisions.
// Every event is logged to the console with a "[laya-sample]" prefix for debugging.
import { Agent } from "./vendor/laya-ts/index.js";

const $ = (id) => document.getElementById(id);
const log = (...a) => console.log("[laya-sample]", ...a);

const DEPT = {
  type: "choice", instructions: "Which department should handle this?",
  criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages, crashes", other: "everything else" },
};
const PRESETS = [
  { name: "EN 重复扣费", text: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    q: { department: DEPT,
         urgency: { type: "score", instructions: "How urgent is this?", criteria: ["not urgent", "soon", "blocking"] },
         churn_risk: { type: "noul", instructions: "Does the user threaten to cancel or leave?" } } },
  { name: "中文 闪退", text: "我打开设置页面应用就闪退，已经重装两次了，还是不行。",
    q: { department: DEPT, frustrated: { type: "noul", instructions: "Is the user frustrated?" } } },
  { name: "हिन्दी 退款", text: "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।", q: { department: DEPT } },
  { name: "ES 崩溃", text: "La aplicación se cierra cada vez que abro la configuración.", q: { department: DEPT } },
  { name: "钓鱼邮件", text: "URGENT: Your account will be suspended in 24h. Verify your password now at http://secure-login.example-bank.co/verify",
    q: { phishing: { type: "noul", instructions: "Is this email a phishing attempt?" },
         action: { type: "choice", instructions: "What should the mail client do?",
                   criteria: { deliver: "normal mail", flag: "suspicious, warn the user", quarantine: "clearly malicious" } } } },
];

let agent = null;
let epFallback = false;
const warm = [];

// laya-ts warns once when a WebGPU encoder run fails and it switches to WASM; surface that in the UI.
const origWarn = console.warn.bind(console);
console.warn = (...a) => {
  if (/WebGPU encoder run failed/i.test(String(a[0]))) { epFallback = true; setEp(); }
  origWarn(...a);
};

function chip(id, text, cls = "") { const el = $(id); el.textContent = text; el.className = `chip ${cls}`; }
function setEp() {
  if (!agent) return;
  if (!navigator.gpu || epFallback) chip("chip-ep", "EP wasm", "warn");
  else chip("chip-ep", "EP webgpu", "hot");
}

async function probeGpu() {
  if (!navigator.gpu) { chip("chip-gpu", "WebGPU ✗", "warn"); log("navigator.gpu missing"); return; }
  try {
    const ad = await navigator.gpu.requestAdapter();
    if (!ad) { chip("chip-gpu", "WebGPU 无 adapter", "warn"); return; }
    const info = ad.info || {};
    chip("chip-gpu", `WebGPU ✓ ${info.vendor || ""} ${info.architecture || ""}`.trim(), "on");
    log("adapter", info.vendor, info.architecture, "maxBuffer", ad.limits.maxBufferSize);
  } catch (e) { chip("chip-gpu", "WebGPU ✗", "warn"); log("adapter error", e); }
}

function renderPresets() {
  const box = $("presets");
  PRESETS.forEach((p, i) => {
    const b = document.createElement("button");
    b.textContent = p.name;
    b.onclick = () => pick(i);
    box.append(b);
  });
  pick(0);
}
function pick(i) {
  [...$("presets").children].forEach((b, j) => b.classList.toggle("sel", i === j));
  $("text").value = PRESETS[i].text;
  $("questions").value = JSON.stringify(PRESETS[i].q, null, 2);
}

function resolveUrl(raw) {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) return raw;
  return new URL(raw, document.baseURI).href;
}

$("load").onclick = async () => {
  const url = resolveUrl($("model").value.trim());
  const bar = $("progress"), fill = bar.firstElementChild;
  bar.hidden = false; fill.style.width = "4%";
  $("load").disabled = true;
  const t0 = performance.now();
  const tick = setInterval(() => { $("log").textContent = `下载 / 建 session 中… ${((performance.now() - t0) / 1000).toFixed(0)}s`; }, 500);
  log("load", url);
  try {
    agent = await Agent.load(url, {
      onProgress: (done, total, file) => {
        fill.style.width = `${Math.max(4, (done / total) * 100)}%`;
        log("progress", file, `${done}/${total}`);
      },
    });
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    $("load-time").textContent = `${s}s`;
    $("log").textContent = `加载完成，用时 ${s}s。`;
    chip("chip-model", `${url.split("/").filter(Boolean).pop()} · max_len ${agent.maxLen}`, "on");
    setEp();
    $("run").disabled = $("bench").disabled = false;
    log("loaded in", s, "s");
  } catch (e) {
    $("log").textContent = `加载失败：${e.message}`;
    $("load").disabled = false;
    log("load failed", e);
  } finally { clearInterval(tick); fill.style.width = "100%"; setTimeout(() => (bar.hidden = true), 600); }
};

$("clear").onclick = async () => {
  const keys = await caches.keys();
  await Promise.all(keys.map((k) => caches.delete(k)));
  $("log").textContent = `已清空 ${keys.length} 个 CacheStorage 分区，下次加载会重新下载。`;
  log("cleared caches", keys);
};

function readQuestions() {
  try { $("qerr").textContent = ""; return JSON.parse($("questions").value); }
  catch (e) { $("qerr").textContent = `questions 不是合法 JSON：${e.message}`; return null; }
}

async function once(text, qs) {
  const t = performance.now();
  const r = await agent.predict(text, qs);
  return { r, ms: performance.now() - t };
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const quant = (arr, q) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

function renderStats(ms, r) {
  $("s-last").textContent = ms.toFixed(0);
  if (warm.length) { $("s-p50").textContent = quant(warm, 0.5).toFixed(0); $("s-p90").textContent = quant(warm, 0.9).toFixed(0); }
  $("s-tok").textContent = r.usage?.input_tokens ?? "—";
  $("routing").textContent = r.usage?.truncated ? "state 被截断" : "";
}

function optRows(probs, top, legend) {
  return Object.entries(probs).map(([k, p]) => {
    const label = legend ? `${k} · ${legend[k]}` : k;
    return `<div class="opt ${k === top ? "top" : ""}"><span>${esc(label)}</span><span class="track"><i style="width:${pct(p)}"></i></span><span class="num">${pct(p)}</span></div>`;
  }).join("");
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function renderAnswers(r) {
  const html = Object.entries(r.answers).map(([qid, a]) => {
    const low = a.low_confidence ? `<span class="low">低置信</span>` : "";
    if (a.type === "choice")
      return `<div class="ans"><div class="ans-head"><span class="verdict">${esc(a.choice)}${low}</span><code>${esc(qid)} · choice · conf ${a.confidence.toFixed(2)}</code></div>${optRows(a.probabilities, a.choice)}</div>`;
    if (a.type === "score") {
      const top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0];
      return `<div class="ans"><div class="ans-head"><span class="verdict">${a.score.toFixed(2)}${low}</span><code>${esc(qid)} · score · conf ${a.confidence.toFixed(2)}</code></div>${optRows(a.probabilities, top, a.legend)}</div>`;
    }
    const yes = a.noul >= 0.5;
    return `<div class="ans"><div class="ans-head"><span class="verdict">${yes ? "是" : "否"} · ${pct(a.noul)}${low}</span><code>${esc(qid)} · noul</code></div>${optRows({ yes: a.noul, no: 1 - a.noul }, yes ? "yes" : "no")}</div>`;
  }).join("");
  $("answers").innerHTML = html;
}

let first = true;
$("run").onclick = async () => {
  const qs = readQuestions(); if (!qs) return;
  $("run").disabled = true;
  try {
    const { r, ms } = await once($("text").value, qs);
    if (!first) warm.push(ms);
    log(first ? "cold run" : "warm run", `${ms.toFixed(1)}ms`, r);
    first = false;
    renderStats(ms, r); renderAnswers(r); setEp();
  } catch (e) { $("qerr").textContent = e.message; log("predict failed", e); }
  finally { $("run").disabled = false; }
};

$("bench").onclick = async () => {
  const qs = readQuestions(); if (!qs) return;
  $("bench").disabled = $("run").disabled = true;
  try {
    if (first) { await once($("text").value, qs); first = false; }
    let last;
    for (let i = 0; i < 20; i++) {
      last = await once($("text").value, qs);
      warm.push(last.ms);
      $("qerr").textContent = `测速 ${i + 1}/20`;
    }
    $("qerr").textContent = "";
    renderStats(last.ms, last.r); renderAnswers(last.r); setEp();
    log("bench", { n: warm.length, p50: quant(warm, 0.5), p90: quant(warm, 0.9), ep: epFallback ? "wasm" : "webgpu" });
    window.__lastBench = { p50: quant(warm, 0.5), p90: quant(warm, 0.9), n: warm.length, ep: (!navigator.gpu || epFallback) ? "wasm" : "webgpu" };
  } catch (e) { $("qerr").textContent = e.message; log("bench failed", e); }
  finally { $("bench").disabled = $("run").disabled = false; }
};

$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("run").click(); });

probeGpu();
renderPresets();
