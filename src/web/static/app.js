/* Face Provenance — dashboard frontend logic. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = {
  file: null,      // File to upload (defaults to the bundled sample image)
  chain: "anvil",
  running: false,
};

/* ---------------------------------------------------------------- *
 *  Initialization
 * ---------------------------------------------------------------- */
async function init() {
  // Default input: the bundled public-domain sample face (zero setup demo).
  const resp = await fetch("/api/sample-image");
  const blob = await resp.blob();
  state.file = new File([blob], "sample_face.jpg", { type: "image/jpeg" });
  drawPreview(state.file, null);
  $("#dz-empty").style.display = "none";

  bindControls();
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      $("#health-mode").textContent =
        h.mode + (h.search_api_key_configured ? " · key ✓" : " · no key");
    })
    .catch(() => {});
}

/* ---------------------------------------------------------------- *
 *  Upload / preview
 * ---------------------------------------------------------------- */
function bindControls() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) setFile(input.files[0]); });

  $("#chain-select").addEventListener("change", (e) => { state.chain = e.target.value; });

  $("#run-btn").addEventListener("click", () => run(state.chain));
}

function setFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file (JPEG/PNG/WebP).", "err");
    return;
  }
  state.file = file;
  drawPreview(file, null);
  $("#dz-empty").style.display = "none";
}

function drawPreview(file, bbox) {
  const canvas = $("#preview-canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  const url = URL.createObjectURL(file);
  img.onload = () => {
    const MAX_W = 320;
    const scale = Math.min(1, MAX_W / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (bbox) {
      const [x, y, w, h] = bbox;
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 3;
      ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);
      ctx.strokeStyle = "rgba(34,211,238,0.35)";
      ctx.lineWidth = 8;
      ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);
    }
    canvas.hidden = false;
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

/* ---------------------------------------------------------------- *
 *  Pipeline execution
 * ---------------------------------------------------------------- */
async function run(chain) {
  if (state.running) return;
  state.running = true;
  setRunning(true);
  setStatus("");

  resetSteps();
  const fd = new FormData();
  fd.append("file", state.file, state.file.name);
  fd.append("mode", "real");
  fd.append("chain", chain);

  // Animate the stepper while we wait.
  animateSteps();

  try {
    const result = await fetchJson("/api/process", fd);
    renderResult(result);
  } catch (err) {
    handleError(err, chain);
  } finally {
    state.running = false;
    setRunning(false);
  }
}

async function fetchJson(url, formData) {
  const resp = await fetch(url, { method: "POST", body: formData });
  let data = null;
  try { data = await resp.json(); } catch (_) { /* non-JSON error body */ }
  if (!resp.ok) {
    const detail = (data && data.detail) || `HTTP ${resp.status}`;
    const err = new Error(detail);
    err.status = resp.status;
    throw err;
  }
  return data;
}

/* ---------------------------------------------------------------- *
 *  Rendering
 * ---------------------------------------------------------------- */
function resetSteps() {
  document.querySelectorAll(".step").forEach((s) => {
    s.classList.remove("done", "failed", "active", "skipped");
  });
  $("#verdict").hidden = true;
  $("#json-drawer").hidden = true;
  $("#pipeline-sub").textContent = "Running…";
}

function animateSteps() {
  // Simulate sequential progress while the pipeline runs (server is fast, so
  // steps complete in a short sweep). Real states are applied by renderResult.
  const steps = document.querySelectorAll(".step");
  steps.forEach((s, i) => {
    s.classList.remove("done", "failed");
    setTimeout(() => {
      if (state.running && !s.classList.contains("done") && !s.classList.contains("failed")) {
        s.classList.add("active");
      }
    }, 120 + i * 130);
  });
}

function renderResult(r) {
  $("#pipeline-sub").textContent = r.completed
    ? "Completed ✓"
    : (r.match ? "Completed — not verified" : "Completed — no match");

  // [1] Face
  const f = r.face;
  markStep(0, f && f.face_detected ? "done" : "failed");
  const fd = $("#steps").querySelector('[data-detail="0"]');
  if (f) {
    fd.innerHTML = `
      <div class="kv"><span class="k">Detected</span><span class="v">${f.face_detected ? "YES" : "NO"}</span></div>
      <div class="kv"><span class="k">Faces</span><span class="v">${f.face_count}</span></div>
      ${f.faces.length
        ? `<div class="kv"><span class="k">Bounding box</span><span class="v mono">[${f.faces[0].bbox.join(", ")}]</span></div>`
        : ""}
      <div class="rationale">${f.privacy_note}</div>`;
    if (f.face_detected && f.faces.length) drawPreview(state.file, f.faces[0].bbox);
  }

  // [2] Search
  if (!r.search) return;
  const s = r.search;
  markStep(1, s.match_found ? "done" : "failed");
  $("#steps").querySelector('[data-detail="1"]').innerHTML = `
    <div class="kv"><span class="k">Provider</span><span class="v">${esc(s.provider)}</span></div>
    <div class="kv"><span class="k">Match</span><span class="v">${s.match_found ? "YES" : "NO"}</span></div>
    <div class="kv"><span class="k">Candidates</span><span class="v">${s.candidates ? s.candidates.length : 0}</span></div>`;

  // [3] Validation
  if (!r.match) {
    markStep(2, "failed");
    ["3", "4", "5"].forEach((n) => {
      const step = $(`[data-step="${n}"]`);
      if (step) step.classList.add("skipped");
    });
    $("#steps").querySelector('[data-detail="2"]').innerHTML = `
      <span class="badge none">NO MATCH</span>
      <div class="rationale">${esc(r.no_match_reason || "No permitted matching public result found")}
      — the pipeline stopped honestly and did not touch the blockchain.</div>`;
    showVerdict("warn", "NO MATCH", r.no_match_reason || "No permitted matching public result found");
    showJson(r);
    $("#pipeline-sub").textContent = "Completed — no match";
    return;
  }

  const m = r.match;
  markStep(2, "done");
  // Pydantic serializes enums to their string value; fall back defensively.
  const mtype = typeof m.match_type === "string" ? m.match_type : m.match_type.value;
  const badgeCls = { EXACT_MATCH: "exact", NEAR_MATCH: "near", PAGE_MATCH: "page", NO_MATCH: "none" }[mtype] || "page";
  $("#steps").querySelector('[data-detail="2"]').innerHTML = `
    <div class="kv"><span class="k">Match type</span><span class="v"><span class="badge ${badgeCls}">${mtype}</span></span></div>
    <div class="kv"><span class="k">Source URL</span><span class="v mono">${esc(m.candidate.url)}</span></div>
    <div class="kv"><span class="k">Title</span><span class="v">${esc(m.candidate.title || "—")}</span></div>
    <div class="rationale">${esc(m.rationale)}</div>`;

  // [4] Fingerprint
  markStep(3, "done");
  const p = r.provenance || {};
  $("#steps").querySelector('[data-detail="3"]').innerHTML = `
    <div class="kv"><span class="k">SHA-256</span><span class="v">
      <span class="hash-copy">${r.fingerprint}<button class="copy-btn" data-copy="${r.fingerprint}">copy</button></span>
    </span></div>
    <div class="kv"><span class="k">Retrieved</span><span class="v mono">${esc(p.retrieved_at || "—")}</span></div>
    <div class="kv"><span class="k">Content SHA</span><span class="v mono">${esc((p.content_sha256 || "").slice(0, 20))}…</span></div>
    <div class="kv"><span class="k">Image SHA</span><span class="v mono">${esc((p.image_sha256 || "").slice(0, 20))}…</span></div>
    <div class="kv"><span class="k">Image pHash</span><span class="v mono">${esc(p.image_phash || "—")}</span></div>`;

  // [5] Blockchain
  markStep(4, r.chain ? "done" : "skipped");
  if (r.chain) {
    const c = r.chain;
    $("#steps").querySelector('[data-detail="4"]').innerHTML = `
      <div class="kv"><span class="k">Blockchain</span><span class="v">${esc(c.blockchain)}
        ${c.simulated ? '<span class="badge sim">SIMULATED</span>' : '<span class="badge exact">ON-CHAIN</span>'}</span></div>
      <div class="kv"><span class="k">Contract</span><span class="v mono">${esc(c.contract_address)}</span></div>
      <div class="kv"><span class="k">Transaction</span><span class="v">
        <span class="hash-copy">${esc(c.transaction_hash)}<button class="copy-btn" data-copy="${esc(c.transaction_hash)}">copy</button></span></span></div>
      <div class="kv"><span class="k">Block</span><span class="v mono">${c.block_number}</span></div>`;
  } else {
    $("#steps").querySelector('[data-detail="4"]').innerHTML =
      `<div class="rationale">Recording skipped for this run.</div>`;
  }

  // [6] Verification
  const v = r.verification;
  if (v) {
    markStep(5, v.verified ? "done" : "failed");
    $("#steps").querySelector('[data-detail="5"]').innerHTML = `
      <div class="kv"><span class="k">Calculated</span><span class="v mono">${esc(v.calculated_hash)}</span></div>
      <div class="kv"><span class="k">On-chain</span><span class="v mono">${esc(v.on_chain_hash || "(none)")}</span></div>
      <div class="kv"><span class="k">Tx</span><span class="v mono">${esc(v.transaction_hash || "(none)")}</span></div>`;
    showVerdict(
      v.verified ? "pass" : "fail",
      v.verified ? "VERIFIED" : "VERIFICATION FAILED",
      v.reason
    );
  }

  showJson(r);
  bindCopyButtons();
}

function markStep(index, cls) {
  const step = $(`[data-step="${index}"]`);
  if (!step) return;
  step.classList.remove("active");
  step.classList.add(cls);
}

function showVerdict(kind, title, reason) {
  const v = $("#verdict");
  v.className = "verdict " + kind;
  $("#verdict-icon").textContent = { pass: "✓", fail: "✕", warn: "!" }[kind];
  $("#verdict-title").textContent = title;
  $("#verdict-reason").textContent = reason;
  v.hidden = false;
}

function showJson(r) {
  const drawer = $("#json-drawer");
  $("#json-pre").textContent = JSON.stringify(r, null, 2);
  drawer.hidden = false;
}

function bindCopyButtons() {
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const text = btn.dataset.copy;
      try {
        await navigator.clipboard.writeText(text);
        const old = btn.textContent;
        btn.textContent = "copied";
        setTimeout(() => (btn.textContent = old), 1200);
      } catch (_) { /* clipboard unavailable */ }
    };
  });
}

/* ---------------------------------------------------------------- *
 *  Errors
 * ---------------------------------------------------------------- */
function handleError(err, chain) {
  const msg = String(err.message || err);
  let stepIdx = -1;
  if (/no face|face detected/i.test(msg)) stepIdx = 0;
  else if (/API key|SEARCH|provider|serper/i.test(msg)) stepIdx = 1;
  else if (/chain|anvil|rpc|blockchain/i.test(msg)) stepIdx = 4;

  if (stepIdx >= 0) {
    const step = $(`[data-step="${stepIdx}"]`);
    if (step) {
      step.classList.remove("active");
      step.classList.add("failed");
      const detail = step.querySelector(".step-detail");
      detail.innerHTML = `<div class="rationale" style="border-left-color:var(--bad)">${esc(msg)}</div>`;
    }
  }
  setStatus(msg, "err");
  $("#pipeline-sub").textContent = "Failed";
}

function setStatus(text, kind) {
  const el = $("#status-line");
  el.hidden = !text;
  el.className = "status-line " + (kind || "");
  el.textContent = text;
}

function setRunning(running) {
  $("#run-btn").disabled = running;
  $("#run-btn .btn-label").textContent = running ? "Running…" : "Run Pipeline";
  $("#run-btn .btn-spinner").hidden = !running;
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

init();
