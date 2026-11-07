(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const state = { style: "none", candidates: 1, info: null, last: null, shown: 0, busy: false };

  /* ---------------- server status ---------------- */
  async function pollHealth() {
    try {
      const r = await fetch("/api/health");
      const info = await r.json();
      state.info = info;
      const s = $("status");
      s.dataset.s = info.status;
      if (info.status === "ready") {
        $("statusText").textContent = info.demo ? "Demo mode — no model loaded" : `${info.model} on ${info.device.toUpperCase()}`;
        applyModelLimits(info);
        return;
      }
      $("statusText").textContent = info.status === "loading" ? "Loading model…" : "Model failed to load";
      if (info.status === "error") showError(info.error);
    } catch {
      $("status").dataset.s = "error";
      $("statusText").textContent = "Server not reachable";
    }
    setTimeout(pollHealth, 4000);
  }

  function applyModelLimits(info) {
    const steps = $("steps");
    steps.max = info.max_steps;
    steps.value = info.default_steps;
    $("stepsOut").textContent = info.default_steps;
    $("guidance").value = info.default_guidance;
    $("guideOut").textContent = info.default_guidance;
    $("negWrap").hidden = !info.supports_negative;
    $("guideWrap").hidden = !info.supports_negative;
  }

  /* ---------------- styles ---------------- */
  async function loadStyles() {
    let list = [{ key: "none", label: "No style" }];
    try { list = await (await fetch("/api/styles")).json(); } catch {}
    const box = $("styles");
    list.forEach(({ key, label }) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.textContent = label;
      b.setAttribute("aria-pressed", key === state.style ? "true" : "false");
      b.addEventListener("click", () => {
        state.style = key;
        box.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
      });
      box.appendChild(b);
    });
  }

  /* ---------------- form bits ---------------- */
  const prompt = $("prompt");
  prompt.addEventListener("input", () => {
    $("count").textContent = prompt.value.length;
    const words = prompt.value.trim().split(/\s+/).filter(Boolean).length;
    $("longHint").hidden = words <= 50;
  });
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("form").requestSubmit(); }
  });
  document.querySelectorAll(".idea").forEach((b) =>
    b.addEventListener("click", () => {
      prompt.value = b.textContent; prompt.dispatchEvent(new Event("input")); prompt.focus();
    })
  );
  $("cand").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state.candidates = +b.dataset.v;
    $("cand").querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", x === b ? "true" : "false"));
  });
  $("steps").addEventListener("input", (e) => ($("stepsOut").textContent = e.target.value));
  $("guidance").addEventListener("input", (e) => ($("guideOut").textContent = e.target.value));

  /* ---------------- noise animation (the "undeveloped" image) ---------------- */
  const cv = $("noise"), ctx = cv.getContext("2d");
  const small = document.createElement("canvas"); small.width = small.height = 128;
  const sctx = small.getContext("2d");
  let raf = null, t0 = 0;

  function drawNoise(level) {
    const img = sctx.createImageData(128, 128), d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = Math.random() * 255 * level + 40 * (1 - level);
      d[i] = v * 0.85 + 20; d[i + 1] = v * 0.9 + 22; d[i + 2] = v + 40; d[i + 3] = 255;
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(small, 0, 0, cv.width, cv.height);
  }
  function startNoise() {
    stopNoise(); t0 = performance.now();
    const tick = (now) => {
      const sec = (now - t0) / 1000;
      $("timer").textContent = sec.toFixed(1);
      // static slowly calms down while we wait
      drawNoise(Math.max(0.35, 1 - sec / 40));
      raf = reduceMotion ? setTimeout(() => tick(performance.now()), 500) : requestAnimationFrame(tick);
    };
    tick(t0);
  }
  function stopNoise() {
    if (raf) { reduceMotion ? clearTimeout(raf) : cancelAnimationFrame(raf); raf = null; }
  }
  drawNoise(0.5);

  /* ---------------- generate ---------------- */
  function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state.busy) return;
    const text = prompt.value.trim();
    if (!text) { showError("Describe the image you want first."); prompt.focus(); return; }
    showError("");
    state.busy = true;
    const btn = $("paint"); btn.disabled = true; btn.textContent = "Painting…";
    const wrap = $("canvasWrap");
    wrap.classList.remove("done");
    $("empty").hidden = true; $("progress").hidden = false;
    $("meta").hidden = true; $("cands").hidden = true;
    startNoise();

    const seedVal = $("seed").value.trim();
    const body = {
      prompt: text,
      negative_prompt: $("negative").value,
      style: state.style,
      candidates: state.candidates,
      width: +$("width").value,
      height: +$("height").value,
      steps: +$("steps").value,
      guidance: +$("guidance").value,
      seed: seedVal === "" ? null : +seedVal,
    };
    try {
      const r = await fetch("/api/generate", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Painting failed.");
      state.last = { ...data, prompt: text };
      render(data.best_index, true);
      renderCandidates();
      saveToGallery(data.images[data.best_index], text);
    } catch (err) {
      showError(err.message === "Failed to fetch" ? "Can't reach the server. Check that app.py is running." : err.message);
      $("progress").hidden = true; $("empty").hidden = false;
      $("empty").textContent = "Nothing painted yet.";
      drawNoise(0.5);
    } finally {
      stopNoise();
      state.busy = false; btn.disabled = false; btn.textContent = "Paint it";
    }
  });

  function describeScore(s) {
    if (s == null) return "Not available in demo mode.";
    if (s >= 32) return "Strong match with your description.";
    if (s >= 27) return "Good match with your description.";
    if (s >= 22) return "Partial match. Try more specific wording.";
    return "Weak match. Rephrase or try another seed.";
  }

  function render(i, fresh) {
    const d = state.last, im = d.images[i];
    state.shown = i;
    const url = "data:image/png;base64," + im.png_b64;
    const img = $("result"), wrap = $("canvasWrap");
    const show = () => {
      img.hidden = false; $("progress").hidden = true;
      requestAnimationFrame(() => wrap.classList.add("done"));
    };
    if (fresh) { wrap.classList.remove("done"); }
    img.onload = show;
    img.src = url;
    img.alt = d.prompt;
    $("meta").hidden = false;
    $("scoreNum").hidden = im.flagged || im.clip_score == null;
    $("scoreNum").textContent = im.clip_score == null ? "" : im.clip_score.toFixed(1);
    $("scoreTxt").textContent = im.flagged ? "Hidden by the safety checker. Try a different prompt." : describeScore(im.clip_score);
    const dl = $("download");
    dl.href = url;
    dl.download = "dreamcanvas-" + d.prompt.slice(0, 40).replace(/[^a-z0-9]+/gi, "-").toLowerCase() + "-" + im.seed + ".png";
    const warn = $("cutoff");
    warn.hidden = !d.ignored_text;
    if (d.ignored_text) warn.textContent = `Your description was too long for the model, so it ignored the end: "…${d.ignored_text}". Put the most important details first, or shorten it to about 50 words.`;
    $("fine").textContent = `${d.model} · seed ${im.seed} · ${d.elapsed}s` + (d.demo ? " · demo image, start the server without GENAI_DEMO for real results" : "");
    document.querySelectorAll(".cand").forEach((c, k) => c.setAttribute("aria-current", k === i ? "true" : "false"));
  }

  function renderCandidates() {
    const d = state.last, box = $("cands");
    box.innerHTML = "";
    if (d.images.length < 2) { box.hidden = true; return; }
    d.images.forEach((im, k) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "cand";
      b.setAttribute("aria-label", `Show candidate ${k + 1}`);
      b.setAttribute("aria-current", k === d.best_index ? "true" : "false");
      const label = im.clip_score == null ? `Seed ${im.seed}` : `${im.clip_score.toFixed(1)}`;
      b.innerHTML = `<img alt="" src="data:image/png;base64,${im.png_b64}"><span class="${k === d.best_index ? "best" : ""}">${label}${k === d.best_index && im.clip_score != null ? " · best" : ""}</span>`;
      b.addEventListener("click", () => render(k, false));
      box.appendChild(b);
    });
    box.hidden = false;
  }

  $("reuse").addEventListener("click", () => {
    if (!state.last) return;
    const adv = document.querySelector(".advanced"); adv.open = true;
    $("seed").value = state.last.images[state.shown].seed;
    $("seed").focus();
  });

  /* ---------------- gallery (kept in this browser) ---------------- */
  const KEY = "dreamcanvas.gallery.v1";
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
  const store = (arr) => { try { localStorage.setItem(KEY, JSON.stringify(arr)); } catch {} };

  function saveToGallery(im, text) {
    if (im.flagged) return;
    const src = new Image();
    src.onload = () => {
      const c = document.createElement("canvas"); c.width = c.height = 256;
      const s = Math.min(src.width, src.height);
      c.getContext("2d").drawImage(src, (src.width - s) / 2, (src.height - s) / 2, s, s, 0, 0, 256, 256);
      const arr = load();
      arr.unshift({ thumb: c.toDataURL("image/jpeg", 0.8), prompt: text, seed: im.seed, score: im.clip_score });
      store(arr.slice(0, 24));
      renderGallery();
    };
    src.src = "data:image/png;base64," + im.png_b64;
  }

  function renderGallery() {
    const arr = load(), grid = $("galGrid");
    grid.innerHTML = "";
    $("galEmpty").hidden = arr.length > 0;
    $("clear").hidden = arr.length === 0;
    arr.forEach((g) => {
      const f = document.createElement("figure");
      const img = document.createElement("img");
      img.src = g.thumb; img.alt = g.prompt; img.tabIndex = 0;
      img.title = "Load this prompt and seed";
      const use = () => {
        prompt.value = g.prompt; prompt.dispatchEvent(new Event("input"));
        $("seed").value = g.seed; document.querySelector(".advanced").open = true;
        $("studio").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
        prompt.focus({ preventScroll: true });
      };
      img.addEventListener("click", use);
      img.addEventListener("keydown", (e) => { if (e.key === "Enter") use(); });
      const cap = document.createElement("figcaption");
      cap.textContent = (g.score != null ? g.score.toFixed(1) + " · " : "") + g.prompt;
      f.append(img, cap); grid.appendChild(f);
    });
  }
  $("clear").addEventListener("click", () => { store([]); renderGallery(); });

  loadStyles();
  renderGallery();
  pollHealth();
})();
