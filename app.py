"""
DreamCanvas — web server.

Run:
    python app.py                    # real model (needs GPU for good speed)
    GENAI_DEMO=1 python app.py       # demo mode, no ML libraries needed
    GENAI_MODEL=turbo python app.py  # fast SD-Turbo model (1-4 steps)

Then open http://localhost:7860
"""

import os
import threading

from flask import Flask, jsonify, request, send_from_directory

from engine import MODELS, STYLES, GenRequest, create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="/static")

engine = create_engine()
threading.Thread(target=engine.load, daemon=True).start()

MAX_PROMPT = 400
ALLOWED_SIZES = {256, 384, 512, 576, 640, 704, 768}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(engine.info())


@app.get("/api/styles")
def styles():
    return jsonify([{"key": k, "label": v["label"]} for k, v in STYLES.items()])


def _bad(msg, code=400):
    return jsonify({"error": msg}), code


@app.post("/api/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return _bad("Please enter a prompt.")
    if len(prompt) > MAX_PROMPT:
        return _bad(f"Prompt is too long (max {MAX_PROMPT} characters).")

    info = engine.info()
    if info["status"] == "loading":
        return _bad("The model is still loading — try again in a minute.", 503)
    if info["status"] == "error":
        return _bad(f"Model failed to load: {info['error']}", 500)

    try:
        width = int(data.get("width", 512))
        height = int(data.get("height", 512))
        if width not in ALLOWED_SIZES or height not in ALLOWED_SIZES:
            return _bad("Unsupported image size.")
        steps = data.get("steps")
        steps = None if steps in (None, "") else max(1, min(int(steps), info["max_steps"]))
        guidance = data.get("guidance")
        guidance = None if guidance in (None, "") else max(0.0, min(float(guidance), 20.0))
        seed = data.get("seed")
        seed = None if seed in (None, "", -1, "-1") else max(0, min(int(seed), 2**31 - 5))
        candidates = max(1, min(int(data.get("candidates", 1)), 4))
    except (TypeError, ValueError):
        return _bad("Invalid number in settings.")

    style = data.get("style", "none")
    if style not in STYLES:
        style = "none"

    req = GenRequest(
        prompt=prompt,
        negative_prompt=str(data.get("negative_prompt", ""))[:MAX_PROMPT],
        style=style,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
        candidates=candidates,
    )
    try:
        res = engine.generate(req)
    except Exception as e:  # e.g. CUDA out of memory
        msg = str(e)
        if "out of memory" in msg.lower():
            msg = "GPU ran out of memory — try a smaller size or fewer candidates."
        return _bad(msg, 500)

    return jsonify(res.__dict__)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"DreamCanvas running on http://localhost:{port}  (models: {', '.join(MODELS)})")
    app.run(host="0.0.0.0", port=port, threaded=True)
