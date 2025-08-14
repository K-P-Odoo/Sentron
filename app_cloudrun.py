# app_cloudrun.py
import os
import json
import time
import threading
from datetime import datetime

import numpy as np
import cv2
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, Response, jsonify
)
from google.cloud import storage

# ================== PIPELINES ==================
# Both should accept BGR np.array and return:
# - face: either frame OR (frame, labels)
# - violence: (annotated_frame, labels)
from Recognition import recognize_faces, get_recent_recognitions

# ---- GRACEFUL violence import (prevents boot crash if TF/model missing) ----
try:
    from Violence import detect_violence as _dv
    VIOLENCE_AVAILABLE = True
except Exception as e:
    print("[BOOT] Violence model disabled:", e)
    VIOLENCE_AVAILABLE = False

    def _dv(frame):
        # passthrough + visible label so UI still works
        return frame, ["VIOLENCE_MODEL_UNAVAILABLE"]

detect_violence = _dv

# ================== FIXED DEFAULTS FOR YOUR DEPLOY ==================
DEFAULT_APP_SECRET   = "sentron-secret-key"  # change in prod if you wish
DEFAULT_ADMIN_PASS   = "password123"         # change in prod if you wish
DEFAULT_GCS_BUCKET   = "sentron-demo-data"
DEFAULT_GCS_PREFIX   = "sentron"             # becomes sentron/ in the bucket
DEFAULT_INGEST_TOKEN = "b8c6b3f5d3f94a4a9ccfbb0a0b3f8a3d4b7f2a1c9e6d4f3ab1c2d3e4f5a6b7c8"

# ================== FLASK ==================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv('APP_SECRET', DEFAULT_APP_SECRET)

# ================== SIMPLE AUTH (UI) ==================
USER = {'admin': os.getenv('ADMIN_PASSWORD', DEFAULT_ADMIN_PASS)}

# ================== INGEST AUTH (Pi -> Cloud) ==================
INGEST_TOKEN = os.getenv("INGEST_TOKEN", DEFAULT_INGEST_TOKEN)
def _auth_ingest(req) -> bool:
    auth = req.headers.get("Authorization", "")
    return bool(INGEST_TOKEN) and auth == f"Bearer {INGEST_TOKEN}"

# ================== GCS CONFIG ==================
GCS_BUCKET = os.getenv("GCS_BUCKET", DEFAULT_GCS_BUCKET)
GCS_PREFIX = os.getenv("GCS_PREFIX", DEFAULT_GCS_PREFIX).strip()
if GCS_PREFIX and not GCS_PREFIX.endswith("/"):
    GCS_PREFIX += "/"

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)

def gcs_path(*parts) -> str:
    """Join parts under configured prefix."""
    return GCS_PREFIX + "/".join(p.strip("/\\") for p in parts if p)

def upload_image_to_gcs(img_bgr: np.ndarray, path: str, jpeg_quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    blob = bucket.blob(path)
    blob.cache_control = "public, max-age=31536000"
    blob.upload_from_string(buf.tobytes(), content_type="image/jpeg")
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{path}"

def upload_json_to_gcs(obj: dict, path: str):
    blob = bucket.blob(path)
    blob.cache_control = "no-store"
    blob.upload_from_string(json.dumps(obj, ensure_ascii=False), content_type="application/json")

# ================== MODE TOGGLE + STREAM EPOCH ==================
MODES = ("face", "violence")
_current_mode = "face"
_mode_lock = threading.Lock()

def get_mode() -> str:
    with _mode_lock:
        return _current_mode

def set_mode(new_mode: str) -> bool:
    global _current_mode
    if new_mode not in MODES:
        return False
    with _mode_lock:
        if _current_mode != new_mode:
            print(f"[MODE] {datetime.utcnow():%Y-%m-%d %H:%M:%S} -> {new_mode}")
        _current_mode = new_mode
    return True

_stream_epoch = 0
_epoch_lock = threading.Lock()
def get_epoch() -> int:
    return _stream_epoch
def bump_epoch() -> int:
    global _stream_epoch
    with _epoch_lock:
        _stream_epoch += 1
        print(f"[STREAM] epoch -> #{_stream_epoch}")
        return _stream_epoch

# ================== FRAME INGEST (Pi pushes here) ==================
_latest_frame = None
_latest_lock = threading.Lock()

@app.post("/ingest")
def ingest():
    """Pi POSTs: files={'image': ('frame.jpg', jpg_bytes, 'image/jpeg')} with Authorization: Bearer <INGEST_TOKEN>"""
    if not _auth_ingest(request):
        return "unauthorized", 401
    if 'image' not in request.files:
        return "no image", 400
    data = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "bad image", 400
    with _latest_lock:
        global _latest_frame
        _latest_frame = img
    return "ok", 200

# ================== THREAT LOGGING (GCS) ==================
THREAT_COOLDOWN = 5.0  # seconds
_last_threat_ts = 0.0

def log_threat(threat_type: str, annotated_bgr: np.ndarray):
    """
    Save snapshot + JSON metadata to GCS.
    Layout:
      snapshots/YYYY-MM-DD/threat_<TYPE>_<YYYYmmdd_HHMMSS_us>.jpg
      logs/YYYY-MM-DD/threat_<TYPE>_<YYYYmmdd_HHMMSS_us>.json
    """
    ts = datetime.utcnow()
    day = ts.strftime("%Y-%m-%d")
    stamp = ts.strftime("%Y%m%d_%H%M%S_%f")
    base = f"threat_{threat_type.lower()}_{stamp}"

    img_path  = gcs_path("snapshots", day, f"{base}.jpg")
    meta_path = gcs_path("logs",      day, f"{base}.json")

    url = upload_image_to_gcs(annotated_bgr, img_path, jpeg_quality=70)
    meta = {
        "time_utc": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "type": threat_type,
        "image_gcs": img_path,
        "image_url": url
    }
    upload_json_to_gcs(meta, meta_path)
    print(f"[THREAT] {threat_type} -> {img_path}")

# ================== STREAMING GENERATOR ==================
def gen_frames():
    """Read latest ingested frame, run selected pipeline, stream MJPEG."""
    global _last_threat_ts
    my_epoch = get_epoch()

    while True:
        if my_epoch != get_epoch():
            print("[STREAM] generator exit (epoch changed)")
            break

        with _latest_lock:
            src = None if _latest_frame is None else _latest_frame.copy()

        if src is None:
            time.sleep(0.05)
            continue

        mode = get_mode()
        labels = []

        if mode == "violence":
            try:
                out, labels = detect_violence(src)
            except Exception as e:
                print("[VIOLENCE] error:", e)
                out, labels = src, []
        else:
            r = recognize_faces(src)
            if isinstance(r, tuple) and len(r) == 2:
                out, labels = r
            else:
                out, labels = r, []

        # Threat detection (rising event, cooldown)
        is_threat = any(
            isinstance(lbl, str) and (("FIGHT" in lbl.upper()) or ("ABUSE" in lbl.upper()))
            for lbl in (labels or [])
        )
        now = time.time()
        if mode == "violence" and is_threat and (now - _last_threat_ts) >= THREAT_COOLDOWN:
            try:
                log_threat("FIGHT", out)
            except Exception as e:
                print("[THREAT] log failed:", e)
            _last_threat_ts = now

        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ================== UI ROUTES ==================
@app.get("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username",""); p = request.form.get("password","")
        if USER.get(u) == p:
            session["user"] = u
            return redirect(url_for("home"))
        return render_template("index.html", error="Invalid credentials.")
    return render_template("index.html")

@app.get("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.post("/mode")
def switch_mode():
    if "user" not in session:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    new_mode = (request.json or {}).get("mode")
    ok = set_mode(new_mode)
    if ok:
        return jsonify({"ok": True, "mode": get_mode(), "epoch": bump_epoch()}), 200
    return jsonify({"ok": False, "mode": get_mode()}), 400

@app.get("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", mode=get_mode(), modes=MODES)

@app.get("/video_feed")
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    resp = Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"  # hint proxies to not buffer MJPEG
    return resp

@app.get("/threats")
def threats():
    if "user" not in session:
        return redirect(url_for("login"))
    items = []
    for blob in storage_client.list_blobs(GCS_BUCKET, prefix=gcs_path("logs/")):
        if not blob.name.lower().endswith(".json"):
            continue
        try:
            data = json.loads(blob.download_as_bytes())
            items.append({
                "time":  data.get("time_utc",""),
                "type":  data.get("type",""),
                "image": data.get("image_url") or f"https://storage.googleapis.com/{GCS_BUCKET}/{data.get('image_gcs','')}"
            })
        except Exception as e:
            print("[THREATS] parse failed:", blob.name, e)
    items.sort(key=lambda x: x.get("time",""), reverse=True)
    return render_template("threats.html", threats=items)

@app.get("/snapshots")
def snapshots():
    if "user" not in session:
        return redirect(url_for("login"))
    snaps = []
    for blob in storage_client.list_blobs(GCS_BUCKET, prefix=gcs_path("snapshots/")):
        low = blob.name.lower()
        if not low.endswith((".jpg",".jpeg",".png")):
            continue
        threat = "Fight" if "fight" in low else ("Unknown user" if "unknown" in low else "Snapshot")
        parts = blob.name.split("/")
        date_hint = parts[-2] if len(parts)>=2 and len(parts[-2])==10 and parts[-2].count("-")==2 else ""
        snaps.append({
            "filepath": blob.name,
            "threat": threat,
            "date": date_hint,
            "time": "",
            "url": f"https://storage.googleapis.com/{GCS_BUCKET}/{blob.name}",
        })
    snaps.sort(key=lambda s: s["filepath"], reverse=True)
    return render_template("snapshots.html", snapshots=snaps)

@app.get("/capture_snapshot/<_>")
def capture_snapshot(_):
    """Manual snapshot of current ingested frame (uses current mode)."""
    if "user" not in session:
        return redirect(url_for("login"))
    with _latest_lock:
        src = None if _latest_frame is None else _latest_frame.copy()
    if src is None:
        return "No frame available", 503

    threat_type = "fight" if get_mode() == "violence" else "unknown"
    ts = datetime.utcnow()
    day = ts.strftime("%Y-%m-%d")
    stamp = ts.strftime("%Y%m%d_%H%M%S_%f")
    img_path = gcs_path("snapshots", day, f"threat_{threat_type}_{stamp}.jpg")
    url = upload_image_to_gcs(src, img_path, jpeg_quality=70)
    return f"Snapshot saved: {url}", 200

@app.get("/recent")
def recent():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("recent.html", recognitions=get_recent_recognitions())

# ---- simple health endpoint (for fast startup checks) ----
@app.get("/healthz")
def healthz():
    return "ok", 200

# ================== MAIN (dev only; Cloud Run uses gunicorn) ==================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
    