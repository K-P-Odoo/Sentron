import os
print("Current working directory:", os.getcwd())

from flask import Flask, render_template, request, redirect, url_for, session, Response, send_file, jsonify
import cv2
import csv, time
from datetime import datetime
import threading

from Recognition import recognize_faces, get_recent_recognitions
# NEW: import your violence-frame function (see note above)
from Violence import detect_violence

app = Flask(__name__, template_folder='templates')
app.secret_key = 'sentron-secret-key'  # Change this in production

# Simulated user database
USER = {'admin': 'password123'}

# ================== MODE TOGGLE ==================
MODES = ("face", "violence")
_current_mode = "face"
_mode_lock = threading.Lock()

def get_mode():
    with _mode_lock:
        return _current_mode

def set_mode(new_mode: str) -> bool:
    global _current_mode
    if new_mode not in MODES:
        return False
    with _mode_lock:
        _current_mode = new_mode
    return True

# ================== THREAT LOGGING ==================
SNAP_DIR = os.path.join("static", "snapshots")
LOG_DIR  = "logs"
THREAT_LOG = os.path.join(LOG_DIR, "threats.csv")
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize CSV if missing
if not os.path.exists(THREAT_LOG):
    with open(THREAT_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "type", "image"])  # headers

THREAT_COOLDOWN = 5  # seconds
_last_threat_ts = 0.0

def log_threat(threat_type: str, frame_bgr):
    """Save snapshot + append CSV row (using the already-annotated frame)."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    save_path = os.path.join(SNAP_DIR, fname)
    try:
        cv2.imwrite(save_path, frame_bgr)
    except Exception as e:
        print("Snapshot save failed:", e)
        return
    rel_path = os.path.join("static", "snapshots", fname)
    with open(THREAT_LOG, "a", newline="") as f:
        csv.writer(f).writerow([ts_str, threat_type, rel_path])

# ========== VIDEO FEED ==========
camera = cv2.VideoCapture(0)

def gen_frames():
    global _last_threat_ts
    while True:
        success, frame = camera.read()
        if not success:
            break

        mode = get_mode()

        # Route to the selected pipeline
        labels = []
        if mode == "violence":
            # Must return (annotated_frame, labels)
            out, labels = detect_violence(frame)
        else:
            # Backward-compatible: user’s recognize_faces() may return frame or (frame, labels)
            r = recognize_faces(frame)
            if isinstance(r, tuple) and len(r) == 2:
                out, labels = r
            else:
                out = r
                labels = []

        # Auto-log threats on rising events with cooldown
        # Consider "FIGHT" or "ABUSE" as threats
        is_fight = any(isinstance(lbl, str) and (("FIGHT" in lbl.upper()) or ("ABUSE" in lbl.upper()))
                       for lbl in (labels or []))
        now = time.time()
        if mode == "violence" and is_fight and (now - _last_threat_ts) >= THREAT_COOLDOWN:
            log_threat("FIGHT", out)
            _last_threat_ts = now

        # Encode frame and stream
        ret, buffer = cv2.imencode('.jpg', out)
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

# ========== AUTH ==========
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','')
        password = request.form.get('password','')
        if USER.get(username) == password:
            session['user'] = username
            return redirect(url_for('home'))
        return render_template('index.html', error="Invalid credentials.")
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ========== MODE ENDPOINT (AJAX from home.html) ==========
@app.route('/mode', methods=['POST'])
def switch_mode():
    if 'user' not in session:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    new_mode = request.json.get('mode') if request.is_json else request.form.get('mode')
    ok = set_mode(new_mode)
    return jsonify({"ok": ok, "mode": get_mode()}), (200 if ok else 400)

# ========== PROTECTED ROUTES ==========
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    # Pass current mode so the toggle can highlight it
    return render_template('home.html', mode=get_mode(), modes=MODES)

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/threats')
def threats():
    if 'user' not in session:
        return redirect(url_for('login'))
    items = []
    if os.path.exists(THREAT_LOG):
        with open(THREAT_LOG, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append({
                    "time": row.get("time",""),
                    "type": row.get("type",""),
                    "image": row.get("image",""),
                })
    # newest first
    items = list(reversed(items))
    return render_template('threats.html', threats=items)

@app.route('/snapshots')
def snapshots():
    if 'user' not in session:
        return redirect(url_for('login'))

    snaps = []
    base_path = 'static/snapshots'

    if os.path.isdir(base_path):
        for filename in os.listdir(base_path):
            if filename.endswith(".jpg"):
                # Try to parse our saved filename patterns (YYYYmmdd_HHMMSS_micro.jpg)
                ts = filename.replace(".jpg", "").split("_", 2)[:2]
                try:
                    dt = datetime.strptime("_".join(ts), "%Y%m%d_%H%M%S")
                    date_str = dt.strftime("%d/%m/%Y")
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    date_str, time_str = "", ""
                snaps.append({
                    'filepath': filename,
                    'threat': "Fight",
                    'date': date_str,
                    'time': time_str
                })

    return render_template('snapshots.html', snapshots=snaps)

@app.route('/capture_snapshot/<threat>')
def capture_snapshot(threat):
    if 'user' not in session:
        return redirect(url_for('login'))

    os.makedirs(SNAP_DIR, exist_ok=True)
    success, frame = camera.read()
    if success:
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        filename = f"threat_{threat}_{timestamp}.jpg"
        filepath = os.path.join(SNAP_DIR, filename)
        cv2.imwrite(filepath, frame)
        return f"Snapshot saved: {filename}", 200
    else:
        return "Failed to capture frame", 500

# ========== RECENT RECOGNITIONS ==========
@app.route('/recent')
def recent():
    if 'user' not in session:
        return redirect(url_for('login'))
    recent_faces = get_recent_recognitions()
    return render_template('recent.html', recognitions=recent_faces)

# ========== DOWNLOAD FILE ==========
@app.route('/download_csv')
def download_csv():
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_file('logs/recognition_log.csv', as_attachment=True)

@app.route('/test')
def test():
    return render_template('index.html')

if __name__ == "__main__":
    try:
        app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
    finally:
        # Clean up camera on exit
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
