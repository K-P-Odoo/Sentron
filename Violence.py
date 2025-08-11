# violence.py  (refactor of 05_infer_unified_multiclass_fast_v8_cpu.py)
# Adds a single-frame wrapper: detect_violence(frame) -> (annotated_frame, ["FIGHT"]|["NORMAL"])

import time, collections
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ---------- NEW: email imports ----------
import os
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
# ---------------------------------------

# ---------- Config ----------
OUT = Path("C:\\Users\\Kailasnath Pillai\\Desktop\\Sentron-2")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_INTERVAL_SEC = 0.10
DET_CONF_THR = 0.60      # person bbox conf (pose model)
IOU_TRACK_THR = 0.30
SEQ_LEN = 5
JOINT_CONF_THR = 0.40
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12

# Runtime flags (window/video writer are disabled in wrapper)
SHOW_WINDOW = False
SAVE_OUTPUT = False
SMOOTH_WIN   = 10
DECISION_THR = 0.50

# ---------- CPU speed knobs (YOLOv8) ----------
DEVICE = "cpu"           # stay on CPU
POSE_WEIGHTS    = "yolov8n-pose.pt"   # tiny pose
POSE_IMGSZ      = 288                 # smaller = faster
WEAPON_WEIGHTS  = "yolov8n.pt"        # tiny detector (COCO has "knife")
WEAPON_IMGSZ    = 224

WEAPON_CONF_THR = 0.28
WEAPON_LABELS   = {"knife","gun","pistol","revolver","rifle","firearm"}

REQUIRE_OVERLAP_WITH_PERSON = False   # set True if you only want weapons near people
WEAPON_EVERY_N = 5                    # run weapon det every N frames
CROP_WEAPON_TO_PERSON = True          # detect weapons on person crops (cheaper)
MAX_PERSONS = 6                       # limit max tracks/boxes per frame

MODEL_PATH = OUT / "model_bilstm.keras"

# ---------- NEW: email config (mirrors recognition.py) ----------
EMAIL_SENDER = 'sentron2025@gmail.com'
EMAIL_PASSWORD = 'vjisvmpflcgxipft'  # Gmail app password (same as recognition.py)
EMAIL_RECEIVERS = ['kilopar336699@gmail.com', 'safwann.mohiuddin@gmail.com']
LIVE_FEED_LINK = 'http://127.0.0.1:5000/video_feed'  # adjust if needed
EMAIL_COOLDOWN = timedelta(minutes=5)

# Ensure snapshots directory exists
(OUT / "static" / "snapshots").mkdir(parents=True, exist_ok=True)

# Cooldown tracker
_last_email_time_violence = None

def send_alert_email(image_path: str):
    """Send a violence/fight email alert with attached snapshot (PNG/JPG)."""
    global _last_email_time_violence
    now = datetime.now()
    if _last_email_time_violence and (now - _last_email_time_violence) < EMAIL_COOLDOWN:
        print("Violence email alert skipped due to cooldown.")
        return

    msg = EmailMessage()
    msg['Subject'] = '🚨 Fight Detected!'
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(EMAIL_RECEIVERS)
    msg.set_content(
        f"A fight was detected by Sentron.\n\n"
        f"📸 Snapshot attached.\n"
        f"🔗 Live Feed: {LIVE_FEED_LINK}\n\n"
        f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Guess subtype from extension (fallback to png)
    ext = os.path.splitext(image_path)[1].lower()
    subtype = 'png' if ext in ('.png', '') else ext.lstrip('.')

    try:
        with open(image_path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype=subtype,
                               filename=os.path.basename(image_path))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, msg.as_string())
        print(f"Violence email alert sent for {image_path}")
        _last_email_time_violence = now
    except Exception as e:
        print(f"Failed to send violence email: {e}")
# ---------------------------------------------------------------

# ================= Module-wide state (loaded once) =================
_initialized = False
pose = None
weapon_det = None
clf = None
names_map = {}

# Stateful buffers across frames (so LSTM sees sequences)
tracks = {}
buffers = collections.defaultdict(lambda: collections.deque(maxlen=SEQ_LEN))
probs_hist = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTH_WIN))
abuse_up = 0
abuse_down = 0
abuse_active = False
frame_idx = 0

# ---------- Helpers from your script ----------
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    if inter<=0: return 0.0
    areaA=(a[2]-a[0])*(a[3]-a[1]); areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-6)

def iou_tracker(prev, boxes, thr=0.3):
    assigned, used, new = {}, set(), {}
    for tid, last in prev.items():
        best_j, best = -1, 0.0
        for j,b in enumerate(boxes):
            if j in used: continue
            v=iou(last,b)
            if v>best: best, best_j=v, j
        if best>=thr and best_j!=-1:
            new[tid]=boxes[best_j]; assigned[best_j]=tid; used.add(best_j)
    next_tid = (max(prev.keys())+1) if prev else 0
    for j,b in enumerate(boxes):
        if j not in assigned:
            new[next_tid]=b; assigned[j]=next_tid; next_tid+=1
    return new, assigned

def norm_kpts_by_box(kxy, box):
    x1,y1,x2,y2 = box
    w=max(1.0,x2-x1); h=max(1.0,y2-y1)
    cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
    k=kxy.astype(np.float32).copy()
    k[:,0]=(k[:,0]-cx)/w; k[:,1]=(k[:,1]-cy)/h
    return k  # (17,2)

def angle_3pts(a,b,c):
    bax=a[0]-b[0]; bay=a[1]-b[1]
    bcx=c[0]-b[0]; bcy=c[1]-b[1]
    nb=(np.hypot(bax,bay)*np.hypot(bcx,bcy))+1e-6
    cosv=(bax*bcx+bay*bcy)/nb
    cosv=np.clip(cosv,-1.0,1.0)
    return np.arccos(cosv).astype(np.float32)

KP={"LS":5,"RS":6,"LE":7,"RE":8,"LW":9,"RW":10,"LH":11,"RH":12,"LK":13,"RK":14,"LA":15,"RA":16}
def ang_feats(k):
    a0=angle_3pts(k[KP["LS"]],k[KP["LE"]],k[KP["LW"]])
    a1=angle_3pts(k[KP["RS"]],k[KP["RE"]],k[KP["RW"]])
    a2=angle_3pts(k[KP["LH"]],k[KP["LS"]],k[KP["LW"]])
    a3=angle_3pts(k[KP["RH"]],k[KP["RS"]],k[KP["RW"]])
    a4=angle_3pts(k[KP["LH"]],k[KP["LK"]],k[KP["LA"]])
    a5=angle_3pts(k[KP["RH"]],k[KP["RK"]],k[KP["RA"]])
    return np.array([a0,a1,a2,a3,a4,a5],dtype=np.float32)

def build_feats_from_buffers(k_seq):
    Kseq=np.stack(k_seq)                       # (T,17,2)
    P=Kseq.reshape(len(k_seq), -1)             # (T,34)
    Vpos=np.zeros_like(P); Vpos[1]=0
    Vpos[1:] = P[1:] - P[:-1]
    A=np.stack([ang_feats(Kseq[t]) for t in range(len(k_seq))])  # (T,6)
    Vang=np.zeros_like(A); Vang[1:] = A[1:] - A[:-1]
    F=np.concatenate([P,Vpos,A,Vang],axis=1).astype(np.float32)  # (T,80)
    return F

def get_names_map(ultra_model):
    names = getattr(ultra_model, "names", None) or getattr(ultra_model.model, "names", None)
    if isinstance(names, dict): return {int(i): str(n).lower() for i,n in names.items()}
    if isinstance(names, (list, tuple)): return {i: str(n).lower() for i,n in enumerate(names)}
    return {}

def any_weapon_from_results(res, names_map, conf_thr=0.28):
    if not res or res[0].boxes is None: return False, []
    r = res[0]
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()
    hits = []
    for i,c in enumerate(cls):
        name = names_map.get(int(c), "")
        if conf[i] >= conf_thr and name in WEAPON_LABELS:
            hits.append((name, float(conf[i]), xyxy[i]))
    return (len(hits) > 0), hits

def expand_and_clip(box, w, h, scale=1.2):
    x1,y1,x2,y2 = box
    cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
    bw=(x2-x1)*scale; bh=(y2-y1)*scale
    nx1=max(0,int(cx-bw/2)); ny1=max(0,int(cy-bh/2))
    nx2=min(w,int(cx+bw/2)); ny2=min(h,intcy+bh/2))
    return nx1,ny1,nx2,ny2

# ================= Init/load once =================
def _init_models():
    global _initialized, pose, weapon_det, clf, names_map
    if _initialized:
        return
    pose = YOLO(POSE_WEIGHTS)
    weapon_det = YOLO(WEAPON_WEIGHTS)
    clf  = tf.keras.models.load_model(MODEL_PATH)
    names_map = get_names_map(weapon_det)
    _initialized = True

# ===================== Wrapper =====================
def detect_violence(frame):
    """
    Single-frame inference for Flask.
    Input:  frame (BGR, np.ndarray)
    Output: annotated_frame (BGR), labels (["FIGHT"]|["NORMAL"] or ["ABUSE"])
    """
    _init_models()
    global tracks, buffers, probs_hist, abuse_up, abuse_down, abuse_active, frame_idx

    h, w = frame.shape[:2]

    # --- Pose pass ---
    r_pose = pose(frame, imgsz=POSE_IMGSZ, device=DEVICE, verbose=False, conf=DET_CONF_THR, max_det=MAX_PERSONS)
    person_boxes, kpts, kconfs = [], [], []
    for r in r_pose:
        if r.boxes is None or r.keypoints is None: continue
        conf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        kxy  = r.keypoints.xy.cpu().numpy()
        kc   = r.keypoints.conf.cpu().numpy()
        order = np.argsort(-conf)[:MAX_PERSONS]
        for i in order:
            if conf[i] < DET_CONF_THR: continue
            person_boxes.append(xyxy[i].tolist()); kpts.append(kxy[i]); kconfs.append(kc[i])

    # --- Track ---
    tracks, assignment = iou_tracker(tracks, person_boxes, IOU_TRACK_THR)

    # --- Weapon pass (every N frames; crops if persons exist) ---
    weapon_seen = False
    weapon_hits = []
    if frame_idx % WEAPON_EVERY_N == 0:
        if CROP_WEAPON_TO_PERSON and len(person_boxes) > 0:
            crops, crop_meta = [], []
            for pb in person_boxes:
                nx1,ny1,nx2,ny2 = expand_and_clip(pb, w, h, scale=1.25)
                crop = frame[ny1:ny2, nx1:nx2]
                if crop.size == 0: continue
                crops.append(crop); crop_meta.append((nx1,ny1,nx2,ny2))
            if crops:
                wr = weapon_det(crops, imgsz=WEAPON_IMGSZ, device=DEVICE, verbose=False,
                                conf=WEAPON_CONF_THR, max_det=6)
                for res, meta in zip(wr, crop_meta):
                    seen, hits = any_weapon_from_results([res], names_map, conf_thr=WEAPON_CONF_THR)
                    if hits:
                        weapon_seen = True
                        for name, confv, xyxy in hits:
                            x1,y1,x2,y2 = xyxy
                            ox1,oy1,_,_ = meta
                            weapon_hits.append((name, confv, np.array([x1+ox1,y1+oy1,x2+ox1,y2+oy1])))
                        break
        else:
            wr = weapon_det(frame, imgsz=WEAPON_IMGSZ, device=DEVICE, verbose=False,
                            conf=WEAPON_CONF_THR, max_det=12)
            weapon_seen, weapon_hits = any_weapon_from_results(wr, names_map, conf_thr=WEAPON_CONF_THR)

        if REQUIRE_OVERLAP_WITH_PERSON and weapon_seen and len(person_boxes):
            filtered = False
            for _,_,wb in weapon_hits:
                for pb in person_boxes:
                    x1=max(wb[0],pb[0]); y1=max(wb[1],pb[1]); x2=min(wb[2],pb[2]); y2=min(wb[3],pb[3])
                    inter=max(0,x2-x1)*max(0,y2-y1)
                    if inter>0: filtered = True; break
                if filtered: break
            weapon_seen = filtered

    # ABUSE hysteresis
    if weapon_seen: abuse_up += 1; abuse_down = 0
    else:           abuse_down += 1; abuse_up = 0
    if not abuse_active and abuse_up >= 2: abuse_active = True
    if abuse_active and abuse_down >= 6:   abuse_active = False

    # --- Update pose buffers ---
    for j, box in enumerate(person_boxes):
        tid = assignment[j]
        k_norm = norm_kpts_by_box(kpts[j], box)
        c = kconfs[j]
        anchors_ok = (c[L_SH] >= JOINT_CONF_THR and c[R_SH] >= JOINT_CONF_THR and
                      c[L_HIP] >= JOINT_CONF_THR and c[R_HIP] >= JOINT_CONF_THR)
        if not anchors_ok and len(buffers[tid]):
            k_norm = 0.9 * buffers[tid][-1] + 0.1 * k_norm
        buffers[tid].append(k_norm)

    # --- Draw & per-person classification ---
    overlay = frame.copy()
    for name, confv, wb in weapon_hits[:6]:
        x1,y1,x2,y2 = map(int, wb)
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(overlay, f"{name} {confv:.2f}", (x1, max(20, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

    any_fight = False
    for j, box in enumerate(person_boxes):
        tid = assignment[j]
        x1,y1,x2,y2 = [int(v) for v in box]
        color=(0,255,255); text="…"
        if abuse_active:
            color=(0,0,255); text="ABUSE"; any_fight = True
        elif len(buffers[tid]) >= SEQ_LEN:
            F = build_feats_from_buffers(list(buffers[tid]))
            X = F[np.newaxis, ...]
            p = float(clf.predict(X, verbose=0).ravel()[0])
            probs_hist[tid].append(p)
            pm = float(np.mean(probs_hist[tid]))
            label = "FIGHT" if pm >= DECISION_THR else "NORMAL"
            color = (0,0,255) if label=="FIGHT" else (0,255,0)
            text = f"ID {tid} {label} {pm:.2f}"
            if label == "FIGHT": any_fight = True
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        cv2.putText(overlay, text, (x1, max(20, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if abuse_active:
        cv2.putText(overlay, "ABUSE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

    annotated = cv2.addWeighted(overlay, 0.95, frame, 0.05, 0)

    frame_idx += 1

    # ---------- NEW: email on fight (save annotated snapshot, then email) ----------
    if abuse_active or any_fight:
        try:
            now = datetime.now()
            date_folder = now.strftime('%Y-%m-%d')
            folder_path = OUT / "static" / "snapshots" / date_folder
            folder_path.mkdir(parents=True, exist_ok=True)
            filename = f"threat_Fight_{now.strftime('%H%M%S')}.png"
            filepath = folder_path / filename
            cv2.imwrite(str(filepath), annotated)
            send_alert_email(str(filepath))
        except Exception as e:
            print("Failed to save/email fight snapshot:", e)
        return annotated, ["FIGHT"]
    # ------------------------------------------------------------------------------

    else:
        return annotated, ["NORMAL"]

# ===================== Standalone mode =====================
# You can still run: python violence.py
def _open_source(SOURCE):
    # Keep your robust open logic
    for api in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(SOURCE, api)
        if cap.isOpened():
            return cap, True
    raise SystemExit(f"❌ Could not open webcam index {SOURCE}.")

def main():
    _init_models()
    SOURCE = 0  # webcam
    cap, _ = _open_source(SOURCE)
    last_t = 0.0
    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if (now - last_t) < FRAME_INTERVAL_SEC:
                continue
            last_t = now

            annotated, labels = detect_violence(frame)

            if SHOW_WINDOW:
                cv2.imshow("Violence Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            if SAVE_OUTPUT:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(OUT/"standalone_out.mp4"), fourcc, 25.0, (annotated.shape[1], annotated.shape[0]))
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
