import pickle
import face_recognition
import cv2
import os
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# ========== CONFIG ==========
EMAIL_SENDER = 'sentron2025@gmail.com'
EMAIL_PASSWORD = 'vjisvmpflcgxipft'  # Use an app password for Gmail
EMAIL_RECEIVERS = ['kilopar336699@gmail.com', 'safwann.mohiuddin@gmail.com']
LIVE_FEED_LINK = 'http://127.0.0.1:5000/video_feed'  # Replace with actual IP or public domain
EMAIL_COOLDOWN = timedelta(minutes=5)  # Minimum interval between alert emails

# Load KNN model
with open("trained_knn_model.clf", 'rb') as f:
    knn_clf = pickle.load(f)

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("static/snapshots", exist_ok=True)

# Path to recognition log
log_path = os.path.join("logs", "recognition_log.csv")

# Initialize log if not present
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("name,confidence,timestamp\n")

# Globals
recent_recognitions = []
last_email_time = None


def send_alert_email(image_path):
    global last_email_time
    now = datetime.now()

    if last_email_time and (now - last_email_time) < EMAIL_COOLDOWN:
        print("Email alert skipped due to cooldown.")
        return

    msg = EmailMessage()
    msg['Subject'] = '🚨 Unknown Face Detected!'
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(EMAIL_RECEIVERS)
    msg.set_content(
        f"An unknown individual was detected.\n\n📸 Snapshot attached.\n🔗 Live Feed: {LIVE_FEED_LINK}\n\nTime: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    with open(image_path, 'rb') as img:
        img_data = img.read()
        msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, msg.as_string())
            print(f"Email alert sent for {image_path}")
            last_email_time = now
    except Exception as e:
        print(f"Failed to send email: {e}")


def recognize_faces(frame, distance_threshold=0.5):
    global recent_recognitions

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        return frame

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    is_recognized = [dist[0] <= distance_threshold for dist in closest_distances[0]]
    predictions = knn_clf.predict(face_encodings)

    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

    for i, (top, right, bottom, left) in enumerate(face_locations):
        name = predictions[i] if is_recognized[i] else "Unknown"
        confidence = 1 - closest_distances[0][i][0]
        color = (0, 255, 0) if is_recognized[i] else (0, 0, 255)
        label = f"{name} ({confidence:.2f})"

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if name == "Unknown":
            date_folder = now.strftime('%Y-%m-%d')
            folder_path = os.path.join("static/snapshots", date_folder)
            os.makedirs(folder_path, exist_ok=True)

            # Crop and enhance the face
            face_crop = frame[top:bottom, left:right]
            face_crop = cv2.resize(face_crop, (200, 200))
            face_yuv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YUV)
            face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
            face_crop = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)

            filename = f"threat_Unknown_{now.strftime('%H%M%S')}.png"
            filepath = os.path.join(folder_path, filename)
            cv2.imwrite(filepath, face_crop)

            send_alert_email(filepath)

        # Log to CSV
        with open(log_path, 'a') as f:
            f.write(f"{name},{confidence:.2f},{timestamp}\n")

        recent_recognitions.append({
            "name": name,
            "confidence": f"{confidence:.2f}",
            "time": timestamp
        })
        recent_recognitions = recent_recognitions[-10:]

    return frame


def get_recent_recognitions():
    return list(recent_recognitions)
