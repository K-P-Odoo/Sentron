# import pickle
# import cv2
# import face_recognition


# def recognize_debug(model_path="trained_knn_model.clf", distance_threshold=0.5):
#     with open(model_path, 'rb') as f:
#         knn_clf = pickle.load(f)

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("[ERROR] Cannot access webcam.")
#         return

#     print("[INFO] Webcam started. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame.")
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)

#         if face_locations:
#             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#             closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
#             is_recognized = [dist[0] <= distance_threshold for dist in closest_distances[0]]
#             predictions = knn_clf.predict(face_encodings)

#             for i, (top, right, bottom, left) in enumerate(face_locations):
#                 name = predictions[i] if is_recognized[i] else "Unknown"
#                 confidence = 1 - closest_distances[0][i][0]

#                 # Draw rectangle and name
#                 color = (0, 255, 0) if is_recognized[i] else (0, 0, 255)
#                 label = f"{name} ({confidence:.2f})"

#                 cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#                 cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#                 print(f"[INFO] Prediction: {name}, Distance: {closest_distances[0][i][0]:.4f}")

#         cv2.imshow("Face Recognition Debug", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("[INFO] Exiting...")
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     recognize_debug()


# recognizer.py

import pickle
import face_recognition
import cv2
import os
from datetime import datetime

# Load KNN model
with open("trained_knn_model.clf", 'rb') as f:
    knn_clf = pickle.load(f)

# Create log and snapshot folders
os.makedirs("logs", exist_ok=True)
os.makedirs("static/snapshots", exist_ok=True)

# Path to log file
log_path = os.path.join("logs", "recognition_log.csv")

# Initialize log if empty
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("name,confidence,timestamp\n")

# Keep a buffer of recent recognitions
recent_recognitions = []

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

        # Draw on frame
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save snapshot if unknown
        if name == "Unknown":
            snap_filename = f"threat_Unknown_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join("static/snapshots", snap_filename), frame)

        # Log to CSV
        with open(log_path, 'a') as f:
            f.write(f"{name},{confidence:.2f},{timestamp}\n")

        # Store recent (max 10)
        recent_recognitions.append({"name": name, "confidence": f"{confidence:.2f}", "time": timestamp})
        recent_recognitions = recent_recognitions[-10:]  # keep last 10 only

    return frame

def get_recent_recognitions():
    return list(recent_recognitions)

