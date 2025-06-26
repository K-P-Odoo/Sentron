import pickle
import cv2
import face_recognition


def recognize_debug(model_path="trained_knn_model.clf", distance_threshold=0.5):
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
            is_recognized = [dist[0] <= distance_threshold for dist in closest_distances[0]]
            predictions = knn_clf.predict(face_encodings)

            for i, (top, right, bottom, left) in enumerate(face_locations):
                name = predictions[i] if is_recognized[i] else "Unknown"
                confidence = 1 - closest_distances[0][i][0]

                # Draw rectangle and name
                color = (0, 255, 0) if is_recognized[i] else (0, 0, 255)
                label = f"{name} ({confidence:.2f})"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                print(f"[INFO] Prediction: {name}, Distance: {closest_distances[0][i][0]:.4f}")

        cv2.imshow("Face Recognition Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_debug()
