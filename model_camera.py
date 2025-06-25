import cv2
import dlib
import pickle
import numpy as np

# Paths
MODEL_PATH = "face_recognizer_model.pkl"
LABELS_PATH = "label_names.pkl"

# Load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the trained model and labels
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(LABELS_PATH, "rb") as f:
    label_dict = pickle.load(f)


def recognize_face(frame):
    """Detect and recognize faces from a video frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = sp(gray, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        face_descriptor = np.array(face_descriptor).reshape(1, -1)

        # Predict person
        prediction = model.predict(face_descriptor)[0]

        # Compute confidence score (distance to nearest neighbor)
        distances, _ = model.kneighbors(face_descriptor)
        confidence = 100 - (distances[0][0] * 100)  # Convert distance to percentage
        confidence = max(0, min(confidence, 100))  # Clamp between 0-100%

        # If confidence is below 50%, label as "Other"
        if confidence < 50:
            person_name = "Other"
        else:
            person_name = label_dict.get(prediction, "Unknown")

        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display name and confidence
        text = f"{person_name}: {confidence:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main():
    """Start live face recognition from webcam."""
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = recognize_face(frame)

        # Display the video feed
        cv2.imshow("Live Face Recognition", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
