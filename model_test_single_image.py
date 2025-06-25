import cv2
import dlib
import pickle
import numpy as np
import sys

# Paths
MODEL_PATH = "face_recognizer_model.pkl"
LABELS_PATH = "label_names.pkl"

# Load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def load_labels():
    """Load label dictionary from file."""
    with open(LABELS_PATH, "rb") as f:
        return pickle.load(f)


def predict_image(image_path):
    """Predict the person in an image using the trained model with confidence score."""
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    label_dict = load_labels()

    # Read and process the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No face detected in the image.")
        return

    # Take the first detected face
    face = faces[0]
    shape = sp(gray, face)
    face_descriptor = facerec.compute_face_descriptor(image, shape)

    # Convert descriptor to NumPy array
    face_descriptor = np.array(face_descriptor).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(face_descriptor)[0]

    # Compute confidence score (distance to nearest neighbor)
    distances, _ = model.kneighbors(face_descriptor)
    confidence = 100 - (distances[0][0] * 100)  # Convert distance to percentage

    # Ensure confidence is within 0-100%
    confidence = max(0, min(confidence, 100))

    person_name = label_dict.get(prediction, "Unknown")

    print(f"Prediction: {person_name}")
    print(f"Confidence: {confidence:.2f}%")

    if confidence < 50:
        print("⚠️ Warning: Low confidence, prediction might not be accurate.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
    else:
        predict_image(sys.argv[1])
