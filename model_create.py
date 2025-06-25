"""
Face Recognition Model Training Script.

This script trains a K-Nearest Neighbors (KNN) classifier for face recognition.
It reads images from a specified dataset directory (where each subdirectory
represents a person), detects faces using dlib's frontal face detector,
extracts 128-dimensional face embeddings (descriptors) using dlib's deep
learning model, and then trains a KNN model on these embeddings.

The trained KNN model and a dictionary mapping numerical labels back to
person names are saved to pickle files.
"""

import cv2
import os
import dlib
import numpy as np
import pickle
from tqdm import tqdm  # Progress bar library
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Configuration Paths
DATA_PATH = "CelebrityFaces"  # Directory containing training images (subdirs named by person)
MODEL_PATH = "face_recognizer_model.pkl"  # Path to save the trained KNN model
LABELS_PATH = "label_names.pkl"  # Path to save the label-to-name mapping

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
# Predictor for finding facial landmarks
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Face recognition model to generate embeddings
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def load_images_and_labels(data_path):
    """
    Loads images from the dataset directory, extracts face embeddings, and assigns labels.

    Iterates through subdirectories in `data_path`. Each subdirectory name is
    treated as a person's label. For each image within a subdirectory, it detects
    the primary face, computes its 128d embedding, and stores the embedding
    along with a numerical label corresponding to the person.

    Args:
        data_path (str): The path to the root directory of the image dataset.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: An array of face embeddings (shape: [n_samples, 128]).
            - numpy.ndarray: An array of numerical labels corresponding to the embeddings.
            - dict: A dictionary mapping numerical labels (int) to person names (str).
    """
    face_embeddings = []
    labels = []
    label_dict = {}  # Maps numerical label ID to person name
    current_label = 0

    print(f"Scanning dataset directory: {data_path}")
    # Iterate through person folders with progress bar
    for person_name in tqdm(os.listdir(data_path), desc="Processing Folders", unit="folder"):
        person_path = os.path.join(data_path, person_name)
        if not os.path.isdir(person_path):
            continue  # Skip files, only process directories

        print(f"  Processing person: {person_name}")
        label_dict[current_label] = person_name  # Store the name for this label ID

        # Iterate through images in the person's folder
        for filename in os.listdir(person_path):
            # Check for common image file extensions
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm', '.bmp')):
                filepath = os.path.join(person_path, filename)

                try:
                    # Read the image
                    image = cv2.imread(filepath)
                    if image is None:
                        print(f"    Warning: Could not read image {filepath}. Skipping...")
                        continue

                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Detect faces in the grayscale image
                    faces = detector(gray)

                    if len(faces) == 0:
                        print(f"    Warning: No face detected in {filepath}. Skipping...")
                        continue
                    elif len(faces) > 1:
                        print(f"    Info: Multiple faces detected in {filepath}. Using the largest.")
                        # Optional: Select the largest face if multiple are found
                        faces = sorted(faces, key=lambda rect: rect.width() * rect.height(), reverse=True)

                    # Process the first (or largest) detected face
                    face = faces[0]
                    # Get facial landmarks
                    shape = sp(gray, face)
                    # Compute the 128d face embedding using the color image and landmarks
                    face_descriptor = facerec.compute_face_descriptor(image, shape)

                    # Store the embedding and its corresponding label
                    face_embeddings.append(np.array(face_descriptor))
                    labels.append(current_label)

                except Exception as e:
                    print(f"    Error processing {filepath}: {e}")

        current_label += 1  # Increment label ID for the next person

    return np.array(face_embeddings), np.array(labels), label_dict


def train_model():
    """
    Orchestrates the model training process.

    Loads image data and labels, splits data into training and testing sets,
    trains a K-Nearest Neighbors classifier, and saves the trained model
    and label dictionary to disk using pickle.
    """
    print("Starting face recognition model training...")
    print("-" * 30)

    print("Step 1: Loading images and extracting face embeddings...")
    embeddings, labels, label_dict = load_images_and_labels(DATA_PATH)

    if len(embeddings) == 0:
        print("\nError: No valid face embeddings extracted. Check dataset structure and image quality.")
        print("Ensure dataset follows 'DATA_PATH/PersonName/image.jpg' structure.")
        return

    num_classes = len(label_dict)
    print(f"\nLoaded {len(embeddings)} face embeddings across {num_classes} classes (people).")
    if num_classes < 2:
        print("Error: Need at least two classes (people) with valid images to train a classifier.")
        return

    print("\nStep 2: Splitting data into training and testing sets...")
    # Split data (80% train, 20% test)
    # stratify=labels ensures proportional representation of each class in splits
    try:
        embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"  Training set size: {len(embeddings_train)} samples")
        print(f"  Testing set size:  {len(embeddings_test)} samples")
    except ValueError as e:
        print(f"Error during train/test split: {e}")
        print("This might happen if some classes have only one sample.")
        print("Using all data for training instead.")
        embeddings_train, labels_train = embeddings, labels # Fallback: train on all data
        embeddings_test, labels_test = [], [] # No testing


    print("\nStep 3: Training the K-Nearest Neighbors (KNN) classifier...")
    # Initialize KNN classifier (k=3 is a common starting point)
    # 'metric=euclidean' is standard for dlib embeddings, 'weights=distance'
    # gives more weight to closer neighbors.
    model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance')
    model.fit(embeddings_train, labels_train)
    print("  KNN model training complete.")

    if len(embeddings_test) > 0:
        print("\nStep 4: Evaluating model accuracy on the test set...")
        accuracy = model.score(embeddings_test, labels_test)
        print(f"  Model Accuracy: {accuracy * 100:.2f}%")
    else:
        print("\nStep 4: Skipping evaluation (no test set available).")

    print("\nStep 5: Saving the trained model and label dictionary...")
    # Save the trained KNN model to a file
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  Trained model saved successfully to: {MODEL_PATH}")

        # Save the label dictionary (ID -> Name mapping)
        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_dict, f)
        print(f"  Label dictionary saved successfully to: {LABELS_PATH}")
    except Exception as e:
        print(f"  Error saving files: {e}")

    print("-" * 30)
    print("Training process finished.")


if __name__ == "__main__":
    # Ensure the necessary dlib model files are present
    required_files = ["shape_predictor_68_face_landmarks.dat",
                      "dlib_face_recognition_resnet_model_v1.dat"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Required dlib model file(s) not found in the current directory:")
        for f in missing_files:
            print(f" - {f}")
        print("Please download them and place them alongside this script.")
    elif not os.path.isdir(DATA_PATH):
        print(f"Error: Dataset directory '{DATA_PATH}' not found.")
        print("Please create this directory and populate it with subfolders for each person.")
    else:
        train_model()