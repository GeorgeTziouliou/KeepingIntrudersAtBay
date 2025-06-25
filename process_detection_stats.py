import cv2
import dlib
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration ---
DATA_PATH = "FacesTested"
RECOGNIZE_FACES = True

MODEL_PATH = "face_recognizer_model.pkl"
LABELS_PATH = "label_names.pkl"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm')
RECOGNITION_CONFIDENCE_THRESHOLD = 50
# -------------------

def load_recognition_models():
    """Loads all necessary models for face recognition."""
    models = {}
    try:
        print("Loading dlib shape predictor model...")
        models['sp'] = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        print("Loading dlib face recognition model...")
        models['facerec'] = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
        print(f"Loading KNN classifier model from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            models['knn_model'] = pickle.load(f)
        print(f"Loading label dictionary from {LABELS_PATH}...")
        with open(LABELS_PATH, "rb") as f:
            models['label_dict_int_name'] = pickle.load(f)
            models['label_dict_name_int'] = {v: k for k, v in models['label_dict_int_name'].items()}
        print("Recognition models loaded successfully.")
        return models
    except FileNotFoundError as e:
        print(f"\nError loading model file: {e}")
        print("Ensure all model paths in the script configuration are correct and files exist.")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred loading models: {e}")
        return None

def analyze_images_flat(data_path, recognize_faces=False):
    """
    Analyzes face detection and optionally recognition across all images in a single directory.
    """
    if not os.path.isdir(data_path):
        print(f"Error: Directory not found at '{data_path}'")
        return None

    print("Loading dlib face detector model...")
    try:
        detector = dlib.get_frontal_face_detector()
        print("Detector loaded successfully.")
    except Exception as e:
        print(f"Error loading dlib detector: {e}")
        return None

    rec_models = None
    known_names = set()
    if recognize_faces:
        rec_models = load_recognition_models()
        if rec_models is None:
            return None
        known_names = set(rec_models['label_dict_int_name'].values())

    total_files_scanned = 0
    total_images_processed = 0
    images_with_faces = 0
    images_with_no_faces = 0
    images_with_multiple_faces = 0
    total_faces_detected = 0
    error_reading_files = []
    no_face_files = []
    multi_face_files = {}
    total_faces_processed_for_recog = 0
    detailed_predictions = defaultdict(list)
    other_predictions_count = 0
    unknown_predictions_count = 0

    print(f"\nProcessing images in directory: '{data_path}'...")
    print(f"Face Recognition: {'ENABLED' if recognize_faces else 'DISABLED'}")

    try:
        file_list = os.listdir(data_path)
    except OSError as e:
        print(f"Error accessing directory '{data_path}': {e}")
        return None

    for filename in tqdm(file_list, desc="Processing Images"):
        total_files_scanned += 1
        file_path = os.path.join(data_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
            try:
                image = cv2.imread(file_path)
                if image is None:
                    print(f"\nWarning: Could not read image file: {filename}")
                    error_reading_files.append(filename)
                    continue

                total_images_processed += 1
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                num_faces = len(faces)

                total_faces_detected += num_faces
                if num_faces == 0:
                    images_with_no_faces += 1
                    no_face_files.append(filename)
                else:
                    images_with_faces += 1
                    if num_faces > 1:
                        images_with_multiple_faces += 1
                        multi_face_files[filename] = num_faces

                if recognize_faces and num_faces > 0:
                    for face in faces:
                        total_faces_processed_for_recog += 1
                        try:
                            shape = rec_models['sp'](gray, face)
                            face_descriptor = rec_models['facerec'].compute_face_descriptor(image, shape)
                            face_descriptor = np.array(face_descriptor).reshape(1, -1)

                            knn_model: KNeighborsClassifier = rec_models['knn_model']
                            prediction_label = knn_model.predict(face_descriptor)[0]
                            distances, _ = knn_model.kneighbors(face_descriptor)
                            confidence = max(0, min(100, 100 - (distances[0][0] * 100)))

                            predicted_name = "Unknown" # Default
                            is_other = False
                            if confidence < RECOGNITION_CONFIDENCE_THRESHOLD:
                                predicted_name = "Other"
                                other_predictions_count += 1
                                is_other = True
                            else:
                                if prediction_label in rec_models['label_dict_int_name']:
                                    predicted_name = rec_models['label_dict_int_name'][prediction_label]
                                else:
                                    unknown_predictions_count += 1

                            detailed_predictions[filename].append({
                                'pred_name': predicted_name,
                                'conf': confidence,
                                'is_other': is_other
                            })

                        except Exception as recog_error:
                            print(f"\nError during recognition for a face in {filename}: {recog_error}")

            except Exception as e:
                print(f"\nError processing file {filename}: {e}")
                error_reading_files.append(filename)

    print("\nProcessing complete.")

    stats = {
        # Detection
        "directory_analyzed": data_path,
        "total_files_scanned": total_files_scanned,
        "total_images_processed": total_images_processed,
        "images_with_faces_detected": images_with_faces,
        "images_with_no_faces_detected": images_with_no_faces,
        "images_with_multiple_faces_detected": images_with_multiple_faces,
        "total_faces_detected": total_faces_detected,
        "files_with_read_errors": error_reading_files,
        "files_with_no_faces": no_face_files,
        "files_with_multiple_faces": multi_face_files,
        # Recognition
        "recognition_enabled": recognize_faces,
        "total_faces_processed_for_recognition": total_faces_processed_for_recog,
        "other_predictions_count": other_predictions_count,
        "unknown_predictions_count": unknown_predictions_count,
        "detailed_predictions": dict(detailed_predictions),
        "known_names_in_model": list(known_names)
    }
    return stats

def print_report_detailed(stats):
    """Prints a formatted report focusing on non-'Other' predictions."""
    if stats is None:
        return

    print("\n--- Image Processing Statistics Report (Flat Directory - Detailed) ---")
    print(f"Directory Analyzed:          {stats['directory_analyzed']}")

    print("\n--- DETECTION STATISTICS ---")
    print(f"Total Files Scanned:         {stats['total_files_scanned']}")
    print(f"Total Valid Images Processed:{stats['total_images_processed']}")
    print(f"Files with Read Errors:      {len(stats['files_with_read_errors'])}")
    print("-" * 35)
    print(f"Images with >= 1 Face:     {stats['images_with_faces_detected']}")
    print(f"Images with 0 Faces:       {stats['images_with_no_faces_detected']}")
    print(f"Images with > 1 Face:      {stats['images_with_multiple_faces_detected']}")
    print(f"Total Faces Detected:        {stats['total_faces_detected']}")
    print("-" * 35)
    if stats['total_images_processed'] > 0:
        detection_rate = (stats['images_with_faces_detected'] / stats['total_images_processed']) * 100
        print(f"Detection Rate (% images with faces): {detection_rate:.2f}%")
        avg_faces_per_processed_image = stats['total_faces_detected'] / stats['total_images_processed']
        print(f"Avg Faces per Processed Image: {avg_faces_per_processed_image:.2f}")
    else:
        print("Detection/Average Rates: N/A (No images processed)")

    # --- RECOGNITION STATISTICS ---
    if stats['recognition_enabled']:
        print("\n--- RECOGNITION STATISTICS ---")
        print(f"Total Faces Processed for Recog.: {stats['total_faces_processed_for_recognition']}")
        print(f"(Confidence Threshold for Known: >={RECOGNITION_CONFIDENCE_THRESHOLD}%)")
        print("-" * 35)
        print(f"Total Predictions as 'Other': {stats['other_predictions_count']}")
        print(f"Total Predictions as 'Unknown': {stats['unknown_predictions_count']}")
        print("-" * 35)

        # --- DETAILED PREDICTIONS (NON-'Other') ---
        print("\n--- DETAILED PREDICTIONS (Excluding 'Other') ---")
        non_other_predictions_found = False


        for filename, predictions in stats['detailed_predictions'].items():
            # Filter out 'Other' predictions for this file
            non_other_preds_in_file = [
                f"{p['pred_name']} ({p['conf']:.1f}%)"
                for p in predictions if not p['is_other']
            ]

            # If there are any non-'Other' predictions for this file, print them
            if non_other_preds_in_file:
                print(f"  - {filename}: {non_other_preds_in_file}")
                non_other_predictions_found = True

        if not non_other_predictions_found:
            print("  (No faces were recognized as known individuals or 'Unknown')")
        print("-" * 35)

    print("\n--- FILE LISTS ---")
    if stats['files_with_read_errors']:
        print("\nFiles with Read Errors:")
        for f in stats['files_with_read_errors']: print(f"  - {f}")

    if stats['files_with_no_faces']:
        print("\nFiles Where No Faces Were Detected:")
        for f in stats['files_with_no_faces']: print(f"  - {f}")

    if stats['files_with_multiple_faces']:
        print("\nFiles Where Multiple Faces Were Detected:")
        for f, count in stats['files_with_multiple_faces'].items(): print(f"  - {f} ({count} faces)")


    print("\n--- End of Report ---")


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Directory '{DATA_PATH}' not found. Please create it and add image files.")
    else:
        all_stats = analyze_images_flat(DATA_PATH, recognize_faces=RECOGNIZE_FACES)
        print_report_detailed(all_stats) # Use the modified report function