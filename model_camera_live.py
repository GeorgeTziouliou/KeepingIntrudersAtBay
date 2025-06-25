import cv2
import dlib
import pickle
import numpy as np
from scipy.spatial import distance
import time
from collections import defaultdict

# Configuration
USE_BLINK_DETECTION = True

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


class FaceLivenessState:
    def __init__(self):
        self.motion_history = []
        self.last_seen = time.time()
        self.blink_counter = 0
        self.total = 0
        self.ear_history = []
        self.last_blink_time = time.time()
        self.blink_detected = False


class LivenessDetector:
    def __init__(self):
        self.face_states = defaultdict(FaceLivenessState)
        self.cleanup_interval = 5
        self.last_cleanup = time.time()

    def cleanup_old_faces(self):
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            inactive_faces = []
            for face_id, state in self.face_states.items():
                if current_time - state.last_seen > 5:
                    inactive_faces.append(face_id)

            for face_id in inactive_faces:
                del self.face_states[face_id]

            self.last_cleanup = current_time

    def get_face_id(self, face, frame_width):
        center_x = (face.left() + face.right()) / 2
        center_y = (face.top() + face.bottom()) / 2
        return f"{int(center_x / frame_width * 100)}_{int(center_y / frame_width * 100)}"

    def detect_blink(self, shape, face_state):
        try:
            # Basic blink implementation - always return True if disabled
            if not USE_BLINK_DETECTION:
                return True

            # Extract eye landmarks
            left_eye = np.array([(shape.part(36).x, shape.part(36).y),
                                 (shape.part(37).x, shape.part(37).y),
                                 (shape.part(38).x, shape.part(38).y),
                                 (shape.part(39).x, shape.part(39).y),
                                 (shape.part(40).x, shape.part(40).y),
                                 (shape.part(41).x, shape.part(41).y)])

            right_eye = np.array([(shape.part(42).x, shape.part(42).y),
                                  (shape.part(43).x, shape.part(43).y),
                                  (shape.part(44).x, shape.part(44).y),
                                  (shape.part(45).x, shape.part(45).y),
                                  (shape.part(46).x, shape.part(46).y),
                                  (shape.part(47).x, shape.part(47).y)])

            # Calculate eye aspect ratios
            left_ear = self.get_eye_aspect_ratio(left_eye)
            right_ear = self.get_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Basic threshold check
            if ear < 0.3:
                face_state.total += 1

            return face_state.total > 0

        except Exception as e:
            print(f"Blink detection error: {str(e)}")
            return True 

    def get_eye_aspect_ratio(self, eye_landmarks):
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_movement(self, face_location, frame_shape, face_state):
        try:
            center_x = (face_location.left() + face_location.right()) / 2
            center_y = (face_location.top() + face_location.bottom()) / 2

            norm_x = center_x / frame_shape[1]
            norm_y = center_y / frame_shape[0]

            face_state.motion_history.append((norm_x, norm_y))
            if len(face_state.motion_history) > 30:
                face_state.motion_history.pop(0)

            if len(face_state.motion_history) >= 10:
                x_coords = [x for x, y in face_state.motion_history]
                y_coords = [y for x, y in face_state.motion_history]
                movement_variance = np.var(x_coords) + np.var(y_coords)

                return 0.00005 < movement_variance < 0.02
            return True
        except:
            return False

    def check_texture_variance(self, frame, face):
        try:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            height, width = frame.shape[:2]

            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                return False

            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                return False

            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray_roi)
            return variance > 400
        except:
            return False


def recognize_face(frame, liveness_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    liveness_detector.cleanup_old_faces()

    for face in faces:
        face_id = liveness_detector.get_face_id(face, frame.shape[1])
        face_state = liveness_detector.face_states[face_id]
        face_state.last_seen = time.time()

        shape = sp(gray, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        face_descriptor = np.array(face_descriptor).reshape(1, -1)

        prediction = model.predict(face_descriptor)[0]
        distances, _ = model.kneighbors(face_descriptor)
        confidence = 100 - (distances[0][0] * 100)
        confidence = max(0, min(confidence, 100))

        if confidence < 50:
            person_name = "Other"
        else:
            person_name = label_dict.get(prediction, "Unknown")

        # Perform liveness checks
        blink_check = True if not USE_BLINK_DETECTION else liveness_detector.detect_blink(shape, face_state)
        movement_detected = liveness_detector.detect_movement(face, frame.shape, face_state)
        texture_check = liveness_detector.check_texture_variance(frame, face)

        is_live = movement_detected and texture_check
        if USE_BLINK_DETECTION:
            is_live = is_live and blink_check

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 255, 0) if is_live else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        status_text = f"{person_name}: {confidence:.2f}%"
        liveness_text = "Live" if is_live else "Fake"
        cv2.putText(frame, status_text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Status: {liveness_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        debug_y = y + h + 15
        if USE_BLINK_DETECTION:
            cv2.putText(frame, f"Blinks: {face_state.total}", (x, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            debug_y += 15
        cv2.putText(frame, f"Movement: {movement_detected}", (x, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, f"Texture: {texture_check}", (x, debug_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    liveness_detector = LivenessDetector()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit...")
    print(f"Blink detection is {'enabled' if USE_BLINK_DETECTION else 'disabled'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = recognize_face(frame, liveness_detector)
        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()