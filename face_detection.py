import cv2
import numpy as np

# Load the trained recognizer and the face detector
model_path = "face_recognizer_model.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


UNLOCK_THRESHOLD = 50

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Press 'q' to quit the application.")

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (200, 200))

        # Predict using the recognizer
        label, confidence = recognizer.predict(face_roi_resized)

        print(f"Label: {label}, Confidence: {confidence}")

        # Unlock door if confidence is below the threshold
        if confidence < UNLOCK_THRESHOLD:
            print("Door Unlocked!")
        else:
            print("Face did not match.")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
