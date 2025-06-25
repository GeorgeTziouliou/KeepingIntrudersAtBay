import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Press 'q' to quit the application.")


def on_key(event):
    if event.key == 'q':
        plt.close()


fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)


video_capture.release()
cv2.destroyAllWindows()
