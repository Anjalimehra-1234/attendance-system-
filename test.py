import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Load data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize camera
video = cv2.VideoCapture(0)

# Load face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Attendance setup
COL_NAMES = ['NAME', 'TIME']
marked_names = set()

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        output = knn.predict(resized_img)

        ts = datetime.now()
        date = ts.strftime("%d-%m-%Y")
        timestamp = ts.strftime("%H:%M:%S")

        filename = f"Attendance/Attendance_{date}.csv"

        if not os.path.exists('Attendance'):
            os.makedirs('Attendance')

        # Avoid duplicate entries
        if output[0] not in marked_names:
            marked_names.add(output[0])

            if not os.path.isfile(filename):
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(COL_NAMES)

            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([output[0], timestamp])

        # Display on screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        cv2.putText(frame, str(output[0]), (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()