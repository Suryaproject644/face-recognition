import cv2
import os
import numpy as np
import csv
from datetime import datetime

# Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_alt.xml'
datasets = 'dataset'  # Dataset folder containing subfolders of registered faces
attendance_file = 'attendance.csv'

# Create the attendance CSV file if it doesn't exist
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Time'])

# Load the face detector
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
def prepare_training_data(datasets):
    faces, labels = [], []
    label_dict = {}
    label_id = 0

    for subdir in os.listdir(datasets):
        if os.path.isdir(os.path.join(datasets, subdir)):
            label_dict[label_id] = subdir
            for filename in os.listdir(os.path.join(datasets, subdir)):
                filepath = os.path.join(datasets, subdir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_id)
            label_id += 1

    return np.array(faces), np.array(labels), label_dict  # Convert lists to NumPy arrays

print("Training the recognizer...")
faces, labels, label_dict = prepare_training_data(datasets)
recognizer.train(faces, labels)  # No error here now
print("Training complete!")

# Function to mark attendance
def mark_attendance(name):
    with open(attendance_file, 'r', newline='') as file:
        reader = csv.reader(file)
        entries = list(reader)

    # Check if the name already exists in the attendance file
    if any(row[0] == name for row in entries):
        print(f"{name} already marked attendance.")
        return

    # If not, mark the attendance with the current time
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, current_time])
        print(f"Attendance marked for {name} at {current_time}.")

# Replace this URL with your IP webcam's URL
# Example: 'http://<IP>:<PORT>/video' (e.g., 'http://192.168.1.101:8080/video')
ip_webcam_url = 'http://192.168.22.58:8080/video'
webcam = cv2.VideoCapture(ip_webcam_url)

if not webcam.isOpened():
    print("Error: Could not access IP webcam. Check the URL.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))

        # Recognize the face
        label, confidence = recognizer.predict(face_resize)
        name = label_dict.get(label, "Unknown")

        if confidence < 100:  # Confidence threshold (adjust as needed)
            mark_attendance(name)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('IP Webcam Face Recognition Attendance', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        break

webcam.release()
cv2.destroyAllWindows()
