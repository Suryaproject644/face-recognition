import cv2
import os
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Dataset folder
datasets = 'dataset'  # Folder containing subfolders of registered faces

# List of subjects
subjects = ['.NET', 'EVS', 'MOBILE TECHNOLOGY', 'PROJECTLAB', 'ERP']
selected_subject = None  # Store selected subject

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

    return np.array(faces), np.array(labels), label_dict

print("Training the recognizer...")
faces, labels, label_dict = prepare_training_data(datasets)
recognizer.train(faces, labels)
print("Training complete!")

# Function to mark attendance
def mark_attendance(name):
    if not selected_subject:
        print("No subject selected! Attendance cannot be marked.")
        return

    attendance_file = f'attendance_{selected_subject.lower()}.csv'

    # Create the attendance file for the subject if it doesn't exist
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Time'])

    # Check if student is already marked in this subject
    with open(attendance_file, 'r', newline='') as file:
        reader = csv.reader(file)
        entries = list(reader)

    if any(row[0] == name for row in entries):
        print(f"{name} is already marked for {selected_subject}.")
        return

    # Mark attendance
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, current_time])
        print(f"Attendance marked for {name} in {selected_subject} at {current_time}.")

# Function to select subject
def select_subject(subject):
    global selected_subject
    selected_subject = subject
    messagebox.showinfo("Subject Selected", f"Subject set to {subject}. You can start attendance.")

# Function to start face recognition
def start_recognition():
    if not selected_subject:
        messagebox.showwarning("Warning", "Please select a subject before starting attendance!")
        return

    ip_webcam_url = 'http://192.168.22.58:8080/video'
    webcam = cv2.VideoCapture(ip_webcam_url)

    if not webcam.isOpened():
        print("Error: Could not access IP webcam. Check the URL.")
        return

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # Convert bounding box to pixel coordinates
                x, y, box_w, box_h = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                      int(bboxC.width * w), int(bboxC.height * h))

                # Ensure bounding box is within image limits
                x, y = max(0, x), max(0, y)
                box_w, box_h = min(w - x, box_w), min(h - y, box_h)

                # Crop and process face
                face = frame[y:y+box_h, x:x+box_w]

                if face.size == 0:  # If face region is empty, skip
                    continue

                # Convert face to grayscale & resize
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (130, 100))

                # Recognize face
                label, confidence = recognizer.predict(face_resized)
                name = label_dict.get(label, "Unknown")

                if confidence < 100:  # Confidence threshold
                    mark_attendance(name)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Draw bounding box for visualization
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)

        cv2.imshow('IP Webcam Face Recognition Attendance', frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break

    webcam.release()
    cv2.destroyAllWindows()

# Tkinter GUI
root = tk.Tk()
root.title("Attendance System")
root.geometry("300x400")

tk.Label(root, text="Select Subject", font=("Arial", 14)).pack(pady=10)

for subject in subjects:
    tk.Button(root, text=subject, font=("Arial", 12), width=20, command=lambda s=subject: select_subject(s)).pack(pady=5)

tk.Button(root, text="Start Attendance", font=("Arial", 12), width=20, bg="green", fg="white", command=start_recognition).pack(pady=20)

root.mainloop()
