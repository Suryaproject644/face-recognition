import cv2
import os
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog

# Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_alt.xml'
datasets = 'dataset'  # Dataset folder containing subfolders of registered faces

# Subjects and their passwords
subjects = {
    '.NET': 'dotnet123',
    'EVS': 'evs123',
    'MOBILE TECHNOLOGY': 'mobile123',
    'PROJECTLAB': 'project123',
    'ERP': 'erp123'
}

selected_subject = None  # Stores the subject selected by the teacher

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

    # Check if the student is already marked in the current subject
    with open(attendance_file, 'r', newline='') as file:
        reader = csv.reader(file)
        entries = list(reader)

    if any(row[0] == name for row in entries):
        print(f"{name} is already marked for {selected_subject}.")
        return

    # If not, mark attendance
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, current_time])
        print(f"Attendance marked for {name} in {selected_subject} at {current_time}.")

# Function to select a subject with password protection
def select_subject(subject):
    global selected_subject

    # Prompt for password
    password = simpledialog.askstring("Password Required", f"Enter password for {subject}:", show='*')

    # Check if the password matches
    if password == subjects[subject]:
        selected_subject = subject
        messagebox.showinfo("Access Granted", f"Access granted! Subject set to {subject}. You can start attendance.")
    else:
        messagebox.showerror("Access Denied", "Incorrect password. Access denied.")

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

# Tkinter GUI
root = tk.Tk()
root.title("Attendance System")
root.geometry("300x400")

tk.Label(root, text="Select Subject", font=("Arial", 14)).pack(pady=10)

for subject in subjects.keys():
    tk.Button(root, text=subject, font=("Arial", 12), width=20, command=lambda s=subject: select_subject(s)).pack(pady=5)

tk.Button(root, text="Start Attendance", font=("Arial", 12), width=20, bg="green", fg="white", command=start_recognition).pack(pady=20)

root.mainloop()
