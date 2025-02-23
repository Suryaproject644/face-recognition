import cv2
import os
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# Haar cascade files for face and eye detection
haar_face = 'haarcascade_frontalface_alt.xml'
haar_eye = 'haarcascade_eye.xml'
datasets = 'dataset'

subjects = ['.NET', 'EVS', 'MOBILE TECHNOLOGY', 'PROJECTLAB', 'ERP']
selected_subject = None

face_cascade = cv2.CascadeClassifier(haar_face)
eye_cascade = cv2.CascadeClassifier(haar_eye)

recognizer = cv2.face.LBPHFaceRecognizer_create()

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

def mark_attendance(name):
    if not selected_subject:
        print("No subject selected! Attendance cannot be marked.")
        return
    attendance_file = f'attendance_{selected_subject.lower()}.csv'
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Time'])
    with open(attendance_file, 'r', newline='') as file:
        reader = csv.reader(file)
        entries = list(reader)
    if any(row[0] == name for row in entries):
        print(f"{name} is already marked for {selected_subject}.")
        return
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, current_time])
        print(f"Attendance marked for {name} in {selected_subject} at {current_time}.")

def select_subject(subject):
    global selected_subject
    selected_subject = subject
    messagebox.showinfo("Subject Selected", f"Subject set to {subject}. You can start attendance.")

def start_recognition():
    if not selected_subject:
        messagebox.showwarning("Warning", "Please select a subject before starting attendance!")
        return

    ip_webcam_url = 'http://192.168.22.58:8080/video'
    webcam = cv2.VideoCapture(ip_webcam_url)
    if not webcam.isOpened():
        print("Error: Could not access IP webcam. Check the URL.")
        return
    
    blink_detected = False
    prev_face_position = None
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_region)
            
            if len(eyes) >= 2:
                blink_detected = True  # Eyes detected, assume real person
            
            if prev_face_position is not None:
                movement = abs(x - prev_face_position[0]) + abs(y - prev_face_position[1])
                if movement > 5:
                    head_moved = True
                else:
                    head_moved = False
            else:
                head_moved = False
            prev_face_position = (x, y)
            
            if blink_detected and head_moved:
                face_resize = cv2.resize(face_region, (130, 100))
                label, confidence = recognizer.predict(face_resize)
                name = label_dict.get(label, "Unknown")
                if confidence < 100:
                    mark_attendance(name)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Fake Attempt", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('IP Webcam Face Recognition Attendance', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Attendance System")
root.geometry("300x400")

tk.Label(root, text="Select Subject", font=("Arial", 14)).pack(pady=10)

for subject in subjects:
    tk.Button(root, text=subject, font=("Arial", 12), width=15, command=lambda s=subject: select_subject(s)).pack(pady=5)

tk.Button(root, text="Start Attendance", font=("Arial", 12), width=15, bg="green", fg="white", command=start_recognition).pack(pady=20)

root.mainloop()
