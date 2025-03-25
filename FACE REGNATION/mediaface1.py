import cv2
import os
import numpy as np
import csv
from datetime import datetime
import mediapipe as mp
from deepface import DeepFace

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Dataset folder
datasets = 'Dataset'

# List of subjects
subjects = ['.NET', 'EVS', 'MOBILE TECHNOLOGY', 'PROJECTLAB', 'ERP']
selected_subject = None  # Store selected subject

# Load and preprocess dataset (generate face embeddings)
def create_face_database():
    database = {}
    for student in os.listdir(datasets):
        student_path = os.path.join(datasets, student)
        if os.path.isdir(student_path):
            for img_name in os.listdir(student_path):
                img_path = os.path.join(student_path, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                    if embedding:
                        database[student] = embedding[0]['embedding']
                except:
                    print(f"Error processing {img_path}")
    return database

print("Creating face database...")
face_database = create_face_database()
print("Face database created successfully!")

# Function to find the best match
def find_best_match(face_embedding):
    best_match = None
    best_distance = float('inf')
    
    for name, stored_embedding in face_database.items():
        distance = np.linalg.norm(np.array(face_embedding) - np.array(stored_embedding))
        if distance < best_distance:  # Lower distance means better match
            best_distance = distance
            best_match = name
    
    return best_match if best_distance < 10 else "Unknown"  # Threshold

# Function to mark attendance
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

# Function to start face recognition
def start_recognition():
    global selected_subject
    if not selected_subject:
        print("Select a subject before starting attendance!")
        return

    ip_webcam_url = 'http://192.168.189.210:8080/video'
    webcam = cv2.VideoCapture(ip_webcam_url)

    if not webcam.isOpened():
        print("Error: Could not access IP webcam. Check the URL.")
        return

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using Mediapipe
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

                # Crop the detected face
                face = frame[y:y+box_h, x:x+box_w]

                if face.size == 0:  # If face region is empty, skip
                    continue

                # Recognize face using DeepFace
                try:
                    embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
                    if embedding:
                        name = find_best_match(embedding[0]['embedding'])
                        if name != "Unknown":
                            mark_attendance(name)
                    
                    # Display recognized name
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    cv2.putText(frame, "Error in recognition", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)

        cv2.imshow('Face Recognition Attendance', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
            break

    webcam.release()
    cv2.destroyAllWindows()

# Function to select subject
def select_subject(subject):
    global selected_subject
    selected_subject = subject
    print(f"Subject set to {subject}. You can start attendance.")

# Start face recognition
select_subject("EVS")  # Example: Set subject manually
start_recognition()
