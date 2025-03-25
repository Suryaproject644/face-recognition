import cv2
import os
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)  # Higher confidence

# Dataset folder setup
datasets = 'Dataset'
sub_data = 'Jayanthi'  # Change this for each student

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)  # Resize dimensions

# Replace with your IP webcam stream URL
ip_webcam_url = 'http://192.168.189.210:8080/video'
webcam = cv2.VideoCapture(ip_webcam_url)

if not webcam.isOpened():
    print("Error: Could not open IP webcam stream. Check the URL.")
    exit()

count = 1
while count <= 51:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture image. Retrying...")
        continue

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            # Convert relative box to pixel coordinates
            x, y, box_w, box_h = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                  int(bboxC.width * w), int(bboxC.height * h))

            # Ensure bounding box is within image limits
            x, y = max(0, x), max(0, y)
            box_w, box_h = min(w - x, box_w), min(h - y, box_h)

            # Add margin to bounding box
            margin = 10  # Adjust margin size as needed
            x, y = max(0, x - margin), max(0, y - margin)
            box_w, box_h = min(w - x, box_w + 2 * margin), min(h - y, box_h + 2 * margin)

            # Crop the face region
            face = frame[y:y+box_h, x:x+box_w]

            if face.size == 0:  # If face region is empty, skip saving
                continue

            # Check if the face is blurry using Laplacian variance
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            blur_value = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if blur_value < 50:  # Adjust threshold as needed
                print(f"Image {count} is too blurry, skipping...")
                continue

            # Convert to grayscale and resize
            face_resized = cv2.resize(gray_face, (width, height))

            # Save cropped face
            cv2.imwrite(f"{path}/{count}.png", face_resized)
            count += 1

            # Draw bounding box for visualization
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)

    cv2.imshow('IP Webcam Feed', frame)

    key = cv2.waitKey(10)
    if key == 27 or count > 20:  # Escape key to exit
        break

webcam.release()
cv2.destroyAllWindows()
