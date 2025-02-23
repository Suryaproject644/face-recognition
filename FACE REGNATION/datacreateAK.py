import cv2
import os                    

haar_file = 'haarcascade_frontalface_alt.xml'
datasets = 'dataset'
sub_data = 'Ezhil'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)

# Replace this URL with your IP webcam stream URL
ip_webcam_url = 'http://192.168.22.58:8080/video'
webcam = cv2.VideoCapture(ip_webcam_url)

if not webcam.isOpened():
    print("Error: Could not open IP webcam stream. Please check the URL.")
    exit()

count = 1
while count < 20:
    print(count)
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image. Retrying...")
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1

    cv2.imshow('IP Webcam Feed', im)
    key = cv2.waitKey(10)
    if key == 27:  # Escape key to exit
        break

webcam.release()
cv2.destroyAllWindows()

