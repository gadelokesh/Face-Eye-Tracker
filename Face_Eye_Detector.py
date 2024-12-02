import cv2
import numpy as np

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(r"C:\Users\gadel\VS Code projects\Face and Eye Detector\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"C:\Users\gadel\VS Code projects\Face and Eye Detector\haarcascade_eye.xml")

# Check if the cascade files are loaded
if face_classifier.empty() or eye_classifier.empty():
    print("Error: Haar cascade file not loaded. Check the file path.")
    exit()

def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # If no faces are detected, return the original image
    if len(faces) == 0:
        return img

    for (x, y, w, h) in faces:
        x = max(0, x - 50)  # Ensure x and y are not negative
        y = max(0, y - 50)
        w += 50
        h += 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Label Face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Label Eye
    
    return img

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    cv2.imshow('Our Face Extractor', face_detector(frame))

    # Break the loop with Enter key
    if cv2.waitKey(1) == 13:  # 13 is Enter key
        break

cap.release()
cv2.destroyAllWindows()