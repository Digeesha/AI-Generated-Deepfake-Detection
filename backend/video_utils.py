import cv2
from torchvision import transforms
from PIL import Image

def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in detections:
            face = frame[y:y+h, x:x+w]
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face)
            faces.append(face_tensor)
            if len(faces) >= 10:
                break
    cap.release()
    return faces
