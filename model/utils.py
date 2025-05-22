import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can upgrade to yolov8s.pt for better accuracy

def extract_faces(video_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0

    while cap.isOpened() and len(faces) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0:
            continue

        results = model(frame)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_img)
                faces.append(face_tensor)
                break
        if len(faces) >= 5:
            break

    cap.release()
    return faces
