import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

CLUTTER_CLASSES = ["bottle", "cup", "book", "backpack", "chair"]
PHONE_CLASS = "cell phone"

def detect_objects(frame):
    results = model(frame, stream=True)
    people = 0
    clutter = 0
    phones = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                people += 1
            elif label == PHONE_CLASS:
                phones += 1
            elif label in CLUTTER_CLASSES:
                clutter += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cleanliness_score = max(0, 100 - clutter * 10)
    return frame, people, phones, cleanliness_score
