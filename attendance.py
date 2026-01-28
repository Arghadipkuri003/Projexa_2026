import face_recognition
import cv2
import os
from datetime import datetime

KNOWN_DIR = "known_faces"
known_encodings = []
known_names = []

for name in os.listdir(KNOWN_DIR):
    img = face_recognition.load_image_file(f"{KNOWN_DIR}/{name}")
    enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(enc)
    known_names.append(name.split(".")[0])

def mark_attendance(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    present = []

    for enc in encodings:
        matches = face_recognition.compare_faces(known_encodings, enc)
        if True in matches:
            idx = matches.index(True)
            present.append(known_names[idx])

    return list(set(present))
