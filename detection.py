import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import os

# Constants
REFERENCE_DIR = "Reference_faces"
MATCH_THRESHOLD = 0.6
AUDIO_FILE = "technologia.mp3"

# Initialize models
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize audio
pygame.mixer.init()

# Load reference images and compute embeddings
known_faces = []

def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_image)
    if len(faces) == 0:
        return None
    shape = shape_predictor(rgb_image, faces[0])
    embedding = np.array(face_encoder.compute_face_descriptor(rgb_image, shape))
    return embedding

for filename in os.listdir(REFERENCE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        path = os.path.join(REFERENCE_DIR, filename)
        embedding = get_face_embedding(path)
        if embedding is not None:
            known_faces.append((name, embedding))
        else:
            print(f"[WARNING] No face found in {filename}")

# Webcam loop
cap = cv2.VideoCapture(0)
people_recognized = set()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_frame)

    for face in faces:
        shape = shape_predictor(rgb_frame, face)
        embedding = np.array(face_encoder.compute_face_descriptor(rgb_frame, shape))

        label = "Unknown"
        best_distance = float("inf")

        for name, ref_embedding in known_faces:
            dist = distance.euclidean(ref_embedding, embedding)
            if dist < MATCH_THRESHOLD and dist < best_distance:
                label = name
                best_distance = dist

        # Play audio if this person was not recognized before
        if label != "Unknown" and label not in people_recognized:
            people_recognized.add(label)
            print(f"[INFO] Recognized: {label}")
            pygame.mixer.music.load(AUDIO_FILE)
            pygame.mixer.music.play()

        # Draw result
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Match", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
