import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame

# Load models
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_image)

    if len(faces) == 0:
        raise ValueError("No face found in reference image.")

    shape = shape_predictor(rgb_image, faces[0])
    embedding = np.array(face_encoder.compute_face_descriptor(rgb_image, shape))
    return embedding

reference_embedding = get_face_embedding("reference.jpg")

# Start webcam
cap = cv2.VideoCapture(0)
is_match_detected = False
pygame.mixer.init()
pygame.mixer.music.load("technologia.mp3")
audio_played = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_frame)

    for face in faces:
        shape = shape_predictor(rgb_frame, face)
        embedding = np.array(face_encoder.compute_face_descriptor(rgb_frame, shape))
        dist = distance.euclidean(reference_embedding, embedding)

        if dist < 0.6:
            is_match_detected = True

        label = "MATCH" if dist < 0.6 else "Unknown"

        # Draw bounding box and label
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0) if label=="MATCH" else (0,0,255), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if is_match_detected and not audio_played:
        pygame.mixer.music.play()
        audio_played = True

    cv2.imshow("Face Match", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
