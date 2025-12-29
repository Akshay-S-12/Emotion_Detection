import cv2
import mediapipe as mp
import pyttsx3
import time
import math

# ---------- TEXT TO SPEECH ----------
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('volume', 0.9)

prev_emotion = ""
last_spoken_time = 0
cooldown = 3

# ---------- FACE MESH ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

video = cv2.VideoCapture(0)

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    ret, img = video.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    emotion = "Neutral"

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        lm = {}

        for i, p in enumerate(face.landmark):
            lm[i] = (int(p.x * w), int(p.y * h))

        # Mouth
        mouth_w = dist(lm[61], lm[291])
        mouth_h = dist(lm[13], lm[14])

        # Eyes
        eye_open = dist(lm[159], lm[145])

        # Ratios
        mouth_ratio = mouth_h / mouth_w
        eye_ratio = eye_open / mouth_w

        # ---------- IMPROVED EMOTION LOGIC ----------
        if mouth_ratio > 0.30 and eye_ratio > 0.10:
            emotion = "Surprised"
        elif mouth_ratio > 0.20:
            emotion = "Happy"
        elif mouth_ratio < 0.10:
            emotion = "Sad"
        else:
            emotion = "Neutral"

        # ---------- SPEECH ----------
        now = time.time()
        if emotion != prev_emotion and now - last_spoken_time > cooldown:
            engine.say(emotion)
            engine.runAndWait()
            prev_emotion = emotion
            last_spoken_time = now

    cv2.putText(img, f"Emotion: {emotion}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Emotion Detection with Speech", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

