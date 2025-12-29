# 🎭 Real-Time Face Emotion Detection with Speech

This project detects **human emotions in real time** using a webcam and **MediaPipe Face Mesh**, and converts the detected emotion into **speech output** using Text-to-Speech.
The system works completely **in real time**, does **not require any pre-trained emotion datasets**.

---

## 📌 Features
- 🎥 Real-time emotion detection using webcam
- 😄 Detects **Happy, Sad, Surprised, Neutral**
- 🗣️ Speaks detected emotion using Text-to-Speech
- ⏱️ Cooldown mechanism to avoid repeated speech
- ⚡ Lightweight & fast (rule-based logic)

---

## 🛠️ Technologies Used
- **Python**
- **OpenCV**
- **MediaPipe (Face Mesh)**
- **pyttsx3 (Text-to-Speech)**

---

## 📂 Project Structure
Face-Emotion-Detection/
│
├── emotion_detection.py
├── README.md
└── requirements.txt


---

## 📦 Requirements
Install required libraries using:

```bash
pip install opencv-python mediapipe pyttsx3

