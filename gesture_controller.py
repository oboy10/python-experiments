import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
model = joblib.load("gesture_model.pkl")

cap = cv2.VideoCapture(0)

actions = {
    "volume_up": lambda: pyautogui.press("volumeup"),
    "pause": lambda: pyautogui.press("space"),
    "next": lambda: pyautogui.hotkey("command", "right"),  # MacOS next track
}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append([lm.x, lm.y, lm.z])
            data = np.array(coords).flatten().reshape(1, -1)

            gesture = model.predict(data)[0]
            cv2.putText(frame, gesture, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if gesture in actions:
                actions[gesture]()

    cv2.imshow("Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
