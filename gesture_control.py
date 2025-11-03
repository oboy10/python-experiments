import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_hand_open(hand_landmarks):
    """Return True if hand looks open based on finger positions."""
    tips = [4, 8, 12, 16, 20]
    open_count = 0
    for tip in tips[1:]:  # skip thumb for simplicity
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_count += 1
    return open_count >= 3  # at least 3 fingers open

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_hand_open(hand_landmarks):
                pyautogui.press("volumeup")
            else:
                pyautogui.press("space")

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
