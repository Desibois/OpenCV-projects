import cv2
import mediapipe as mp
import numpy as np
import win32api
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video = cv2.VideoCapture(0)
click = 0

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                    win32api.SetCursorPos((index_x*5, index_y*4))
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                    distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

                    if distance < 30:
                        click += 1
                        if click % 5 == 0:
                            print("Click: " + str(click))
                            pyautogui.click()
                except:
                    pass

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()
