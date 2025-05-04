import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Initialize MediaPipe and webcam
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Cursor smoothing setup
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothening = 5

# Click cooldowns
click_cooldown = 0.5
last_left_click = time.time()
last_right_click = time.time()

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            lm = handLms.landmark

            # Get finger tip positions
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)     # Index tip
            mx, my = int(lm[12].x * w), int(lm[12].y * h)   # Middle tip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)     # Thumb tip

            # Map index finger to screen
            screen_x = np.interp(lm[8].x, [0, 1], [0, screen_w])
            screen_y = np.interp(lm[8].y, [0, 1], [0, screen_h])

            # Smooth cursor movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw cursor point
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), cv2.FILLED)

            # --- Left Click (Pinch index + thumb) ---
            distance = math.hypot(tx - ix, ty - iy)
            if distance < 40:
                if time.time() - last_left_click > click_cooldown:
                    pyautogui.click()
                    last_left_click = time.time()
                    cv2.circle(frame, (ix, iy), 12, (0, 0, 255), cv2.FILLED)

            # --- Right Click (Index + Middle close together) ---
            right_click_dist = math.hypot(ix - mx, iy - my)
            if right_click_dist < 40:
                if time.time() - last_right_click > click_cooldown:
                    pyautogui.rightClick()
                    last_right_click = time.time()
                    cv2.circle(frame, (mx, my), 12, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
