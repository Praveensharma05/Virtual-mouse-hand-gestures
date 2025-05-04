import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
draw_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

last_click_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark
            draw_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

            # Get coordinates of thumb, index, middle
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)  # Index tip
            mx, my = int(lm[12].x * w), int(lm[12].y * h)  # Middle tip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)  # Thumb tip

            # Move cursor
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Visual guide lines
            cv2.line(frame, (ix, iy), (tx, ty), (255, 0, 0), 2)  # Thumb
            cv2.line(frame, (ix, iy), (mx, my), (0, 0, 255), 2)  # Middle

            # Left click gesture: Index and Thumb close
            if abs(ix - tx) < 40 and abs(iy - ty) < 40:
                if time.time() - last_click_time > 1:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Right click gesture: Index and Middle close
            elif abs(ix - mx) < 40 and abs(iy - my) < 40:
                if time.time() - last_click_time > 1:
                    pyautogui.rightClick()
                    last_click_time = time.time()
                    cv2.putText(frame, "Right Click", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
