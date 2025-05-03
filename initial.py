import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Setup camera
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = lm_list[8]

            x, y = index_finger_tip

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0

    # Merge the canvas with the frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("AirCanvas", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):  # Press 'c' to clear the canvas
        canvas = np.zeros_like(frame)
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

