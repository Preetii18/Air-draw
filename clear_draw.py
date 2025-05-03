import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to check which fingers are up
def fingers_up(lm_list):
    fingers = []
    # Thumb (optional — skipping for now)
    
    # Index finger
    fingers.append(lm_list[8][1] < lm_list[6][1])  # tip.y < pip.y → finger is up
    # Middle
    fingers.append(lm_list[12][1] < lm_list[10][1])
    # Ring
    fingers.append(lm_list[16][1] < lm_list[14][1])
    # Pinky
    fingers.append(lm_list[20][1] < lm_list[18][1])

    return fingers

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

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cy, cx))  # id, y, x for easy logic

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) >= 21:
                fingers = fingers_up(lm_list)

                index_finger_tip = (lm_list[8][2], lm_list[8][1])  # x, y

                # Only draw if index is up and others are down
                if fingers[0] and not any(fingers[1:]):
                    x, y = index_finger_tip
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y

                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

    # Merge canvas and frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("AirCanvas - Draw with Index Finger", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
