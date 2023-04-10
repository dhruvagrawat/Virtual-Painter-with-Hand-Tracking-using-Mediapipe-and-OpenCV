import cv2
import mediapipe as mp
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the drawing canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize the hand tracking module in Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the previous finger position
prev_x, prev_y = None, None

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while True:
        # Get the webcam feed
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the hand landmarks
        results = hands.process(image)

        # Map the hand landmarks to the canvas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the position of the wrist and index finger
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                finger_x, finger_y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

                # Calculate the distance between the wrist and index finger
                distance = np.sqrt((finger_x - wrist_x) ** 2 + (finger_y - wrist_y) ** 2)

                # Draw a line on the canvas if the finger is in drawing position
                if distance > 40 and finger_y < 400:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (finger_x, finger_y), (0, 0, 255), 5)
                    prev_x, prev_y = finger_x, finger_y
                else:
                    prev_x, prev_y = None, None

        # Overlay the canvas on top of the webcam feed
        alpha = 0.5
        cv2.addWeighted(canvas, alpha, frame[0:480, 0:640], 1 - alpha, 0, frame[0:480, 0:640])

        # Display the canvas and the webcam feed
        cv2.imshow('Virtual Painter', frame)

        # Exit the program if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
