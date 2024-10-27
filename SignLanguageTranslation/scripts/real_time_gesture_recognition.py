import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Define gesture actions
actions = ["Alright", "Good Morning", "Good Afternoon", "Hello", "How Are You"]
gesture_text = ""

# Load model or define your gesture recognition logic
# For simplicity, we are using dummy predictions based on landmark positions

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Get coordinates of specific landmarks
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]

            # Convert to pixel values
            thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            middle_x, middle_y = int(middle_tip.x * frame_width), int(middle_tip.y * frame_height)

            # Gesture detection logic
            gesture_text = ""

            # Example gesture conditions
            if thumb_y < index_y:  # Example condition for "Alright"
                gesture_text = actions[0]  # Alright
            elif index_y < thumb_y and middle_y < thumb_y:  # Example condition for "Good Morning"
                gesture_text = actions[1]  # Good Morning
            elif middle_y < thumb_y:  # Example condition for "Good Afternoon"
                gesture_text = actions[2]  # Good Afternoon
            elif index_y < thumb_y and middle_y > thumb_y:  # Example condition for "Hello"
                gesture_text = actions[3]  # Hello
            elif abs(index_y - thumb_y) < 20:  # Example condition for "How Are You"
                gesture_text = actions[4]  # How Are You

    # Display the gesture text on the screen
    cv2.putText(frame, gesture_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
