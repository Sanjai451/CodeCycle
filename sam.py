import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Initialize RL model for gesture recognition
# Define DQN model architecture
def create_dqn_model(input_shape, action_space):
    model = Sequential([
        Dense(64, input_shape=input_shape, activation="relu"),
        Dense(32, activation="relu"),
        Dense(action_space, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Initialize model, parameters
state_size = (42,)  # Assume 21 hand landmarks * 2 (x, y) coordinates
action_space = 5  # Number of gestures
model = create_dqn_model(state_size, action_space)
memory = deque(maxlen=2000)
gamma = 0.95  # Discount rate
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

# Function to preprocess hand landmarks
def preprocess_landmarks(landmarks, frame_width, frame_height):
    return np.array([
        [int(lm.x * frame_width), int(lm.y * frame_height)] for lm in landmarks
    ]).flatten()

# Start video capture
cap = cv2.VideoCapture(0)

# Variable to store detected gesture
gesture_text = "Sign Language Translator"

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
            state = preprocess_landmarks(landmarks, frame_width, frame_height).reshape(1, -1)

            # RL decision for gesture classification
            if np.random.rand() <= epsilon:
                action = random.randint(0, action_space - 1)  # Explore action space
            else:
                q_values = model.predict(state)
                action = np.argmax(q_values[0])  # Choose action with highest Q-value

            # Mapping action to gesture
            gestures = ["Thumbs Up", "Peace Sign", "Fist", "OK Sign", "Stop"]
            gesture_text = gestures[action]

            # Draw gesture text on frame
            cv2.putText(frame, f"Gesture: {gesture_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Assume some reward feedback mechanism for supervised training
            # reward = 1 if correct_classification(action) else -1
            # memory.append((state, action, reward, next_state))

            # Train DQN model
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next in minibatch:
                    target = r
                    if s_next is not None:
                        target = r + gamma * np.amax(model.predict(s_next)[0])
                    target_f = model.predict(s)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)

            # Update epsilon to decrease exploration over time
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

    # Show video frame
    cv2.imshow("Sign Language Translator with RL", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
