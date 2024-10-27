import os
import cv2
import json
import numpy as np
import mediapipe as mp

def extract_pose_data(video_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(video_path)
    pose_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_data = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                pose_data.append(landmark_data)

    cap.release()
    return pose_data

def save_pose_data(folder_path):
    gesture_data = {}
    for gesture in os.listdir(folder_path):
        gesture_folder = os.path.join(folder_path, gesture)
        if os.path.isdir(gesture_folder):
            pose_samples = []
            for video in os.listdir(gesture_folder):
                video_path = os.path.join(gesture_folder, video)
                pose_samples.append(extract_pose_data(video_path))
            gesture_data[gesture] = pose_samples

    with open('gesture_data.json', 'w') as json_file:
        json.dump(gesture_data, json_file)

if __name__ == "__main__":
    dataset_folder = "./dataset" 
    save_pose_data(dataset_folder)
