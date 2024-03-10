# Import necessary libraries
import tempfile
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
import torch
from SignLanguageTranslatorAPP.utils.preprocessing_utils import FeaturePreprocess, load_relevant_data_subset, FIXED_FRAMES
from SignLanguageTranslatorAPP.utils.common import IDX_MAP



# Function to extract landmarks using MediaPipe Holistic
def extract_landmarks(frame_rgb):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)
    results = holistic.process(frame_rgb)
    landmarks = np.zeros((1, 543, 3))  # Initialize array to hold all landmarks
    for landmark_type, landmarks_list in zip(['lips', 'left_hand', 'right_hand', 'upper_body'], 
                                             [results.face_landmarks, results.left_hand_landmarks, 
                                              results.right_hand_landmarks, results.pose_landmarks]):
        if landmarks_list is not None:
            for i, landmark in enumerate(landmarks_list.landmark):
                landmarks[0, i + IDX_MAP()[landmark_type][0]] = [landmark.x, landmark.y, landmark.z]
    return landmarks


class PreprocessFrame:
    def __init__(self) -> None:
        self.feature_preprocess = FeaturePreprocess()

    def preprocess_frame(self, frame):
        try:
            # Convert frame to RGB (MediaPipe requires RGB input)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks using MediaPipe Holistic
            landmarks_data = extract_landmarks(frame_rgb)

            # # Landmarks reduction using IDX_MAP
            # idx_map = IDX_MAP()
            # lips = landmarks_data[:, idx_map['lips']]
            # lhand = landmarks_data[:, idx_map['left_hand']]
            # pose = landmarks_data[:, idx_map['upper_body']]
            # rhand = landmarks_data[:, idx_map['right_hand']]
            # landmarks_data = np.concatenate([lips, lhand, pose, rhand], axis=1)

            # Replace NaN values with 0
            # landmarks_data[np.isnan(landmarks_data)] = 0

            # Preprocess the frame using FeaturePreprocess
            preprocessed_data = self.feature_preprocess(torch.tensor(landmarks_data)).cpu().numpy()

            return preprocessed_data

        except Exception as e:
            raise e

