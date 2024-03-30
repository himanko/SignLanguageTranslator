import os
import cv2
import torch
import numpy as np 
import pandas as pd
import mediapipe as mp
from SignLanguageTranslatorAPP.config.configuration import LandmarksExtractionConfig
from SignLanguageTranslatorAPP.components.preprocessing import FeaturePreprocess, load_relevant_data_subset


class load_process_predict:
    def __init__(self, config: LandmarksExtractionConfig):
        self.config: config
        self.output_folder: config.output_dir
    def load_process_predict(self, video_path,id):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic

        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)

        video_file = video_path
        cap = cv2.VideoCapture(video_file)

        video_frames = []
        frame_no = 0
        while cap.isOpened():
            
            success, image = cap.read()

            if not success: break
            image = cv2.resize(image, dsize=None, fx=4, fy=4)
            height,width,_ = image.shape

            #print(image.shape)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = holistic.process(image)

            data = [] 
            fy = height/width

            # -----------------------------------------------------
            if result.left_hand_landmarks is None:
                for i in range(21):  #
                    data.append({
                        'type': 'left_hand',
                        'landmark_index': i,
                        'x': np.nan,
                        'y': np.nan,
                        'z': np.nan,
                    })
            else:
                assert (len(result.left_hand_landmarks.landmark) == 21)
                for i in range(21):  #
                    xyz = result.left_hand_landmarks.landmark[i]
                    data.append({
                        'type': 'left_hand',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })

            # -----------------------------------------------------
            if result.right_hand_landmarks is None:
                for i in range(21):  #
                    data.append({
                        'type': 'right_hand',
                        'landmark_index': i,
                        'x': np.nan,
                        'y': np.nan,
                        'z': np.nan,
                    })
            else:
                assert (len(result.right_hand_landmarks.landmark) == 21)
                for i in range(21):  #
                    xyz = result.right_hand_landmarks.landmark[i]
                    data.append({
                        'type': 'right_hand',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
                zz = 0
            frame_df = pd.DataFrame(data)
            frame_df.loc[:,'frame'] =  frame_no
            video_frames.append(frame_df)

            #=========================
            frame_no += 1

        video_df = pd.concat(video_frames, ignore_index=True)
        # video_df_list.append(video_df)
        self.output_folder = self.output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        parquet_file_path = os.path.join(self.output_folder, f"{id}.parquet")
        # os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)  # Create directories if they don't exist
        video_df.to_parquet(parquet_file_path)

        cap.release()
        holistic.close()

    # def apply(self):
    #     # reading the parquet file and feature generation

    #     # Load a single file for visualizing
    #     df = pd.read_parquet(f'model 1/artifacts/landmarks/{id}.parquet')
    #     df.sample(10)
    #     # Load parquet file and convert it to required shape
        
    #     x_in = torch.tensor(load_relevant_data_subset(f'model 1/artifacts/landmarks/{id}.parquet'))
    #     feature_preprocess = FeaturePreprocess()
    #     print(feature_preprocess(x_in).shape, x_in[0])

    #     inputX = feature_preprocess(x_in)
    #     inputX = inputX.cpu().detach().numpy()

    #     inputX = np.expand_dims(inputX, axis=0)

    #     preds = model.predict(inputX)

    #     ind=np.argmax(preds)

    #     return signs[ind]