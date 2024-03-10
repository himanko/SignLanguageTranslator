import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from SignLanguageTranslatorAPP.utils.common import IDX_MAP
from SignLanguageTranslatorAPP import logger

FIXED_FRAMES = 30

class FeaturePreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        try:
            n_frames = x_in.shape[0]

            # Normalization to a common mean
            x_in = x_in - x_in[~torch.isnan(x_in)].mean(0, keepdim=True)
            x_in = x_in / x_in[~torch.isnan(x_in)].std(0, keepdim=True)

            # Landmarks reduction
            idx_map = IDX_MAP()
            lips = x_in[:, idx_map['lips']]
            lhand = x_in[:, idx_map['left_hand']]
            pose = x_in[:, idx_map['upper_body']]
            rhand = x_in[:, idx_map['right_hand']]
            x_in = torch.cat([lips, lhand, pose, rhand], 1)  # (n_frames, n_landmarks, 3)

            # Replace nan with 0 before Interpolation
            x_in[torch.isnan(x_in)] = 0

            # If n_frames < k, use linear interpolation,
            # else, use nearest neighbor interpolation
            x_in = x_in.permute(2, 1, 0)  # (3, n_landmarks, n_frames)
            if n_frames < FIXED_FRAMES:
                x_in = F.interpolate(x_in, size=(FIXED_FRAMES), mode='linear')
            else:
                x_in = F.interpolate(x_in, size=(FIXED_FRAMES), mode='nearest-exact')

            return x_in.permute(2, 1, 0)  # (n_frames, n_landmarks, 3)

        except Exception as e:
            raise e

    

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    try:
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        # Convert NaN values to 0
        data = data.fillna(0)
        n_frames = int(len(data) / ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
        return data.astype(np.float32)
    
    except Exception as e:
            raise e