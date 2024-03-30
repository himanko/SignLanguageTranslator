import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from SignLanguageTranslatorAPP.utils.common import IDX_MAP
from SignLanguageTranslatorAPP.config.configuration import PreprocessingConfig



class FeaturePreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        n_frames = x_in.shape[0]

        # Normalization to a common mean
        x_in = x_in - x_in[~torch.isnan(x_in)].mean(0,keepdim=True) 
        x_in = x_in / x_in[~torch.isnan(x_in)].std(0, keepdim=True)

        # Landmarks reduction
        lips     = x_in[:, IDX_MAP['lips']]
        lhand    = x_in[:, IDX_MAP['left_hand']]
        pose     = x_in[:, IDX_MAP['upper_body']]
        rhand    = x_in[:, IDX_MAP['right_hand']]
        x_in = torch.cat([lips,
                          lhand,
                          pose,
                          rhand], 1) # (n_frames, n_landmarks, 3)

        # Replace nan with 0 before Interpolation
        x_in[torch.isnan(x_in)] = 0

        # If n_frames < k, use linear interpolation,
        # else, use nearest neighbor interpolation
        x_in = x_in.permute(2,1,0) #(3, n_landmarks, n_frames)
        if n_frames < PreprocessingConfig.params_FIXED_FRAMES:
            x_in = F.interpolate(x_in, size=(PreprocessingConfig.params_FIXED_FRAMES), mode= 'linear')
        else:
            x_in = F.interpolate(x_in, size=(PreprocessingConfig.params_FIXED_FRAMES), mode= 'nearest-exact')

        return x_in.permute(2,1,0) # (n_frames, n_landmarks, 3)
    

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / PreprocessingConfig.params_ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, PreprocessingConfig.params_ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)