import os
import torch
import numpy as np
import pandas as pd
from mapping import signs
from SignLanguageTranslatorAPP.components.preprocessing import *
from SignLanguageTranslatorAPP.config.configuration import ConfigurationManager
from SignLanguageTranslatorAPP.components.landmarks_extraction import load_process_predict
from SignLanguageTranslatorAPP.components.model import PrepareBaseModel
from SignLanguageTranslatorAPP import logger


STAGE_NAME = "Single stage"

class PradictionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            # Get Base Model configuratio
            landmarks_extraction_config = config.get_landmarks_extraction_config()
            # Get Base Model configuratio
            base_model_config = config.get_base_model_config()
            
            
            # Initialize LandmarksExtraction with the configuration
            landmarks_extraction = load_process_predict(config=landmarks_extraction_config)
            landmarks_extraction.load_process_predict
            
            # Initialize model with the configuration
            prepare_base_model = PrepareBaseModel(config=base_model_config)
            prepare_base_model.load_model_weights
            prepare_base_model.get_base_model
            model = prepare_base_model.apply

            # Load a single file for visualizing
            df = pd.read_parquet(f'model 1/artifacts/landmarks/{id}.parquet')
            df.sample(10)
            # Load parquet file and convert it to required shape
            
            x_in = torch.tensor(load_relevant_data_subset(f'model 1/artifacts/landmarks/{id}.parquet'))
            feature_preprocess = FeaturePreprocess()
            print(feature_preprocess(x_in).shape, x_in[0])

            inputX = feature_preprocess(x_in)
            inputX = inputX.cpu().detach().numpy()

            inputX = np.expand_dims(inputX, axis=0)

            preds = model.predict(inputX)

            ind=np.argmax(preds)

            return signs[ind]

        except Exception as e:
            logger.exception(f"Error in BaseModelPipeline: {e}")
            raise e
        




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PradictionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e