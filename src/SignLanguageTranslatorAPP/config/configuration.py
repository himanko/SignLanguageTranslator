import os
from SignLanguageTranslatorAPP.constants import *
from SignLanguageTranslatorAPP.utils.common import read_yaml, create_directories
from SignLanguageTranslatorAPP.entity.config_entity import (LandmarksExtractionConfig,
                                                            PreprocessingConfig,
                                                            ModelConfig
                                                            )

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_preprocessing_config(self) -> PreprocessingConfig:
        config = self.config.preprocessing

        create_directories([config.root_dir])

        preprocessing_config = PreprocessingConfig(
            root_dir=Path(config.root_dir),
            params_FIXED_FRAMES=self.params.FIXED_FRAMES,
            params_ROWS_PER_FRAME=self.params.ROWS_PER_FRAME,
            params_UNITS=self.params.UNITS

            
        )

        return preprocessing_config
    
    def get_landmarks_extraction_config(self) -> LandmarksExtractionConfig:
        config = self.config.landmarks_extraction

        create_directories([config.root_dir])

        landmarks_extraction_config = LandmarksExtractionConfig(
            root_dir=config.root_dir,
            output_dir=config.output_dir
            
        )

        return landmarks_extraction_config
    
    def get_model_config(self) -> ModelConfig:
        config = self.config.model

        create_directories([config.root_dir])

        model_config = ModelConfig(
            root_dir=config.root_dir,
            model_weight_dir=config.model_weight_dir,
            params_CLASSIFIER_DROPOUT_RATIO=self.params.CLASSIFIER_DROPOUT_RATIO,
            params_EMBEDDING_DROPOUT=self.params.EMBEDDING_DROPOUT,
            params_MLP_DROPOUT_RATIO=self.params.MLP_DROPOUT_RATIO,
            params_MLP_RATIO=self.params.MLP_RATIO,
            params_N_EPOCHS=self.params.N_EPOCHS,
            params_NUM_BLOCKS=self.params.NUM_BLOCKS,
            params_UNITS=self.params.UNITS,
        
            
            
        )

        return model_config