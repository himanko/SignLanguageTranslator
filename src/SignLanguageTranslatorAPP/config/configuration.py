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
            root_dir=config.root_dir
            
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
            model_weight_dir=config.model_weight_dir
            
            
        )

        return model_config