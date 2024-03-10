from SignLanguageTranslatorAPP.constants import *
from SignLanguageTranslatorAPP.utils.common import read_yaml, create_directories
from SignLanguageTranslatorAPP.entity.config_entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)

        create_directories([self.config.model_root])


    
    def get_emergency_model_config(self) -> EmergencyModelConfig:
        config = self.config.emergancy_model

        create_directories([config.root_dir])

        emergency_model_config = EmergencyModelConfig(
            root_dir=config.root_dir,
             
        )

        return emergency_model_config
    
    def get_include_model_config(self) -> IncludeModelConfig:
        config = self.config.Include_model

        create_directories([config.root_dir])

        include_model_config = IncludeModelConfig(
            root_dir=config.root_dir,
             
        )

        return include_model_config
    
    def get_wlasl_model_config(self) -> EmergencyModelConfig:
        config = self.config.WLASL_model

        create_directories([config.root_dir])

        wlasl_model_config = EmergencyModelConfig(
            root_dir=config.root_dir,
             
        )

        return wlasl_model_config
    

    class ConfigurationManager1:
        def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH):

            self.config = read_yaml(config_filepath)

            create_directories([self.config.artifacts_root])


        def get_preprocessing_config(self) -> PreprocessingConfig:
            config = self.config.preprocessing

            create_directories([config.root_dir])

            preprocessing_config = PreprocessingConfig(
                root_dir=config.root_dir,
                
            )

            return preprocessing_config