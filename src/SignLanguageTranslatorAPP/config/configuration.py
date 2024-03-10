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
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        emergency_model_config = EmergencyModelConfig(
            root_dir=config.root_dir,
             
        )

        return emergency_model_config
    
    def get_include_model_config(self) -> IncludeModelConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        include_model_config = IncludeModelConfig(
            root_dir=config.root_dir,
             
        )

        return include_model_config
    
    def get_wlasl_model_config(self) -> EmergencyModelConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        wlasl_model_config = EmergencyModelConfig(
            root_dir=config.root_dir,
             
        )

        return wlasl_model_config