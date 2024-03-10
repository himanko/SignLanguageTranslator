import os
import pandas as pd
from SignLanguageTranslatorAPP.entity.config_entity import PreprocessingConfig
from SignLanguageTranslatorAPP.utils.preprocessing_utils import *
from SignLanguageTranslatorAPP.utils.common import IDX_MAP
from SignLanguageTranslatorAPP import logger




class Preprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.feature_preprocess = FeaturePreprocess()

    def get_generate_labels(self):
        try:
     
            signs = []
            paths = []

            # root_folder = self.config.root_dir
            root_folder = os.listdir(self.config.root_dir)

            for class_name in root_folder:
                class_path = os.path.join(root_folder, class_name)
                if not os.path.isdir(class_path):
                    continue

                for file_name in os.listdir(class_path):
                    phrase = class_name
                    # path = f"/kaggle/input/landmarks-wlasl/landmarks_files_(alphabatically)/WLASL_300/{class_name}/{file_name}"
                    path = os.path.join(self.config.root_dir, class_name, file_name)
                    signs.append(phrase)
                    paths.append(path)
            data = {
                'sign': signs,
                'path': paths,
            }

            train_df = pd.DataFrame(data)

            # Save the CSV file inside self.config.com_dir
            csv_filename = os.path.join(self.config.com_dir, 'train.csv')

            train_df.to_csv(csv_filename, index=False)

            # Convert Signs to Orginal Encodings
            train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes

            # Dictionaries to translate sign <-> ordinal encoded sign
            SIGN2ORD = train_df[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
            ORD2SIGN = train_df[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

            # # Checking the output of SIGN2ORD and ORD2SIGN
            # print(SIGN2ORD)  # Translate sign name to ordinal encoding
            # print(ORD2SIGN)  # Translate ordinal encoding to sign name

            return train_df, SIGN2ORD, ORD2SIGN

        except Exception as e:
            raise e
        

