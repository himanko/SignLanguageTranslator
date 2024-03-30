import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "SignLanguageTranslatorAPP"

list_of_files = [
    "model 2/.github/workflow/.gitkeep",
    f"model 2/src/{project_name}/__init__.py",
    f"model 2/src/{project_name}/components/__init__.py",
    f"model 2/src/{project_name}/config/__init__.py",
    f"model 2/src/{project_name}/config/configuration.py",
    f"model 2/src/{project_name}/utils/__init__.py",
    f"model 2/src/{project_name}/pipeline/__init__.py",
    f"model 2/src/{project_name}/entity/__init__.py",
    f"model 2/src/{project_name}/constants/__init__.py",
    "model 2/config/config.yaml",
    "model 2/model/__init__.py",
    "model 2/dvc.yaml",
    "model 2/params.yaml",
    "requirements.txt",
    "setup.py",
    "model 2/research/trails.ipynb",
    "model 2/templates/index.html"

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
                     
                     
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty File: {filepath}")
        
            
    else:
        logging.info(f"{filename} is already exist.")
                    