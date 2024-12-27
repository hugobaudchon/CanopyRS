import os
from pathlib import Path

config_folder_path = Path(os.path.abspath(__file__)).parent

predefined_configs = [item.name for item in config_folder_path.iterdir()
                      if item.is_dir() and item.name != "__pycache__"]