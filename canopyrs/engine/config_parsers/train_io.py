from canopyrs.engine.config_parsers.base import BaseConfig


class TrainIOConfig(BaseConfig):
    wandb_project: str = 'detector'
    train_subfolder_selection: str = None
    test_subfolder_selection: str = None
    train_aoi_name: str = 'train'
    valid_aoi_name: str = 'valid'