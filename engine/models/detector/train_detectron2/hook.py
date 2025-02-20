import wandb
import tempfile
import yaml
import os

from detectron2.engine import HookBase
from detectron2.utils import comm
from engine.models.detector.train_detectron2.utils import lazyconfig_to_dict


def clean_config(cfg):
    if isinstance(cfg, dict):
        new_cfg = {}
        for key, value in cfg.items():
            # If the value is callable (e.g., a function), convert it to a string.
            if callable(value):
                try:
                    new_cfg[key] = f"{value.__module__}.{value.__name__}"
                except Exception:
                    new_cfg[key] = str(value)
            else:
                new_cfg[key] = clean_config(value)
        return new_cfg
    elif isinstance(cfg, list):
        return [clean_config(item) for item in cfg]
    elif isinstance(cfg, tuple):
        return tuple(clean_config(item) for item in cfg)
    else:
        return cfg


class WandbWriterHook(HookBase):
    """
    A simple Detectron2 hook to log training losses (and other metrics)
    to Weights & Biases (wandb) every few iterations.
    """
    def __init__(self, cfg, config, train_log_interval: int, wandb_project_name: str, wandb_model_name: str):
        self.cfg = cfg
        self.config = config
        self.train_log_interval = train_log_interval
        self.wandb_project_name = wandb_project_name
        self.wandb_model_name = wandb_model_name

    def before_train(self):
        # Initialize wandb with the trainer’s config.
        # (Replace "your_project_name" with your wandb project name.)
        if comm.is_main_process():
            run = wandb.init(
                project=self.wandb_project_name,
                name=self.wandb_model_name,
                config=self.config.dict()
            )

            wandb.define_metric("total_loss", summary="min")
            wandb.define_metric("bbox/AP", summary="max")
            wandb.define_metric("bbox/AP50", summary="max")
            wandb.define_metric("bbox/AP75", summary="max")
            wandb.define_metric("bbox/APs", summary="max")

            # Convert your lazy config to YAML
            yaml_config = yaml.dump(clean_config(lazyconfig_to_dict(self.cfg)), Dumper=yaml.SafeDumper)

            # Write YAML to a temporary file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
                tmp.write(yaml_config)
                tmp.flush()
                temp_filename = tmp.name  # path to the temporary file

            # Log the temporary file as an artifact
            artifact = wandb.Artifact("second_config", type="config")
            artifact.add_file(temp_filename)
            run.log_artifact(artifact)
            artifact.wait()  # blocks until upload is complete
            os.remove(temp_filename)

            print("wandb initialized.")

    def after_step(self):
        # Log training loss every train_log_interval iterations.
        if self.trainer.iter % self.train_log_interval == 0 and comm.is_main_process():
            # Retrieve the metrics stored in Detectron2's storage
            metrics = self.trainer.storage.latest()

            # Retrieve total loss if available
            log_data = {}
            if "total_loss" in metrics:
                log_data["total_loss"] = metrics["total_loss"][0]

            lr = self.trainer.storage.history("lr").latest()
            log_data["learning_rate"] = lr

            wandb.log(log_data, step=self.trainer.iter)

    def after_train(self):
        wandb.finish()
        print("wandb finished.")
