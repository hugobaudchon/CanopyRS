import wandb
import tempfile
import yaml
import os

from detectron2.engine import HookBase
from detectron2.utils import comm
from canopyrs.engine.models.detector.train_detectron2.utils import lazyconfig_to_dict


def clean_config(cfg):
    # Process dictionaries
    if isinstance(cfg, dict):
        new_cfg = {}
        for key, value in cfg.items():
            new_cfg[key] = clean_config(value)
        return new_cfg
    # Process lists
    elif isinstance(cfg, list):
        return [clean_config(item) for item in cfg]
    # Process tuples
    elif isinstance(cfg, tuple):
        return tuple(clean_config(item) for item in cfg)
    # For callables, return a module-qualified name
    elif callable(cfg):
        try:
            return f"{cfg.__module__}.{cfg.__name__}"
        except Exception:
            return str(cfg)
    # For simple types, just return them
    elif isinstance(cfg, (str, int, float, bool)) or cfg is None:
        return cfg
    # For everything else (e.g. custom objects), fallback to a string representation
    else:
        return str(cfg)



class WandbWriterHook(HookBase):
    """
    A simple Detectron2 hook to log training losses (and other metrics)
    to Weights & Biases (wandb) every few iterations.
    """
    def __init__(self, cfg, config, train_log_interval: int, wandb_project_name: str, wandb_model_name: str, task: str):
        self.cfg = cfg
        self.config = config
        self.train_log_interval = train_log_interval
        self.wandb_project_name = wandb_project_name
        self.wandb_model_name = wandb_model_name
        self.task = task
        assert task in ["detection", "segmentation"], "Task must be either 'detection' or 'segmentation'"

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

            if self.task == "segmentation":
                wandb.define_metric("segm/AP", summary="max")
                wandb.define_metric("segm/AP50", summary="max")
                wandb.define_metric("segm/AP75", summary="max")
                wandb.define_metric("segm/APs", summary="max")

            # Convert your lazy config to YAML
            yaml_config = yaml.dump(clean_config(lazyconfig_to_dict(self.cfg)), Dumper=yaml.SafeDumper)

            # Write YAML to a temporary file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
                tmp.write(yaml_config)
                tmp.flush()
                temp_filename = tmp.name  # path to the temporary file

            # Log the temporary file as an artifact
            artifact = wandb.Artifact(f"config_{self.wandb_model_name}", type="config")
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

            log_data = {
                key: value[0]
                for key, value in metrics.items()
                if "loss" in key and not key[-1].isdigit()
            }

            lr = self.trainer.storage.history("lr").latest()
            log_data["learning_rate"] = lr

            wandb.log(log_data, step=self.trainer.iter)

    def after_train(self):
        wandb.finish()
        print("wandb finished.")
