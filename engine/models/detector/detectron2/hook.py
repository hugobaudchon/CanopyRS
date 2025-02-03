import wandb

from detectron2.engine import HookBase
from engine.models.detector.detectron2.utils import lazyconfig_to_dict


class WandbWriterHook(HookBase):
    """
    A simple Detectron2 hook to log training losses (and other metrics)
    to Weights & Biases (wandb) every few iterations.
    """
    def __init__(self, cfg, train_log_interval: int, wandb_project_name: str, wandb_model_name: str):
        self.cfg = cfg
        self.train_log_interval = train_log_interval
        self.wandb_project_name = wandb_project_name
        self.wandb_model_name = wandb_model_name

    def before_train(self):
        # Initialize wandb with the trainer’s config.
        # (Replace "your_project_name" with your wandb project name.)
        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_model_name,
            config=lazyconfig_to_dict(self.cfg)
        )
        print("wandb initialized.")

    def after_step(self):
        # Log training loss every train_log_interval iterations.
        if self.trainer.iter % self.train_log_interval == 0:
            # The Detectron2 storage collects a bunch of metrics.
            # Here we assume that "total_loss" is logged.
            # (You can add additional keys if desired.)
            metrics = self.trainer.storage.latest()
            if "total_loss" in metrics:
                wandb.log({"total_loss": metrics["total_loss"][0]},
                          step=self.trainer.iter)

    def after_train(self):
        wandb.finish()
        print("wandb finished.")
