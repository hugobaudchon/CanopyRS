from datetime import datetime
from typing import List, Optional, Any, Dict, Set
import uuid
import os
import sys
from pathlib import Path
import copy
import itertools

import torch

import wandb

from detectron2.data import DatasetCatalog, build_detection_train_loader, DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, BestCheckpointer, hooks
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils import comm

from detectron2.utils.events import get_event_storage
from fvcore.nn import get_bn_modules

from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.models.detector.train_detectron2.augmentation import AugmentationAdder
from canopyrs.engine.models.detector.train_detectron2.dataset import register_detection_dataset
from canopyrs.engine.models.detector.train_detectron2.hook import WandbWriterHook
from canopyrs.engine.models.detector.train_detectron2.lr_scheduler import build_lr_scheduler


def _resolve_architecture_path(architecture: str) -> Optional[Path]:
    """
    Resolve an architecture path that might live outside the Detectron2 model zoo.

    Returns a Path if the file exists either as-is or relative to the repo root,
    otherwise returns None.
    """
    arch_path = Path(architecture)
    if arch_path.is_file():
        return arch_path

    # Try to locate the repo root (marked by pyproject.toml or .git) and look relative to it.
    repo_root = None
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root:
        candidate = repo_root / architecture
        if candidate.is_file():
            return candidate

    return None


class TrainerWithValidation(DefaultTrainer):
    """Custom trainer class that includes periodic validation."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator = COCOEvaluator(
            dataset_name,
            output_dir=output_folder
        )
        return evaluator

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                augmentations=AugmentationAdder().get_augmentation_detectron2_train(cfg),   # Using my custom augmentations here
                is_train=True)
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=DatasetMapper(
                cfg,
                augmentations=AugmentationAdder().get_augmentation_detectron2_test(cfg),   # Using my custom augmentations here
                is_train=False
            )
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Build a learning rate scheduler for the optimizer.
        """
        lr_scheduler = build_lr_scheduler(
            lr_scheduler_name=cfg.SOLVER.LR_SCHEDULER_NAME,
            lr_steps=cfg.SOLVER.STEPS,
            lr_gamma=cfg.SOLVER.GAMMA,
            max_iter=cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            optimizer=optimizer
        )

        return lr_scheduler

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Override the test method so that after evaluation we log the mAP metrics to wandb.
        The COCOEvaluator typically returns a dict with keys like "bbox/AP".
        """
        results = super().test(cfg, model, evaluators)

        train_dataset_length = sum([len(DatasetCatalog.get(dataset_name)) for dataset_name in cfg.DATASETS.TRAIN])
        current_step = get_event_storage().iter
        current_epoch = round(current_step * cfg.SOLVER.IMS_PER_BATCH / train_dataset_length)
        wandb.log({"epoch": current_epoch})

        # Log results of first dataset to wandb
        if len(results) > 0:
            def log_ap_metrics(prefix, metrics):
                if metrics is None:
                    return
                if "AP" in metrics:
                    wandb.log({f"{prefix}/AP": metrics["AP"]})
                    wandb.log({f"{prefix}/AP50": metrics["AP50"]})
                    wandb.log({f"{prefix}/AP75": metrics["AP75"]})
                    wandb.log({f"{prefix}/APs": metrics["APs"]})
                    wandb.log({f"{prefix}/APm": metrics["APm"]})
                    wandb.log({f"{prefix}/APl": metrics["APl"]})
                else:
                    raise Exception(f"Warning: AP metric not found in evaluation results for {prefix}.")

            log_ap_metrics("bbox", results.get("bbox"))
            log_ap_metrics("segm", results.get("segm"))

        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            (
                hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def get_base_detectron2_model_cfg(config):
    cfg = get_cfg()

    # Load base configs for Faster R-CNN
    cfg.merge_from_file(model_zoo.get_config_file(config.architecture))

    # Load pre-trained model weights
    if config.backbone_model_pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.architecture)

    if config.checkpoint_path is not None:
        cfg.MODEL.WEIGHTS = config.checkpoint_path

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    if config.anchor_sizes is not None:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [list(s) for s in config.anchor_sizes]
    cfg.SOLVER.AMP.ENABLED = config.use_amp

    if config.box_score_thresh is not None:
        if cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST != config.box_score_thresh:
            print(f"Changing box score threshold from {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST} to {config.box_score_thresh}.")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.box_score_thresh
    if config.box_nms_thresh is not None:
        if cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST != config.box_nms_thresh:
            print(f"Changing box NMS threshold from {cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST} to {config.box_nms_thresh}.")
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.box_nms_thresh

    # Augmentations
    AugmentationAdder().modify_detectron2_augmentation_config(config, cfg)

    cfg.TEST.DETECTIONS_PER_IMAGE = config.box_predictions_per_image

    return cfg


def setup_trainer(train_dataset_names: List[str], valid_dataset_names: List[str], config: DetectorConfig, model_name: str, task):
    """
    Set up a basic Faster R-CNN trainer with default configurations.

    Parameters
    ----------
    train_dataset_names : List[str]
        Name(s) of the registered dataset(s) to use for training
    valid_dataset_names : List[str]
        Name(s) of the registered dataset(s) to use for validation
    config : DetectorConfig
        Configuration object for the detector model
    model_name : str

    Returns
    -------
    DefaultTrainer
        Configured train_detectron2 trainer
    """
    cfg = get_base_detectron2_model_cfg(config)

    dataset_length = sum([len(DatasetCatalog.get(dataset_name)) for dataset_name in train_dataset_names])

    # Dataset config
    cfg.DATASETS.TRAIN = tuple(train_dataset_names)
    cfg.DATASETS.TEST = tuple(valid_dataset_names)

    # Training config
    cfg.SEED = config.seed
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    print(f"Changing batch size from {cfg.SOLVER.IMS_PER_BATCH} to {config.batch_size}.")
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size  # This is the real "batch size"
    print(f"Changing base learning rate from {cfg.SOLVER.BASE_LR} to {config.lr}.")
    cfg.SOLVER.BASE_LR = config.lr
    print(f"Changing scheduler gamma from {cfg.SOLVER.GAMMA} to {config.scheduler_gamma}.")
    cfg.SOLVER.MAX_ITER = config.max_epochs * dataset_length // config.batch_size
    cfg.SOLVER.LOG_PERIOD = config.train_log_interval

    cfg.SOLVER.LR_SCHEDULER_NAME = config.scheduler_type
    if config.scheduler_epochs_steps is not None:
        steps = [step * dataset_length // config.batch_size for step in config.scheduler_epochs_steps]
        print(f"Changing scheduler steps from {cfg.SOLVER.STEPS} to {steps}.")
        cfg.SOLVER.STEPS = steps
    if config.scheduler_gamma is not None:
        print(f"Changing scheduler gamma from {cfg.SOLVER.GAMMA} to {config.scheduler_gamma}.")
        cfg.SOLVER.GAMMA = config.scheduler_gamma
    if config.scheduler_warmup_steps is not None:
        print(f"Changing scheduler warmup steps from {cfg.SOLVER.WARMUP_ITERS} to {config.scheduler_warmup_steps}.")
        cfg.SOLVER.WARMUP_ITERS = config.scheduler_warmup_steps
    if config.freeze_layers:
        print(f"Changing freeze layers from {cfg.MODEL.BACKBONE.FREEZE_AT} to {config.freeze_layers}.")
        cfg.MODEL.BACKBONE.FREEZE_AT = config.freeze_layers


    # Evaluation config
    cfg.TEST.EVAL_PERIOD = config.eval_epoch_interval * dataset_length // config.batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = None

    # Output directory
    output_dir = os.path.join(config.train_output_path, model_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir

    print(cfg)

    # Save config
    config.to_yaml(os.path.join(output_dir, "config.yaml"))

    # Create trainer
    trainer = TrainerWithValidation(cfg)
    trainer.resume_or_load(resume=False)

    if comm.is_main_process():
        trainer.register_hooks([
            WandbWriterHook(
                cfg=cfg,
                config=config,
                train_log_interval=config.train_log_interval,
                wandb_project_name=config.wandb_project,
                wandb_model_name=model_name,
                task=task
            ),
            BestCheckpointer(
                eval_period=cfg.TEST.EVAL_PERIOD,
                checkpointer=trainer.checkpointer,
                val_metric=config.main_metric,
                mode="max",
                file_prefix="model_best"
            )
        ])

    return trainer


def train_detectron2(config: DetectorConfig, task):
    """
    Train a Faster R-CNN model on the custom dataset.

    Parameters
    ----------
    config : DetectorConfig
        Configuration object for the detector model
    task : str
        The task type, e.g., 'detection' or 'segmentation'
    """

    print("Setting up datasets...")
    d2_train_datasets_name = register_detection_dataset(
        root_path=[f"{config.data_root_path}/{path}" for path in config.train_dataset_names],
        fold="train",
        force_binary_class=True if config.num_classes == 1 else False,
        segmentation_only=False
    )

    d2_valid_datasets_name = register_detection_dataset(
        root_path=[f"{config.data_root_path}/{path}" for path in config.valid_dataset_names],
        fold="valid",
        force_binary_class=True if config.num_classes == 1 else False,
        segmentation_only=(task == 'segmentation')
    )

    print(f"Setting up trainer for dataset(s): {d2_train_datasets_name}")
    u = uuid.uuid4()
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        model_name = f"{config.model}_{now}_{slurm_job_id}"
    else:
        model_name = f"{config.model}_{now}_{u.hex[:4]}"
    trainer = setup_trainer([d2_train_datasets_name], [d2_valid_datasets_name], config, model_name, task)

    print("Starting training...")
    trainer.train()
    print(f"Training completed. Model saved in {config.train_output_path}")
