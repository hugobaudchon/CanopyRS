#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from pprint import pprint
import uuid

from canopyrs.engine.config_parsers.detector import DetectorConfig
import torch
import wandb
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate, CfgNode
from detectron2.data import DatasetCatalog
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter, get_event_storage
)
from detectron2.checkpoint import DetectionCheckpointer

import detrex
from detrex.modeling import ema
from canopyrs.engine.models.detector.train_detectron2.augmentation import AugmentationAdder
from canopyrs.engine.models.detector.train_detectron2.dataset import register_detection_dataset
from canopyrs.engine.models.detector.train_detectron2.hook import WandbWriterHook
from canopyrs.engine.models.detector.train_detectron2.lr_scheduler import build_lr_scheduler
from canopyrs.engine.models.detector.train_detectron2.utils import lazyconfig_to_dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            amp=False,
            clip_grad_params=None,
            grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


def log_model_param_stats(model, logger):
    def _gather_params(module):
        params = []
        seen = set()
        for param in module.parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(param)
        return params

    def _param_stats(params):
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen

    def _log_stats(label, params):
        total, trainable, frozen = _param_stats(params)
        msg = f"{label} parameters - total: {total:,}, trainable: {trainable:,}, frozen: {frozen:,}"
        if logger:
            logger.info(msg)
        print(msg)

    model_params = _gather_params(model)
    backbone_module = getattr(model, "backbone", None)
    head_module = None
    for attr in ("head", "roi_heads", "sem_seg_head", "bbox_head", "mask_head"):
        if hasattr(model, attr):
            head_module = getattr(model, attr)
            break

    backbone_params = _gather_params(backbone_module) if backbone_module else []
    if head_module is not None:
        head_params = _gather_params(head_module)
    else:
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model_params if id(p) not in backbone_ids]

    _log_stats("Model", model_params)
    if backbone_params:
        _log_stats("Backbone", backbone_params)
    else:
        logger.info("Backbone parameters - module not found")
    _log_stats("Head", head_params)


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("train_detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret

    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)

        if comm.is_main_process():
            ret_bbox = ret["bbox"]
            wandb.log({"bbox/AP": ret_bbox["AP"]})
            wandb.log({"bbox/AP50": ret_bbox["AP50"]})
            wandb.log({"bbox/AP75": ret_bbox["AP75"]})
            wandb.log({"bbox/APs": ret_bbox["APs"]})
            wandb.log({"bbox/APm": ret_bbox["APm"]})
            wandb.log({"bbox/APl": ret_bbox["APl"]})
            if "segm" in ret:
                ret_segm = ret["segm"]
                wandb.log({"segm/AP": ret_segm["AP"]})
                wandb.log({"segm/AP50": ret_segm["AP50"]})
                wandb.log({"segm/AP75": ret_segm["AP75"]})
                wandb.log({"segm/APs": ret_segm["APs"]})
                wandb.log({"segm/APm": ret_segm["APm"]})
                wandb.log({"segm/APl": ret_segm["APl"]})

            # Logging epoch number
            if isinstance(cfg.dataloader.train.dataset.names, str):
                train_dataset_length = len(DatasetCatalog.get(cfg.dataloader.train.dataset.names))
            else:
                train_dataset_length = sum([len(DatasetCatalog.get(dataset_name)) for dataset_name in cfg.dataloader.train.dataset.names])
            current_step = get_event_storage().iter
            current_epoch = round(current_step * cfg.dataloader.train.total_batch_size / train_dataset_length)
            wandb.log({"epoch": current_epoch})

        return ret


def do_train(args, cfg, config: DetectorConfig, task):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("train_detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    # instantiate optimizer
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # build training loader
    try:
        train_loader = instantiate(cfg.dataloader.train)
    except TypeError as e:
        del cfg.dataloader.train.mapper.augmentation_with_crop
        train_loader = instantiate(cfg.dataloader.train)

    # create ddp model
    model = create_ddp_model(model, **cfg.train.ddp)

    # build model ema
    ema.may_build_model_ema(cfg, model)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    log_model_param_stats(model, logger)

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]

    scheduler = build_lr_scheduler(
        lr_scheduler_name=cfg.lr_multiplier.scheduler.name,
        lr_steps=cfg.lr_multiplier.scheduler.steps,
        lr_gamma=cfg.lr_multiplier.scheduler.gamma,
        max_iter=cfg.train.max_iter,
        warmup_factor=cfg.lr_multiplier.warmup_factor,
        warmup_iters=cfg.lr_multiplier.warmup_steps,
        warmup_method=cfg.lr_multiplier.warmup_method,
        optimizer=optim
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=scheduler),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.BestCheckpointer(
                eval_period=cfg.train.eval_period,
                checkpointer=checkpointer,
                val_metric=config.main_metric,
                mode="max",
                file_prefix="model_best"
            ) if comm.is_main_process() else None,
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            WandbWriterHook(cfg=cfg,
                            config=config,
                            train_log_interval=cfg.train.log_period,
                            wandb_project_name=cfg.train.wandb.params.project,
                            wandb_model_name=cfg.train.wandb.params.name,
                            task=task)
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)



def train_detrex(config, task):
    u = uuid.uuid4()
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        model_name = f"{config.model}_{now}_{slurm_job_id}"
    else:
        model_name = f"{config.model}_{now}_{u.hex[:4]}"

    launch(
        _train_detrex_process,
        torch.cuda.device_count(),
        num_machines=1,
        machine_rank=0,
        dist_url="env://",
        args=(config, model_name, task),
    )


def get_base_detrex_model_cfg(config):
    detrex_root = Path(next(iter(detrex.__path__))).resolve()
    if str(detrex_root) not in sys.path:
        sys.path.insert(0, str(detrex_root))

    # fixing architecture path from old CanopyRS versions
    if config.architecture == "dino-swin/dino_swin_large_384_5scale_36ep.py":
        config.architecture = "dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py"

    # loading base config
    cfg = LazyConfig.load(str(detrex_root / 'projects' / config.architecture))
    cfg.train.init_checkpoint = config.checkpoint_path

    # dino
    if hasattr(cfg.model, 'num_classes'):
        cfg.model.num_classes = config.num_classes
    elif hasattr(cfg.model, 'params') and hasattr(cfg.model.params, 'num_classes'):
        cfg.model.params.num_classes = config.num_classes

    # mask2former
    if hasattr(cfg.model, "sem_seg_head") and hasattr(cfg.model.sem_seg_head, "num_classes"):
        cfg.model.sem_seg_head.num_classes = config.num_classes

    if (
        hasattr(cfg.model, "sem_seg_head")
        and hasattr(cfg.model.sem_seg_head, "transformer_predictor")
        and hasattr(cfg.model.sem_seg_head.transformer_predictor, "num_classes")
    ):
        cfg.model.sem_seg_head.transformer_predictor.num_classes = config.num_classes

    # optimizer
    if hasattr(cfg.model, "criterion") and hasattr(cfg.model.criterion, "num_classes"):
        cfg.model.criterion.num_classes = config.num_classes

    # Custom Augmentations
    augmentation_adder = AugmentationAdder()
    cfg.dataloader.train.mapper.augmentation = augmentation_adder.get_augmentation_detrex_train(config)
    cfg.dataloader.train.mapper.augmentation_with_crop = None   # we have our own set of augmentations, including cropping, in augmentation_adder
    cfg.dataloader.test.mapper.augmentation = augmentation_adder.get_augmentation_detrex_test(config)

    # Enable AMP (mixed-precision).
    cfg.train.amp.enabled = config.use_amp

    return cfg


def _train_detrex_process(config, model_name, task):
    print("Setting up datasets...")
    d2_train_datasets_name = register_detection_dataset(
        root_path=[f"{config.data_root_path}/{path}" for path in config.train_dataset_names],
        fold="train",
        force_binary_class=True if config.num_classes == 1 else False,
        segmentation_only=False # sometimes model can be trained on both detection and segmentation datasets
    )

    d2_valid_datasets_name = register_detection_dataset(
        root_path=[f"{config.data_root_path}/{path}" for path in config.valid_dataset_names],
        fold="valid",
        force_binary_class=True if config.num_classes == 1 else False,
        segmentation_only=False # TODO change this and add parameter box_only_allowed instead to use this if set to False: (task == 'segmentation')  # evaluation should only happen on masks for segmentation task
    )

    dataset_length = len(DatasetCatalog.get(d2_train_datasets_name))
    output_path = os.path.join(config.train_output_path, model_name)
    os.makedirs(output_path, exist_ok=True)

    cfg = get_base_detrex_model_cfg(config)

    if hasattr(cfg.model, 'metadata'):
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(d2_train_datasets_name)
        cfg.model.metadata = metadata

    cfg.dataloader.train.dataset.names = d2_train_datasets_name
    cfg.dataloader.test.dataset.names = d2_valid_datasets_name
    cfg.dataloader.train.num_workers = config.dataloader_num_workers
    cfg.dataloader.test.num_workers = config.dataloader_num_workers
    cfg.dataloader.evaluator.output_dir = os.path.join(output_path, f"eval")
    cfg.dataloader.train.total_batch_size = config.batch_size
    cfg.train.seed = config.seed
    cfg.train.output_dir = output_path
    cfg.train.log_period = config.train_log_interval
    cfg.train.max_iter = config.max_epochs * dataset_length // config.batch_size
    cfg.train.eval_period = config.eval_epoch_interval * dataset_length // config.batch_size
    cfg.train.checkpointer.period = None   # only saving best checkpoint after eval

    cfg.lr_multiplier = CfgNode()
    cfg.lr_multiplier.warmup_steps = config.scheduler_warmup_steps
    cfg.lr_multiplier.warmup_factor = 0.001
    cfg.lr_multiplier.warmup_method = "linear"

    cfg.lr_multiplier.scheduler = CfgNode()
    cfg.lr_multiplier.scheduler.name = config.scheduler_type
    if config.scheduler_epochs_steps is not None:
        cfg.lr_multiplier.scheduler.steps = [step * dataset_length // config.batch_size for step in config.scheduler_epochs_steps]
    else:
        cfg.lr_multiplier.scheduler.steps = None
    cfg.lr_multiplier.scheduler.gamma = config.scheduler_gamma

    if config.lr:
        print(f"Changing base learning rate from {cfg.optimizer.lr} to {config.lr}.")
        cfg.optimizer.lr = config.lr
        cfg.optimizer.params.base_lr = config.lr

    # Enable checkpointing in the transformer encoder, lowering memory consumption.
    if hasattr(cfg.model, 'transformer') and hasattr(cfg.model.transformer, 'encoder') and hasattr(cfg.model.transformer, 'decoder'):
        cfg.model.transformer.encoder.use_checkpoint = config.use_gradient_checkpointing
        cfg.model.transformer.decoder.use_checkpoint = config.use_gradient_checkpointing

    if config.wandb_project is not None:
        cfg.train.wandb.enabled = True
        cfg.train.wandb.params.project = config.wandb_project
        cfg.train.wandb.params.dir = output_path
        cfg.train.wandb.params.name = model_name

    pprint(lazyconfig_to_dict(cfg))

    # Save config
    config.to_yaml(os.path.join(output_path, "config.yaml"))

    args = default_argument_parser().parse_args([])
    do_train(args, cfg, config, task)


def setup_mila_cluster_ddp():
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    # DDP Job is being run via `srun` on a slurm cluster.
    rank = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else 0
    world_size = int(os.environ["SLURM_NTASKS"]) if "SLURM_NTASKS" in os.environ else 1

    # SLURM var -> torch.distributed vars in case needed
    # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
    # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


def eval_detrex(config: DetectorConfig, fold_name: str):
    print("Setting up validation/test datasets...")
    d2_eval_datasets_name = register_detection_dataset(
        root_path=config.data_root_path,
        fold=fold_name,
        force_binary_class=True if config.num_classes == 1 else False
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(config.train_output_path, f"eval_{config.model}_{now}")
    os.makedirs(output_path, exist_ok=True)

    cfg = get_base_detrex_model_cfg(config)
    cfg.dataloader.test.dataset.names = d2_eval_datasets_name
    cfg.dataloader.test.num_workers = config.dataloader_num_workers
    cfg.dataloader.evaluator.output_dir = os.path.join(output_path, "eval")

    # Instantiate the model and move to device
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    # Build EMA if enabled
    ema.may_build_model_ema(cfg, model)

    # Load checkpoint
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)

    # If using EMA for eval, apply it here
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        ema.apply_model_ema(model)

    # Run evaluation
    results = do_test(cfg, model, eval_only=True)
    print("Evaluation results:")
    print(results)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    # Enable fast debugging by running several iterations to check for any bugs.
    if cfg.train.fast_dev_run.enabled:
        cfg.train.max_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        print(do_test(cfg, model, eval_only=True))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
