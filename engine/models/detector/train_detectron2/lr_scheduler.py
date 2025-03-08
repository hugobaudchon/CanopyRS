import logging

import torch
from detectron2.config import instantiate
from detectron2.solver import WarmupParamScheduler, LRMultiplier
from detrex.config.configs.common.common_schedule import exponential_lr_scheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler


def build_lr_scheduler(
        lr_scheduler_name: str,
        lr_steps: list,
        lr_gamma: float,
        max_iter: int,
        warmup_factor: float,
        warmup_iters: int,
        warmup_method: str,
        optimizer: torch.optim.Optimizer,
) -> LRMultiplier:
    """
    Build a LR scheduler from config.
    """
    name = lr_scheduler_name

    if name == "WarmupMultiStepLR":
        steps = [x for x in lr_steps if x <= max_iter]
        if len(steps) != len(lr_steps):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[lr_gamma ** k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=max_iter,
        )
    elif name == "WarmupExponentialLR":
        # already has a warmup so return it directly
        sched = instantiate(exponential_lr_scheduler(
            start_value=1.0,
            decay=lr_gamma,
            warmup_steps=warmup_iters,
            num_updates=max_iter,
            warmup_method=warmup_method,
            warmup_factor=warmup_factor,
        ))
        return LRMultiplier(optimizer, multiplier=sched, max_iter=max_iter)
    elif name == "WarmupCosineLR":
        sched = CosineParamScheduler(
            start_value=1.0,
            end_value=lr_gamma,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        warmup_factor,
        min(warmup_iters / max_iter, 1.0),
        warmup_method,
        False,  # rescale interval
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=max_iter)

