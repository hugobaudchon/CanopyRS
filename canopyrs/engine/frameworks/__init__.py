# One package per ML framework, holding ALL its CanopyRS glue: model-config builders,
# augmentations, and training scripts. IMPORTANT: keep this and per-framework __init__s free
# of train-script imports — inference wrappers import the cfg/augmentation leaf modules, and
# an eager train import would drag training deps (wandb, trainers, evaluators) into inference.
