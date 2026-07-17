"""Train RSPrompter through the standard CanopyRS `train.py` + SegmenterConfig flow.

This is the mmdet analogue of ``train_detectron2``/``train_detrex``: it builds the mmengine
config **programmatically** (no .py/.yaml mmdet config files) and runs ``Runner.train()``.
The dicts below are a faithful port of the configs that produced the published v3/v4 models
(``rsprompter_anchor-selvamask-NEW-666-2666_{3,4}_seed*`` in the mmdet_rsprompter fork),
including the train/test augmentation pipelines, which were painful to get right:
SAM only accepts 1024x1024 inputs, so the pipeline is RandomFlip (h+v) ->
RandomSquareCrop(666..2666) -> DeterministicResizeWithinRange(1024) -> BatchFixedSizePad(1024)
in the data preprocessor. The custom transforms live in
``mmdet.rsprompter.selvamask_augmentations`` (our fork).

Requires the mmdet stack: `canopyrs setup mmdet`.

Data layout (same as the tilerized SelvaMask training data):
    <data_root>/<site>/<coco json per fold>          (geodataset CocoNameConvention names)
    <data_root>/<site>/tiles/<fold>/*.tif
Sites are auto-discovered by parsing every ``<site>/*.json`` with geodataset's
CocoNameConvention; ``config.train_dataset_names`` (if set) filters to those site names.
"""

import os
from pathlib import Path

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.extras import require_extra

# (hf model id, feature-aggregator select_layers stop) per SAM backbone size — base: ViT-B 12
# blocks -> layers 1..13, large: 24 -> 1..25, huge: 32 -> 1..33 (step 2), per upstream comments.
SAM_ARCHS = {
    'base': ('facebook/sam-vit-base', 13),
    'large': ('facebook/sam-vit-large', 25),
    'huge': ('facebook/sam-vit-huge', 33),
}
_ARCH_SYNONYMS = {'b': 'base', 'l': 'large', 'h': 'huge'}

CLASSES = ('tree',)
PROMPT_SHAPE = (70, 5)   # (pointsets per image, points per pointset) — as trained


def _sam_arch(config: SegmenterConfig) -> str:
    arch = _ARCH_SYNONYMS.get(config.architecture, config.architecture)
    if arch not in SAM_ARCHS:
        raise ValueError(f"RSPrompter training expects architecture in {list(SAM_ARCHS)} "
                         f"(or {list(_ARCH_SYNONYMS)}), got '{config.architecture}'")
    return arch


def _ensure_sam_weights(arch: str) -> tuple:
    """Return (hf_pretrain_name_or_path, pytorch_model.bin path), downloading if needed.

    The fork's modules load SAM weights via mmengine ``init_cfg`` from a plain state-dict
    ``pytorch_model.bin``. transformers>=5 only writes safetensors, so we materialize the .bin
    with torch.save ourselves. Cache dir override: $CANOPYRS_SAM_CACHE.
    """
    import torch

    hf_name, _ = SAM_ARCHS[arch]
    cache_root = Path(os.environ.get('CANOPYRS_SAM_CACHE',
                                     Path.home() / '.cache' / 'canopyrs' / 'sam_cache'))
    cache_dir = cache_root / f"sam_vit_{arch}"
    ckpt = cache_dir / 'pytorch_model.bin'
    if not ckpt.exists() or not (cache_dir / 'config.json').exists():
        print(f"[train_rsprompter] downloading SAM weights ({hf_name}) to {cache_dir} ...")
        from transformers import SamModel
        model = SamModel.from_pretrained(hf_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cache_dir)
        torch.save(model.state_dict(), ckpt)
        del model
    return str(cache_dir), str(ckpt)


def _discover_site_datasets(data_root: str, site_filter=None) -> dict:
    """{site: {fold: ann_file}} for every site dir whose COCO jsons parse with geodataset's
    CocoNameConvention — robust to scale/ground-resolution naming variants."""
    from geodataset.utils.file_name_conventions import CocoNameConvention

    sites = {}
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root_path '{data_root}' is not a directory")
    for site_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if site_filter and site_dir.name not in site_filter:
            continue
        folds = {}
        for j in site_dir.glob('*.json'):
            try:
                product, _, _, fold = CocoNameConvention.parse_name(j.name)
            except Exception:
                continue
            if product == site_dir.name:
                folds[fold] = str(j)
        if 'train' in folds:
            sites[site_dir.name] = folds
    if not sites:
        raise FileNotFoundError(
            f"No site datasets found under '{data_root}' (expected <site>/<CocoNameConvention "
            f"json> + <site>/tiles/<fold>/). Site filter: {site_filter}")
    return sites


def _site_dataset(data_root: str, site: str, ann_file: str, fold: str, pipeline: list) -> dict:
    return dict(
        type='CocoDataset',
        data_root=str(Path(data_root) / site),
        ann_file=ann_file,
        data_prefix=dict(img=f'tiles/{fold}'),
        metainfo=dict(classes=CLASSES),
        pipeline=pipeline,
        test_mode=(fold != 'train'),
    )


def _train_pipeline() -> list:
    """The trained SelvaMask augmentation pipeline (see module docstring)."""
    return [
        dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(type='RandomSquareCrop', min_side=666, max_side=2666, allow_negative_crop=True),
        dict(type='DeterministicResizeWithinRange', min_size=1024, max_size=1024),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
        dict(type='PackDetInputs'),
    ]


def _test_pipeline() -> list:
    return [
        dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='DeterministicResizeWithinRange', min_size=1024, max_size=1024),
        dict(type='PackDetInputs'),
    ]


def _model_cfg(config: SegmenterConfig, sam_name: str, sam_ckpt: str, select_stop: int,
               data_preprocessor: dict) -> dict:
    """RSPrompterAnchor as trained (base `_3`/`_4` family + leaf overrides merged)."""
    num_classes = config.num_classes
    pretrained = dict(type='Pretrained', checkpoint=sam_ckpt)
    return dict(
        type='RSPrompterAnchor',
        data_preprocessor=data_preprocessor,
        decoder_freeze=False,
        shared_image_embedding=dict(
            type='RSSamPositionalEmbedding', hf_pretrain_name=sam_name, init_cfg=pretrained),
        backbone=dict(
            type='RSSamVisionEncoder', hf_pretrain_name=sam_name,
            extra_config=dict(output_hidden_states=True), init_cfg=pretrained),
        neck=dict(
            type='RSFPN',
            feature_aggregator=dict(
                type='RSFeatureAggregator',
                in_channels=sam_name,               # fork derives base/large/huge from the name
                out_channels=256,
                hidden_channels=32,
                select_layers=range(1, select_stop, 2)),
            feature_spliter=dict(
                type='RSSimpleFPN', backbone_channel=256, in_channels=[64, 128, 256, 256],
                out_channels=256, num_outs=5, norm_cfg=dict(type='LN2d', requires_grad=True))),
        rpn_head=dict(
            type='RPNHead', in_channels=256, feat_channels=256,
            anchor_generator=dict(type='AnchorGenerator', scales=[4, 8],
                                  ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[.0, .0, .0, .0],
                            target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='RSPrompterAnchorRoIPromptHead',
            with_extra_pe=True,
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256, featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.],
                                target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256, featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='RSPrompterAnchorMaskHead',
                mask_decoder=dict(type='RSSamMaskDecoder', hf_pretrain_name=sam_name,
                                  init_cfg=pretrained),
                in_channels=256, roi_feat_size=14,
                per_pointset_point=PROMPT_SHAPE[1],
                with_sincos=True, multimask_output=False, class_agnostic=True,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                point_emb_norm_type=None)),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                              min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                # sampler num reduced from 256 as trained (pos-embedding collapse workaround)
                sampler=dict(type='RandomSampler', num=64, pos_fraction=0.5,
                             neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=-1, pos_weight=-1, debug=False),
            rpn_proposal=dict(nms_pre=2000, max_per_img=1000,
                              nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                              min_pos_iou=0.5, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=256, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=(1024, 1024),   # SAM decodes full-tile masks, not 28x28 crops
                pos_weight=-1, debug=False)),
        test_cfg=dict(
            rpn=dict(nms_pre=1000, max_per_img=1000,
                     nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5),
                      max_per_img=200, mask_thr_binary=0.5)),
    )


def get_rsprompter_train_cfg(config: SegmenterConfig):
    """Build the full mmengine Config from a CanopyRS SegmenterConfig."""
    from mmengine.config import Config

    arch = _sam_arch(config)
    sam_name, sam_ckpt = _ensure_sam_weights(arch)
    _, select_stop = SAM_ARCHS[arch]

    sites = _discover_site_datasets(config.data_root_path,
                                    site_filter=config.train_dataset_names or None)
    print(f"[train_rsprompter] sites: {list(sites)}")

    train_pipe, test_pipe = _train_pipeline(), _test_pipeline()
    train_sets = [_site_dataset(config.data_root_path, s, f['train'], 'train', train_pipe)
                  for s, f in sites.items()]
    val_sets = [_site_dataset(config.data_root_path, s, f['valid'], 'valid', test_pipe)
                for s, f in sites.items() if 'valid' in f]

    # Pad to SAM's fixed 1024x1024 AFTER the resize in the pipeline (as trained).
    data_preprocessor = dict(
        type='DetDataPreprocessor',
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        bgr_to_rgb=True, pad_mask=True, pad_size_divisor=32,
        batch_augments=[dict(type='BatchFixedSizePad', size=(1024, 1024), img_pad_value=0,
                             pad_mask=True, mask_pad_value=0, pad_seg=False)])

    max_iters = config.max_iters or 10000
    val_interval = max(50, max_iters // 50)
    # As trained: 50 linear-warmup iters. SegmenterConfig's scheduler_warmup_steps default
    # (1000) is a SAM-trainer default — only honor it when explicitly set, and keep it < max_iters.
    warmup = (config.scheduler_warmup_steps
              if 'scheduler_warmup_steps' in config.model_fields_set else 50)
    warmup = max(1, min(warmup, max_iters - 1))
    loader_common = dict(batch_size=config.batch_size,
                         num_workers=config.dataloader_num_workers,
                         persistent_workers=config.dataloader_num_workers > 0)

    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=config.lr, weight_decay=0.05))
    if config.use_amp:
        optim_wrapper.update(type='AmpOptimWrapper', dtype='float16')
    else:
        optim_wrapper.update(type='OptimWrapper')

    vis_backends = [dict(type='LocalVisBackend')]
    if config.wandb_project:
        run_name = f"rsprompter_anchor_{arch}_seed{config.seed}"
        vis_backends.append(dict(type='WandbVisBackend',
                                 init_kwargs=dict(project=config.wandb_project, name=run_name)))

    cfg = dict(
        default_scope='mmdet',
        work_dir=config.train_output_path,
        custom_imports=dict(imports=['mmdet.rsprompter', 'mmdet.rsprompter.selvamask_augmentations'],
                            allow_failed_imports=False),
        model=_model_cfg(config, sam_name, sam_ckpt, select_stop, data_preprocessor),
        train_dataloader=dict(
            **loader_common,
            sampler=dict(type='InfiniteSampler', shuffle=True),
            dataset=dict(type='ConcatDataset', datasets=train_sets)),
        val_dataloader=dict(
            **loader_common, drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(type='ConcatDataset', datasets=val_sets)),
        val_evaluator=dict(type='CocoMetric', metric=['bbox', 'segm'], format_only=False,
                           backend_args=None),
        train_cfg=dict(type='IterBasedTrainLoop', max_iters=max_iters,
                       val_interval=val_interval),
        val_cfg=dict(type='ValLoop'),
        optim_wrapper=optim_wrapper,
        param_scheduler=[
            dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=warmup),
            dict(type='CosineAnnealingLR', eta_min=config.lr * 0.001,
                 begin=warmup, end=max_iters, T_max=max_iters - warmup, by_epoch=False)],
        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=config.train_log_interval),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, by_epoch=False,
                            save_best='coco/segm_mAP', rule='greater', save_last=False),
            sampler_seed=dict(type='DistSamplerSeedHook')),
        env_cfg=dict(cudnn_benchmark=False,
                     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                     dist_cfg=dict(backend='nccl')),
        visualizer=dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'),
        log_processor=dict(type='LogProcessor', window_size=50, by_epoch=False),
        log_level='INFO',
        randomness=dict(seed=config.seed, deterministic=False),
        find_unused_parameters=True,
        load_from=config.checkpoint_path or None,
        resume=False,
        launcher='pytorch' if os.environ.get('RANK') is not None else 'none',
    )
    return Config(cfg)


def train_rsprompter(config: SegmenterConfig):
    """Entry point called by train.py for model in ('rsprompter_anchor', 'rsprompter_query')."""
    require_extra('mmdet', packages=('mmengine', 'mmdet', 'mmcv'))
    if config.model != 'rsprompter_anchor':
        raise NotImplementedError(
            f"Only 'rsprompter_anchor' training is ported so far (got '{config.model}'). "
            "The query variant needs its own model dict — see the fork's rsprompter_query configs.")

    # Register the fork's models + custom transforms before the Runner touches the registry.
    import mmdet.rsprompter  # noqa: F401
    import mmdet.rsprompter.selvamask_augmentations  # noqa: F401
    from mmengine.runner import Runner

    cfg = get_rsprompter_train_cfg(config)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()
