"""Shared component scaffolding.

``Component`` is the base every pipeline component subclasses. It holds the identity the pipeline
assigns (``component_id`` + per-component ``out_dir``), and the two bits of boilerplate the model
components repeat: resolving a model from its registry and building a tile image loader. Components are
constructed with their ``config`` only — the pipeline owns output folders, so it sets ``component_id``
and ``out_dir`` before calling ``run`` (a component reads ``self.out_dir`` when it needs to write).

``flatten_by_tile`` is the other shared piece: detector/segmenter inference returns predictions grouped
per tile, and turning those into one row per object is the same loop everywhere.
"""

import os

from canopyrs.engine.models.registry import Registry
from canopyrs.engine.data import Imagery
from canopyrs.engine.loader import tile_loader

# kind -> component class, populated by the @register_component decorators on each component. Lets the
# pipeline instantiate components from config (Pipeline.from_config) without importing each class.
COMPONENT_REGISTRY = Registry("component")
register_component = COMPONENT_REGISTRY.register


def default_num_workers():
    """Image loader workers: one per available CPU less one for the main process, capped at 10.
    Counts the CPUs actually allocated to this process, which on a cluster is the job's share rather
    than the whole machine's."""
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, min(n_cpu - 1, 10))


class Component:
    name = None          # defaults to the class name lowercased (set in __init_subclass__)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            cls.name = cls.__name__.lower()

    def __init__(self, config):
        self.config = config
        self.component_id = None     # assigned by the pipeline
        self.out_dir = None          # assigned by the pipeline before run()
        self.num_workers = default_num_workers()   # the pipeline overrides it when its config sets one

    @property
    def label(self) -> str:
        """``{id}_{name}`` once the pipeline has assigned an id, else just the name."""
        return f"{self.component_id}_{self.name}" if self.component_id is not None else self.name

    def _model(self, registry):
        """The model class named by ``config.model``, or a clear error.

        Resolution happens at component construction (= pipeline build), not run time, so a
        missing/broken optional framework fails BEFORE tilerizing. ``registry.get`` explains
        which extra installs an unavailable model; a wrapper's optional ``preflight(config)``
        classmethod then verifies its framework actually works (e.g. detrex compiled with GPU
        support) — both raise MissingExtraError, which the pipeline aggregates.
        """
        model_class = registry.get(self.config.model)
        preflight = getattr(model_class, "preflight", None)
        if preflight is not None:
            preflight(self.config)
        return model_class

    def _loader(self, source, batch_size):
        """A tile image loader over ``source`` — an ``Imagery`` table (its reading frame) or an already
        built reading frame."""
        frame = source.reading_frame() if isinstance(source, Imagery) else source
        return tile_loader(frame, batch_size=batch_size, num_workers=self.num_workers)


def flatten_by_tile(tile_ids, **per_tile):
    """Flatten parallel per-tile prediction lists into row-aligned flat lists.

    ``tile_ids`` is one id per tile (group). Each value in ``per_tile`` is a list aligned to
    ``tile_ids`` — one inner list per tile, holding that tile's per-object values (geometry, scores,
    ...). Returns ``(flat_tile_ids, {col: flat_list})`` with every column expanded to one row per
    object, all lists the same length."""
    keys = list(per_tile)
    flat = {key: [] for key in keys}
    flat_tile_ids = []
    for i, tile_id in enumerate(tile_ids):
        groups = [per_tile[key][i] for key in keys]
        n = len(groups[0]) if groups else 0
        flat_tile_ids.extend([tile_id] * n)
        for key, group in zip(keys, groups):
            flat[key].extend(group)
    return flat_tile_ids, flat
