"""Shared component scaffolding.

``Component`` is the base every pipeline component subclasses. It holds the identity the pipeline
assigns (``component_id`` + per-component ``out_dir``), and the two bits of boilerplate the model
components repeat: resolving a model from its registry and building a tile image loader. Components are
constructed with their ``config`` only — the pipeline owns output folders, so it sets ``component_id``
and ``out_dir`` before calling ``run`` (a component reads ``self.out_dir`` when it needs to write).

``flatten_by_tile`` is the other shared piece: detector/segmenter inference returns predictions grouped
per tile, and turning those into one row per object is the same loop everywhere.
"""

from canopyrs.engine.models.registry import Registry
from canopyrs.engine.data import Tiles
from canopyrs.engine.loader import tile_loader

# kind -> component class, populated by the @register_component decorators on each component. Lets the
# pipeline instantiate components from config (Pipeline.from_config) without importing each class.
COMPONENT_REGISTRY = Registry("v3_component")
register_component = COMPONENT_REGISTRY.register


class Component:
    name = None          # defaults to the class name lowercased (set in __init_subclass__)
    NUM_WORKERS = 4

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            cls.name = cls.__name__.lower()

    def __init__(self, config):
        self.config = config
        self.component_id = None     # assigned by the pipeline
        self.out_dir = None          # assigned by the pipeline before run()

    @property
    def label(self) -> str:
        """``{id}_{name}`` once the pipeline has assigned an id, else just the name."""
        return f"{self.component_id}_{self.name}" if self.component_id is not None else self.name

    def _model(self, registry):
        """The model class named by ``config.model``, or a clear error."""
        if self.config.model not in registry:
            raise ValueError(f"Invalid {self.name} model: {self.config.model}")
        return registry[self.config.model]

    def _loader(self, source, batch_size):
        """A tile image loader over ``source`` — a ``Tiles`` table (its reading frame) or an already
        built reading frame."""
        frame = source.reading_frame() if isinstance(source, Tiles) else source
        return tile_loader(frame, batch_size=batch_size, num_workers=self.NUM_WORKERS)


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
