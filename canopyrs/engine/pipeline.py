"""
v3 pipeline: run components in order, threading data by type.

Each component declares ``requires`` (entries: a data-class type, a ``Need(type, …)``, or a ``one_of``)
and ``produces`` (a type or a ``Need`` describing its output). The pipeline keeps one list per data type
— ``sources`` / ``tiles`` / ``objects``, latest last — and, for each component, passes the *latest*
instance of each required type after checking it satisfies the ``Need`` (precondition). It then checks
the returned table matches what the component promised (postcondition), stores it, and — when an
``output_dir`` is set — saves it as parquet under ``{id}_{name}/`` and records it in a run manifest. A
component never names its predecessor, only the kinds (and shape) of data it needs.

Two validations off the *same* declarations: ``run`` enforces requires/produces at runtime against real
tables; ``validate`` is an optional pre-flight that threads ``produces`` forward as simulated ``Schema``s
and checks every ``requires`` before any compute. Both consume ``thread_schemas`` / ``Need.check``.

Beyond running: ``from_config`` builds components from ``(kind, config)`` steps; ``resume`` skips the
contiguous prefix of components already done (config unchanged + outputs on disk); ``from_dir`` reloads
a saved run; ``export`` writes a GPKG or COCO for the Objects at a chosen step.
"""

import shutil
from pathlib import Path

import geopandas as gpd

from canopyrs.engine.utils import green_print, parse_tilerizer_aoi_config
from canopyrs.engine.raster_validation import validate_raster_rgb_bands
from canopyrs.engine import store
from canopyrs.engine.constants import Col, MASK
from canopyrs.engine.contracts import Requirement, Schema, as_requirements
from canopyrs.engine.data import Sources, Tiles, Objects
from canopyrs.engine.components import COMPONENT_REGISTRY
from canopyrs.engine.visualizer import PipelineFlowVisualizer


class Pipeline:
    def __init__(self, components, sources=None, tiles=None, objects=None, output_dir=None):
        """Seeds — at least one is needed to ``run``:
          - ``sources``: the input imagery — a raster path, a list of paths, or ``{path, modality,
            timestamp}`` descriptors (built into a seed ``Sources`` table; a ``Sources`` passes through);
          - ``tiles``: a pre-cut tiles folder path (``Tiles.from_tiles_dir``) or a ``Tiles`` instance;
          - ``objects``: a gpkg path (``Objects.from_gpkg``) or an ``Objects`` instance (prior detections).
        Given the seeds and components, the wiring is validated here so a bad pipeline fails at
        construction, before any compute."""
        self.components = list(components)
        for i, component in enumerate(self.components):
            component.component_id = i
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = self._build_seeds(sources, tiles, objects)
        self.sources, self.tiles, self.objects = [], [], []
        self._lists = {Sources: self.sources, Tiles: self.tiles, Objects: self.objects}
        self.outputs = []   # per-component list of produced tables (aligned to self.components)
        self.manifest = None
        self._resume_requested = False   # set by from_config(resume_from=...); run() honors it by default
        if self.seeds and self.components:
            self.validate()

    @staticmethod
    def _build_seeds(sources, tiles, objects):
        """The seed tables from whatever inputs were given, in dependency order (Sources, Tiles,
        Objects). A path is built into its table; an instance passes through. A path-seeded Objects is
        linked to a path-seeded Tiles when both are given, so its ancestry walk works."""
        seeds = []
        if sources is not None:
            seeds.append(Sources.from_paths(sources))
        if tiles is not None:
            seeds.append(tiles if isinstance(tiles, Tiles) else Tiles.from_tiles_dir(tiles))
        if objects is not None:
            if isinstance(objects, Objects):
                seeds.append(objects)
            else:
                tiles_seed = next((s for s in seeds if isinstance(s, Tiles)), None)
                seeds.append(Objects.from_gpkg(objects, tiles=tiles_seed))
        return seeds

    @classmethod
    def from_config(cls, steps, sources=None, tiles=None, objects=None, output_dir=None, aoi=None,
                    resume_from=None, initialize_from=None):
        """Build a pipeline from ordered ``(kind, config)`` steps (instantiated from the registry) over
        the given seeds. ``aoi`` (a gpkg path) restricts tilerizing to an area of interest. Pass one of:
        ``resume_from`` (continue a prior run — its unchanged prefix is skipped; same folder, or a new
        ``output_dir`` for cross-folder) or ``initialize_from`` (seed this new-config run from a prior
        run's latest Tiles + Objects; components restart at id 0)."""
        if resume_from and initialize_from:
            raise ValueError("Pass only one of resume_from / initialize_from, not both.")
        aois_config = cls._build_aois_config(aoi)
        components = [cls._make_component(kind, config, aois_config) for kind, config in steps]

        if initialize_from:
            prior = cls.from_dir(initialize_from)
            tiles = tiles if tiles is not None else prior.latest(Tiles)
            objects = objects if objects is not None else prior.latest(Objects)
        elif resume_from and output_dir and Path(output_dir).resolve() != Path(resume_from).resolve():
            shutil.copytree(resume_from, output_dir, dirs_exist_ok=True)   # cross-folder: bring the prior run over

        pipe = cls(components, sources=sources, tiles=tiles, objects=objects, output_dir=output_dir)
        pipe._resume_requested = bool(resume_from)
        return pipe

    @staticmethod
    def _make_component(kind, config, aois_config):
        """Instantiate a component from the registry; the tilerizer also receives the run-level AOI."""
        if kind == "tilerizer":
            return COMPONENT_REGISTRY[kind](config, aois_config=aois_config)
        return COMPONENT_REGISTRY[kind](config)

    @staticmethod
    def _build_aois_config(aoi):
        """A geodataset AOIConfig restricting tilerizing to ``aoi`` (a gpkg path), or None (whole raster)."""
        if aoi is None:
            return None
        return parse_tilerizer_aoi_config(aoi_config="package", aoi_type=None, aois={"infer": str(aoi)})

    # --- run -----------------------------------------------------------------
    def run(self, resume=None, verbose=True, strict_rgb_validation=True):
        """Run the components in order over the seeds (given at construction), and return ``self``
        (inspect ``self.tiles`` / ``self.objects`` / ``self.sources`` after). With ``output_dir`` set,
        each output is saved and a manifest written. ``resume`` (defaulting to what ``from_config``
        recorded) skips an already-completed prefix (unchanged config + outputs on disk).
        ``strict_rgb_validation``: raise (True) or warn (False) when a source raster's bands aren't
        tagged R,G,B."""
        resume = self._resume_requested if resume is None else resume
        if not self.seeds and not resume:
            raise ValueError("Pipeline has no seeds; construct with sources= / tiles= / objects=")
        self._validate_sources(strict_rgb_validation)
        if verbose:
            self.print_flow_chart()
        for seed in self.seeds:
            self._store(seed)
        self.outputs = []
        start = self._resume_prefix() if resume else 0

        for i in range(start, len(self.components)):
            component = self.components[i]
            green_print(f"Running {component.label}...")
            component.out_dir = self._component_dir(component)
            args = [self._resolve(req, component) for req in component.requires]
            produced = self._collect(component.run(*args))
            self._check_produced(component, produced)
            for table in produced:
                self._store(table)
            self.outputs.append(produced)
            if self.output_dir is not None:
                self._save(component, produced)

        if self.output_dir is not None:
            store.save_seeds(self.output_dir, self.seeds)
            self.manifest = store.write_manifest(self.output_dir, self.components)
            self._write_final_gpkg()
        green_print("Pipeline finished")
        return self

    def _validate_sources(self, strict_rgb_validation):
        """Pre-flight on the seed rasters before any compute: first 3 bands are RGB-tagged (per
        ``strict_rgb_validation``), uint8, and in [0, 255]. Only rgb Sources are checked — other
        modalities aren't RGB rasters, and a tiles/objects-seeded run has no Sources to check."""
        sources = next((s for s in self.seeds if isinstance(s, Sources)), None)
        if sources is None:
            return
        df = sources.df
        if Col.MODALITY in df.columns:
            df = df[df[Col.MODALITY] == "rgb"]
        for path in df[Col.SOURCE_PATH]:
            validate_raster_rgb_bands(Path(path), strict_color_interp=strict_rgb_validation)

    def _write_final_gpkg(self):
        """Auto-write the final result GPKG at the run root (``final.gpkg``) — the latest Objects, widened
        with their ancestry columns. Best effort: skipped with a note if there are no Objects or they're
        in tile-pixel coords (no CRS to write)."""
        try:
            anchor = self._objects_at(self._last_objects_index())
        except ValueError:
            return
        if not anchor.crs_set:
            print("Final GPKG skipped: final Objects are in tile-pixel coords (no CRS).")
            return
        print(f"Final GPKG: {self.export('gpkg', path=self.output_dir / 'final.gpkg')}")

    def validate(self):
        """Pre-flight wiring check, no compute: thread each component's ``produces`` forward as Schemas
        and confirm every ``requires`` is satisfiable. Raises on the first unmet requirement. Called at
        construction. Optimistic about ancestry (a produced column stays reachable while the chain links
        ``prev_objects``); ``run``'s checks are the ground truth."""
        for component, before, _ in self.thread_schemas():
            if component is None:
                continue
            for req in component.requires:
                desc, err = Requirement.coerce(req).resolve(before.get)
                if desc is None:
                    raise ValueError(f"{type(component).__name__} {err}")
        return self

    def thread_schemas(self):
        """Yield ``(component, before, after)`` per step — ``before``/``after`` are ``{data_type:
        Schema}`` snapshots of what's available immediately before/after the component. The first yield
        is ``(None, None, seed)`` (the seed ``Sources``). Pure simulation, no compute: the single source
        of "what's available when", consumed by ``validate`` and the flow chart."""
        available = {type(seed): seed.schema() for seed in self.seeds}
        yield None, None, dict(available)
        for component in self.components:
            before = dict(available)
            for need in as_requirements(component.produces):
                prev = available.get(need.data_type)
                columns = set(need.columns)
                links = set(need.links)
                fks = getattr(need.data_type, "fks", {})
                columns |= {fks[link] for link in need.links if link in fks}   # a link's FK column is a column too
                if prev is not None and "prev_objects" in need.links:          # ancestry: reach back through the chain
                    columns |= prev.columns
                    links |= prev.links
                available[need.data_type] = Schema(columns=columns, links=links, crs=need.crs)
            yield component, before, dict(available)

    def print_flow_chart(self):
        PipelineFlowVisualizer(self).print()

    def latest(self, data_type):
        """The most recently produced (or seeded) instance of ``data_type``, or None."""
        produced = self._lists[data_type]
        return produced[-1] if produced else None

    # --- export --------------------------------------------------------------
    def export(self, fmt, end_at=None, start_at=0, path=None, scores_column=None, categories_column=None):
        """Write a single GPKG or COCO file for the Objects produced at component ``end_at`` (default:
        the last that produced Objects) and return its path. Columns are merged from the ancestry but
        only those introduced by components in ``[start_at, end_at]``; with repeats (e.g. two
        aggregators) the latest value wins (``Objects.column``). ``fmt`` is ``"gpkg"`` or ``"coco"``."""
        manifest = self.manifest or (store.read_manifest(self.output_dir) if self.output_dir else None)
        if manifest is None:
            raise ValueError("no run manifest available; run() the pipeline with an output_dir before exporting")
        end_at = self._last_objects_index() if end_at is None else end_at
        anchor = self._objects_at(end_at)
        if anchor is None:
            raise ValueError(f"component {end_at} did not produce Objects to export")
        columns = self._window_columns(manifest, start_at, end_at)
        directory = self.output_dir / f"{manifest[end_at]['id']}_{manifest[end_at]['name']}"
        if fmt == "gpkg":
            return store.write_gpkg(self._wide_gdf(anchor, columns), Path(path) if path else directory / "export.gpkg")
        if fmt == "coco":
            return self._export_coco(anchor, columns, Path(path) if path else directory / "export.coco.json",
                                     scores_column, categories_column)
        raise ValueError(f"unknown export format '{fmt}' (use 'gpkg' or 'coco')")

    def _wide_gdf(self, anchor: Objects, columns):
        if not anchor.crs_set:
            raise ValueError("GPKG export needs georeferenced Objects (crs set); these are in tile-pixel coords")
        data = {Col.GEOMETRY: anchor.df.geometry.values}
        for col in sorted(columns):
            values = self._reach(anchor, col)
            if values is not None:
                data[col] = values
        tile_path = self._tile_path_per_object(anchor)   # latest tile_path, if reachable
        if tile_path is not None:
            data[Col.TILE_PATH] = tile_path.values
        return gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=anchor.df.crs)

    def _export_coco(self, anchor: Objects, columns, path, scores_column, categories_column):
        tiles = anchor.linked("tiles")
        if tiles is None:
            raise ValueError("COCO export needs tiles linked to the objects (none reachable in the ancestry)")
        try:
            tile_id = anchor.column(Col.TILE_ID)
        except KeyError:
            raise ValueError("COCO export needs each object's tile_id (none reachable in the ancestry)")
        if Col.TILE_PATH not in tiles.df.columns:
            raise ValueError("COCO export needs tile images on disk (re-run the tilerizer with save_tiles_to_disk=True)")
        paths = tile_id.map(tiles.df.set_index(Col.TILE_ID)[Col.TILE_PATH])
        if paths.isna().any() or (paths.astype(str) == "").any():
            raise ValueError("COCO export needs tile images on disk (re-run the tilerizer with save_tiles_to_disk=True)")

        scores_column = scores_column or self._latest_in(columns, store.SCORE_COLS)
        categories_column = categories_column or self._latest_in(columns, store.CLASS_COLS)

        data = {Col.TILE_PATH: paths.values, Col.GEOMETRY: anchor.df.geometry.values}
        others = []
        for col in sorted(columns):
            if col in (Col.TILE_PATH, Col.GEOMETRY):
                continue
            values = self._reach(anchor, col)
            if values is not None:
                data[col] = values
                if col not in (scores_column, categories_column):
                    others.append(col)
        for col in (scores_column, categories_column):       # ensure score/category cols are present
            if col and col not in data:
                values = self._reach(anchor, col)
                if values is not None:
                    data[col] = values
        # CRS geometry -> geodataset converts to each tile's pixels; pixel geometry (crs=None) is used as-is.
        gdf = gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=anchor.df.crs)
        use_rle = bool(len(anchor)) and anchor.df[Col.GEOM_KIND].iloc[0] == MASK
        return store.write_coco(gdf, path, scores_column=scores_column, categories_column=categories_column,
                                other_attributes_columns=others, use_rle=use_rle, categories=None)

    def _last_objects_index(self) -> int:
        for i in range(len(self.outputs) - 1, -1, -1):
            if any(isinstance(table, Objects) for table in self.outputs[i]):
                return i
        raise ValueError("pipeline produced no Objects to export")

    def _objects_at(self, index):
        if not 0 <= index < len(self.outputs):
            return None
        return next((table for table in self.outputs[index] if isinstance(table, Objects)), None)

    @staticmethod
    def _window_columns(manifest, start_at, end_at):
        columns = set()
        for entry in manifest[start_at:end_at + 1]:
            for produced in entry["produces"]:
                if produced["type"] == Objects.__name__:
                    columns |= set(produced["columns"])
        return columns

    @staticmethod
    def _reach(anchor: Objects, col):
        try:
            return anchor.column(col).values
        except KeyError:
            return None

    def _tile_path_per_object(self, anchor: Objects):
        tiles = anchor.linked("tiles")
        if tiles is None or Col.TILE_PATH not in tiles.df.columns:
            return None
        try:
            tile_id = anchor.column(Col.TILE_ID)
        except KeyError:
            return None
        return tile_id.map(tiles.df.set_index(Col.TILE_ID)[Col.TILE_PATH])

    @staticmethod
    def _latest_in(columns, ordered):
        present = [col for col in ordered if col in columns]
        return present[-1] if present else None

    # --- reload + resume -----------------------------------------------------
    @classmethod
    def from_dir(cls, root):
        """Reload a saved run for inspection / re-export: reconstruct the typed tables in order,
        re-linking FKs from their persisted columns. Data-only (no components) — everything ``export``
        needs comes from the manifest and the reloaded tables."""
        root = Path(root)
        manifest = store.read_manifest(root)
        if manifest is None:
            raise FileNotFoundError(f"no {store.MANIFEST} in {root}")
        pipe = cls([], output_dir=root)
        pipe.manifest = manifest
        pipe._load_seeds(root)                              # seed tables first, so produced FKs relink
        pipe._load_prefix(manifest, len(manifest), root)
        return pipe

    def _load_seeds(self, root):
        """Reload persisted seed tables (``_seed/``) before any component output, so produced Objects
        can relink their FKs (e.g. ``tiles``) to a seeded table — the case of a run seeded from a
        pre-cut tiles folder rather than a tilerizer component."""
        seeds = store.read_seeds(root)
        if not seeds:
            return
        directory = Path(root) / store.SEED_DIR
        for spec in seeds:
            data_type = store.TYPE_BY_NAME[spec["type"]]
            df = store.load_df(data_type, directory / spec["file"])
            self._store(self._rebuild(data_type, df))

    def _resume_prefix(self) -> int:
        """The number of leading components to skip: the longest contiguous prefix whose recorded
        config_hash matches the current component and whose output files exist. Loads those outputs."""
        manifest = store.read_manifest(self.output_dir) if self.output_dir else None
        if not manifest:
            return 0
        done = self._done_prefix(manifest)
        self._load_prefix(manifest, done, self.output_dir)
        for entry in manifest[:done]:
            green_print(f"Skipping {entry['id']}_{entry['name']} (already done, resumed)")
        return done

    def _done_prefix(self, manifest) -> int:
        done = 0
        for i, component in enumerate(self.components):
            if i >= len(manifest):
                break
            entry = manifest[i]
            if entry.get("name") != component.name or entry.get("config_hash") != store.config_hash(component.config):
                break
            directory = self._component_dir(component)
            if not all((directory / produced["file"]).exists() for produced in entry["produces"]):
                break
            done = i + 1
        return done

    def _load_prefix(self, manifest, count, root):
        for entry in manifest[:count]:
            directory = root / f"{entry['id']}_{entry['name']}"
            produced = []
            for spec in entry["produces"]:
                data_type = store.TYPE_BY_NAME[spec["type"]]
                df = store.load_df(data_type, directory / spec["file"])
                table = self._rebuild(data_type, df)
                self._store(table)
                produced.append(table)
            self.outputs.append(produced)

    def _rebuild(self, data_type, df):
        """A typed table from its saved dataframe, re-linking FKs to the latest loaded parent of each
        linked type (the same latest-wins rule ``run`` threads inputs by)."""
        if data_type is Sources:
            return Sources(df)
        if data_type is Tiles:
            sources = self.latest(Sources)
            related = {"sources": sources} if (sources is not None and self._has_fk(df, Col.SOURCE_ID)) else {}
            return Tiles(df, **related)
        related = {}
        tiles = self.latest(Tiles)
        if tiles is not None and self._has_fk(df, Col.TILE_ID):
            related["tiles"] = tiles
        prev = self.latest(Objects)
        if prev is not None and self._has_fk(df, Col.PREV_OBJECT_ID):
            related["prev_objects"] = prev
        return Objects(df, **related)

    @staticmethod
    def _has_fk(df, col) -> bool:
        return col in df.columns and df[col].notna().any()

    # --- internals -----------------------------------------------------------
    def _component_dir(self, component):
        return self.output_dir / f"{component.component_id}_{component.name}" if self.output_dir else None

    def _save(self, component, produced):
        directory = self._component_dir(component)
        for table in produced:
            store.save_table(table, directory)

    @staticmethod
    def _collect(outputs):
        return list(outputs if isinstance(outputs, tuple) else (outputs,))

    def _store(self, data):
        self._lists[type(data)].append(data)

    def _resolve(self, req, component):
        """The latest instance satisfying a ``requires`` entry (a bare type, a ``Need``, or an ``AnyOf``).
        Resolution is by type via ``self.latest``; an ``AnyOf`` picks the first available alternative."""
        inst, err = Requirement.coerce(req).resolve(self.latest)
        if inst is None:
            raise ValueError(f"{type(component).__name__} {err}")
        return inst

    def _check_produced(self, component, produced):
        """Postcondition: the component actually produced what its ``produces`` promised. Catches a
        declaration drifting from the ``run`` body at the producer, not three components downstream."""
        name = type(component).__name__
        for need in as_requirements(component.produces):
            inst = next((table for table in produced if type(table) is need.data_type), None)
            if inst is None:
                raise ValueError(f"{name} promised to produce {need.data_type.__name__} but did not")
            err = need.check(inst)
            if err:
                raise ValueError(f"{name} produced {need.data_type.__name__} but {err}")
