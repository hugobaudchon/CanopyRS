"""
Pipeline: run components in order, threading typed data by need.

See ``canopyrs/engine/README.md`` for the model in five sentences.

The pipeline keeps one list per data type (``sources`` / ``tiles`` / ``crops`` / ``objects``, latest
last) and, per component, binds each ``requires`` entry to the **newest instance of its type**, then
checks the Need against it — a mismatch fails loudly instead of falling back to an older table. Roles
are types, so a ``Need(Sources)`` finds the raster no matter how many tiles exist. The returned
tables are checked against ``produces`` (postcondition), stored, and — with an ``output_dir`` — saved
as parquet under ``{id}_{name}/`` and recorded in the run record (``run.json``). A component never
names its predecessor, only the type and shape of data it needs.

``validate`` (called at construction) runs the same checks statically: ``thread_schemas`` threads each
component's ``produces`` forward as declared ``Schema``s, newest per type — matching only ever binds
the newest instance of a type, so that's all the simulation keeps.

Beyond running: ``from_config`` builds components from ``(kind, config)`` steps; ``resume`` skips the
contiguous prefix already done (config unchanged + outputs on disk); ``from_dir`` reloads a saved run;
``export`` writes a GPKG or COCO for the Objects at a chosen step.
"""

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

from canopyrs.engine.utils import green_print, parse_tilerizer_aoi_config
from canopyrs.engine.raster_validation import validate_raster_rgb_bands
from canopyrs.engine import store
from canopyrs.engine.constants import Col, GeomKind, Modality
from canopyrs.engine.contracts import Schema, as_requirements
from canopyrs.engine.data import Crops, Imagery, Objects, Sources, Tiles, has_usable_values
from canopyrs.engine.components import COMPONENT_REGISTRY
from canopyrs.engine.models.extras import MissingExtraError
from canopyrs.engine.visualizer import PipelineFlowVisualizer


class Pipeline:
    def __init__(self, components, sources=None, tiles=None, objects=None, output_dir=None):
        """Seeds — at least one is needed to ``run``:
          - ``sources``: the input scenes — a raster path, a list of paths, or ``{path, modality,
            timestamp}`` descriptors (built into a ``Sources`` seed; an imagery instance passes through);
          - ``tiles``: a folder of pre-cut images, or an imagery instance. A folder is typed by what
            the components ask for (``Tiles`` for a detector, ``Crops`` for a classifier-only run) —
            see ``_seed_image_type``;
          - ``objects``: a gpkg path (``Objects.from_gpkg``) or an ``Objects`` instance (prior detections).
        Given the seeds and components, the wiring is validated here so a bad pipeline fails at
        construction, before any compute."""
        self.components = list(components)
        for i, component in enumerate(self.components):
            component.component_id = i
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sources, self.tiles, self.crops, self.objects = [], [], [], []
        self._lists = {Sources: self.sources, Tiles: self.tiles, Crops: self.crops, Objects: self.objects}
        self._imagery_log = []   # every stored imagery table, any type, chronological (reload relinking)
        self.seeds = self._build_seeds(sources, tiles, objects)
        self.outputs = []   # per-component list of produced tables (aligned to self.components)
        self.run_record = None
        self._resume_requested = False   # set by from_config(resume_from=...); run() honors it by default
        if self.seeds and self.components:
            self.validate()

    def _build_seeds(self, sources, tiles, objects):
        """The seed tables from whatever inputs were given, in dependency order (source scenes,
        images, objects). A path is built into its table; an instance passes through. A path-seeded
        Objects is linked to the imagery seed its rows were found in (the images seed preferred), so
        its ancestry walk works. Ends with the canonical derivations (``_derive_seeds``)."""
        seeds = []
        if sources is not None:
            seeds.append(Sources.from_paths(sources))
        if tiles is not None:
            if isinstance(tiles, Imagery):
                seeds.append(tiles)
            else:
                image_type = self._seed_image_type()
                if image_type is Crops:
                    print("Seeded images are used as Crops (the pipeline needs crops and no component makes them).")
                seeds.append(image_type.from_image_dir(tiles))
        if objects is not None:
            if isinstance(objects, Objects):
                seeds.append(objects)
            else:
                imagery_seed = next(
                    (s for s in reversed(seeds) if isinstance(s, Imagery)), None)   # images seed preferred
                seeds.append(Objects.from_gpkg(objects, imagery=imagery_seed))
        return seeds + self._derive_seeds(seeds)

    def _requires_before_produced(self):
        """Yield ``(need, produced)`` for every ``requires`` entry, in component order — ``produced``
        is the set of data types produced by earlier components. The shared scan behind the seed
        rules: "what does the pipeline ask for that nothing earlier makes?"."""
        produced = set()
        for component in self.components:
            for need in as_requirements(component.requires):
                yield need, produced
            produced |= {need.data_type for need in as_requirements(component.produces)}

    def _seed_image_type(self):
        """The role of a seeded image folder: scan the components in order and take the first
        requirement referencing an imagery role — by its data type, or by an
        Objects need's ``on=`` — that no earlier component produces. Default ``Tiles``: a folder
        nothing asks for by role is only ever consumed through object links, where the label doesn't
        change behavior."""
        for need, produced in self._requires_before_produced():
            role = need.data_type if need.data_type in (Tiles, Crops) else need.on
            if role in (Tiles, Crops) and role not in produced:
                return role
        return Tiles

    def _derive_seeds(self, seeds):
        """Canonical derivations for needs no seed or component covers — data statements, not
        guesses. Today one rule: an unmet ``Need(Objects, on=Crops)`` over a Crops seed derives one
        Object per crop (its footprint) — '1 image = 1 class' expressed in the data model, so a
        classifier-only run over a crops folder just works. Extend by adding rules here, never
        by special-casing components."""
        crops_seed = next((s for s in seeds if isinstance(s, Crops)), None)
        if crops_seed is None or any(isinstance(s, Objects) for s in seeds):
            return []
        for need, produced in self._requires_before_produced():
            if need.data_type is Objects and Objects not in produced and need.on is Crops:
                print("Derived one Object per seeded crop (a component needs Objects on Crops).")
                return [Objects.from_imagery(crops_seed)]
        return []

    @classmethod
    def from_config(cls, steps, sources=None, tiles=None, objects=None, output_dir=None, aoi=None,
                    resume_from=None, initialize_from=None):
        """Build a pipeline from ordered ``(kind, config)`` steps (instantiated from the registry) over
        the given seeds. ``aoi`` (a gpkg path) restricts tilerizing to an area of interest. Pass one of:
        ``resume_from`` (continue a prior run — its unchanged prefix is skipped; same folder, or a new
        ``output_dir`` for cross-folder) or ``initialize_from`` (seed this new-config run from a prior
        run's latest Imagery + Objects; components restart at id 0)."""
        if resume_from and initialize_from:
            raise ValueError("Pass only one of resume_from / initialize_from, not both.")
        aois_config = cls._build_aois_config(aoi)
        components = cls._make_components(steps, aois_config)

        if initialize_from:
            prior = cls.from_dir(initialize_from)
            tiles = tiles if tiles is not None else prior.latest(Imagery)
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

    @classmethod
    def _make_components(cls, steps, aois_config):
        """Instantiate every component, collecting MissingExtraError across ALL of them before
        raising — a pipeline missing e.g. detrex AND mmdet reports both, with one combined
        ``canopyrs setup`` command, instead of a fix-one-rerun-hit-the-next loop. Only
        MissingExtraError is collected; any other construction failure raises immediately."""
        components, missing = [], []
        for kind, config in steps:
            try:
                components.append(cls._make_component(kind, config, aois_config))
            except MissingExtraError as e:
                missing.append((kind, getattr(config, 'model', '?'), e))
        if missing:
            targets = sorted({e.target for _, _, e in missing if e.target})
            message = ("This pipeline needs optional frameworks that aren't installed or aren't working:\n"
                       + "\n".join(f"  - {kind} ({model}): {e.reason}" for kind, model, e in missing))
            if targets:
                message += f"\n\nInstall everything at once:\n  canopyrs setup {' '.join(targets)}"
            raise MissingExtraError(message, target=targets[0] if targets else None)
        return components

    @staticmethod
    def _build_aois_config(aoi):
        """A geodataset AOIConfig restricting tilerizing to ``aoi`` (a gpkg path), or None (whole raster)."""
        if aoi is None:
            return None
        return parse_tilerizer_aoi_config(aoi_config="package", aoi_type=None, aois={"infer": str(aoi)})

    # --- run -----------------------------------------------------------------
    def run(self, resume=None, verbose=True, strict_rgb_validation=True):
        """Run the components in order over the seeds (given at construction), and return ``self``
        (inspect ``self.tiles`` / ``self.objects`` / ... after). With ``output_dir`` set, each output is
        saved and the run record written. ``resume`` (defaulting to what ``from_config`` recorded)
        skips an already-completed prefix (unchanged config + outputs on disk).
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
            args = [self._resolve(need, component) for need in as_requirements(component.requires)]
            out = component.run(*args)
            produced = list(out) if isinstance(out, tuple) else [out]   # a component returns one table or a tuple
            self._check_produced(component, produced)
            for table in produced:
                self._store(table)
            self.outputs.append(produced)
            if self.output_dir is not None:
                self._save(component, produced)

        if self.output_dir is not None:
            store.save_seeds(self.output_dir, self.seeds)
            self.run_record = store.write_run_record(self.output_dir, self.components, self.outputs)
            self._write_final_gpkg()
        green_print("Pipeline finished")
        return self

    def _validate_sources(self, strict_rgb_validation):
        """Pre-flight on the seed imagery before any compute: first 3 bands are RGB-tagged (per
        ``strict_rgb_validation``), uint8, and in [0, 255]. Every rgb source scene is checked; a
        seeded tiles/crops folder is spot-checked on its first image (one folder shares one format).
        Other modalities aren't RGB rasters and are skipped."""
        for seed in self.seeds:
            if not isinstance(seed, Imagery) or Col.PATH not in seed.df.columns:
                continue
            df = seed.df
            if Col.MODALITY in df.columns:
                df = df[df[Col.MODALITY] == Modality.RGB]
            paths = df[Col.PATH].dropna()
            if not isinstance(seed, Sources):
                paths = paths.iloc[:1]
            for path in paths:
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
        (newest per type, mirroring runtime input matching) and confirm every ``requires`` is satisfiable.
        Raises on the first unmet requirement. Called at construction. Optimistic about ancestry (a
        produced column stays reachable while the chain links ``prev_objects``); ``run``'s checks are
        the ground truth."""
        for component, before, _ in self.thread_schemas():
            if component is None:
                continue
            for need in as_requirements(component.requires):
                desc, err = need.resolve(before.get(need.data_type))
                if desc is None:
                    raise ValueError(f"{type(component).__name__} {err}")
        return self

    def thread_schemas(self):
        """Yield ``(component, before, after)`` per step — ``before``/``after`` are ``{data_type:
        Schema}`` snapshots of the newest schema per type immediately before/after the component
        (matching only ever binds the newest instance of a type, so that's all the simulation keeps;
        types are separate keys, so e.g. produced tiles never shadow the source raster). The first
        yield is ``(None, None, seeds)``. Pure simulation, no compute: the single source of "what's
        available when", consumed by ``validate`` and the flow chart."""
        available, newest_imagery = {}, None
        for seed in self.seeds:
            available[type(seed)] = seed.schema()
            if isinstance(seed, Imagery):
                newest_imagery = available[type(seed)]
        yield None, None, dict(available)
        for component in self.components:
            before = dict(available)
            consumes_objects = Objects in before and any(
                need.data_type is Objects for need in as_requirements(component.requires))
            for need in as_requirements(component.produces):
                prev = available.get(need.data_type)
                columns = set(need.columns)
                links = set(need.links)
                fks = getattr(need.data_type, "fks", {})
                columns |= {fks[link] for link in need.links if link in fks}   # a link's FK column is a column too
                # Ancestry: an Objects-consuming component's Objects output chains lineage at runtime
                # (prev_objects), so everything reachable on the input stays reachable on the output.
                if prev is not None and need.data_type is Objects \
                        and ("prev_objects" in need.links or consumes_objects):
                    columns |= prev.columns
                    links |= prev.links
                # Modalities propagate producer <- input: a tilerizer's tiles hold its source's modalities.
                modalities = None
                if issubclass(need.data_type, Imagery) and newest_imagery is not None:
                    modalities = newest_imagery.modalities
                on = need.on if not isinstance(need.on, tuple) else None   # a tuple is a requires-only OR
                schema = Schema(columns=columns, links=links, crs=need.crs, on=on,
                                modalities=modalities)
                available[need.data_type] = schema
                if issubclass(need.data_type, Imagery):
                    newest_imagery = schema
            yield component, before, dict(available)

    def print_flow_chart(self):
        PipelineFlowVisualizer(self).print()

    def latest(self, data_type):
        """The most recently produced (or seeded) instance of ``data_type``, or None. ``Imagery``
        (the base) means the newest imagery of any role."""
        if data_type is Imagery:
            return self._imagery_log[-1] if self._imagery_log else None
        produced = self._lists[data_type]
        return produced[-1] if produced else None

    # --- export --------------------------------------------------------------
    def export(self, fmt, end_at=None, start_at=0, path=None, scores_column=None, categories_column=None):
        """Write a single GPKG or COCO file for the Objects produced at component ``end_at`` (default:
        the last that produced Objects) and return its path. Columns are merged from the ancestry but
        only those introduced by components in ``[start_at, end_at]``; with repeats (e.g. two
        aggregators) the latest value wins (``Objects.column``). ``fmt`` is ``"gpkg"`` or ``"coco"``."""
        record = self.run_record or (store.read_run_record(self.output_dir) if self.output_dir else None)
        if record is None:
            raise ValueError("no run record available; run() the pipeline with an output_dir before exporting")
        end_at = self._last_objects_index() if end_at is None else end_at
        anchor = self._objects_at(end_at)
        if anchor is None:
            raise ValueError(f"component {end_at} did not produce Objects to export")
        columns = self._window_columns(record, start_at, end_at)
        directory = self.output_dir / f"{record[end_at]['id']}_{record[end_at]['name']}"
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
        image_path = self._image_path_per_object(anchor)   # each object's image file, if reachable
        if image_path is not None:
            data[store.EXPORT_TILE_PATH] = image_path.values
        return gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=anchor.df.crs)

    def _export_coco(self, anchor: Objects, columns, path, scores_column, categories_column):
        imagery = anchor.linked("imagery")
        if imagery is None:
            raise ValueError("COCO export needs imagery linked to the objects (none reachable in the ancestry)")
        try:
            image_id = anchor.column(Col.IMAGE_ID)
        except KeyError:
            raise ValueError("COCO export needs each object's image_id (none reachable in the ancestry)")
        # CRS geometry converts per image file, so window images (e.g. crops) may export against their
        # materialized ancestor's file; pixel geometry is only valid against the image's own file.
        if anchor.crs_set:
            paths = image_id.map(pd.Series(imagery.resolved_paths().values,
                                           index=imagery.df[Col.IMAGE_ID].values))
        elif Col.PATH in imagery.df.columns:
            paths = image_id.map(imagery.df.set_index(Col.IMAGE_ID)[Col.PATH])
        else:
            paths = pd.Series([None] * len(anchor.df))
        if paths.isna().any() or (paths.astype(str) == "").any():
            raise ValueError("COCO export needs tile images on disk (re-run the tilerizer with save_tiles_to_disk=True)")

        scores_column = scores_column or self._latest_in(columns, store.SCORE_COLS)
        categories_column = categories_column or self._latest_in(columns, store.CLASS_COLS)

        data = {store.EXPORT_TILE_PATH: paths.values, Col.GEOMETRY: anchor.df.geometry.values}
        others = []
        for col in sorted(columns):
            if col in (store.EXPORT_TILE_PATH, Col.GEOMETRY):
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
        use_rle = bool(len(anchor)) and anchor.df[Col.GEOM_KIND].iloc[0] == GeomKind.MASK
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
    def _window_columns(record, start_at, end_at):
        columns = set()
        for entry in record[start_at:end_at + 1]:
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

    def _image_path_per_object(self, anchor: Objects):
        imagery = anchor.linked("imagery")
        if imagery is None or Col.PATH not in imagery.df.columns:
            return None
        try:
            image_id = anchor.column(Col.IMAGE_ID)
        except KeyError:
            return None
        return image_id.map(imagery.df.set_index(Col.IMAGE_ID)[Col.PATH])

    @staticmethod
    def _latest_in(columns, ordered):
        present = [col for col in ordered if col in columns]
        return present[-1] if present else None

    # --- reload + resume -----------------------------------------------------
    @classmethod
    def from_dir(cls, root):
        """Reload a saved run for inspection / re-export: reconstruct the typed tables in order,
        re-linking FKs from their persisted columns. Data-only (no components) — everything ``export``
        needs comes from the run record and the reloaded tables."""
        root = Path(root)
        record = store.read_run_record(root)
        if record is None:
            raise FileNotFoundError(f"no {store.RUN_RECORD} in {root}")
        pipe = cls([], output_dir=root)
        pipe.run_record = record
        pipe._load_seeds(root)                              # seed tables first, so produced FKs relink
        pipe._load_prefix(record, len(record), root)
        return pipe

    def _load_seeds(self, root):
        """Reload persisted seed tables (``_seed/``) before any component output, so produced tables
        can relink their FKs (e.g. ``imagery``) to a seeded table — the case of a run seeded from a
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
        record = store.read_run_record(self.output_dir) if self.output_dir else None
        if not record:
            return 0
        done = self._done_prefix(record)
        self._load_prefix(record, done, self.output_dir)
        for entry in record[:done]:
            green_print(f"Skipping {entry['id']}_{entry['name']} (already done, resumed)")
        return done

    def _done_prefix(self, record) -> int:
        done = 0
        for i, component in enumerate(self.components):
            if i >= len(record):
                break
            entry = record[i]
            if entry.get("name") != component.name or entry.get("config_hash") != store.config_hash(component.config):
                break
            directory = self._component_dir(component)
            if not all((directory / produced["file"]).exists() for produced in entry["produces"]):
                break
            done = i + 1
        return done

    def _load_prefix(self, record, count, root):
        for entry in record[:count]:
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
        """A typed table from its saved dataframe, re-linking FKs (matching how tables were threaded
        at run time: tables reload in the same order, so a table's imagery relations point at the most
        recently loaded imagery whose ids cover the FK values)."""
        if issubclass(data_type, Imagery):
            parent = self._covering_imagery(df, Col.PARENT_ID)
            related = {"parent": parent} if parent is not None else {}
            return data_type(df, **related)
        related = {}
        imagery = self._covering_imagery(df, Col.IMAGE_ID)
        if imagery is not None:
            related["imagery"] = imagery
        prev = self.latest(Objects)
        if prev is not None and has_usable_values(df, Col.PREV_OBJECT_ID):
            related["prev_objects"] = prev
        return Objects(df, **related)

    def _covering_imagery(self, df, col):
        """The most recently loaded imagery table (any role) whose ids cover ``df[col]``, or None."""
        values = set(df[col].dropna()) if col in df.columns else set()
        if not values:
            return None
        return next((table for table in reversed(self._imagery_log)
                     if values <= set(table.df[table.pk])), None)

    # --- internals -----------------------------------------------------------
    def _component_dir(self, component):
        return self.output_dir / f"{component.component_id}_{component.name}" if self.output_dir else None

    def _save(self, component, produced):
        directory = self._component_dir(component)
        for table in produced:
            store.save_table(table, directory)

    def _store(self, data):
        self._lists[type(data)].append(data)
        if isinstance(data, Imagery):
            self._imagery_log.append(data)

    def _resolve(self, need, component):
        """The input for a ``requires`` Need: the newest stored instance of the requested type,
        strictly checked — a mismatch raises instead of falling back to an older instance."""
        tables = self._lists.get(need.data_type)
        inst, err = need.resolve(tables[-1] if tables else None)
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
            err = need.check(inst.schema())
            if err:
                raise ValueError(f"{name} produced {need.data_type.__name__} but {err}")
