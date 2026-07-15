from pathlib import Path
from typing import Dict, Type, Union, Iterator, Tuple, Optional, List

from geodataset.utils import CocoNameConvention

from canopyrs.data.detection.preprocessed_datasets import QuebecTreesDataset


class BaseClassifierPreprocessedDataset:
    dataset_name: str = None
    pipeline_outputs_root: Optional[Path] = None

    def verify_dataset(
            self,
            root_output_path: Union[str, Path],
            folds: List[str],
    ):
        pass

    def iter_fold_classifier(
            self,
            root_output_path: Union[str, Path],
            fold: str,
            hf_root: str = "CanopyRS"
    ) -> Iterator[Tuple[
        str,
        str,
        Path,
        Optional[str],
        Optional[str],
        str,
    ]]:
        raise NotImplementedError


DATASET_REGISTRY: Dict[str, Type[BaseClassifierPreprocessedDataset]] = {}


def register_dataset(
        cls: Type[BaseClassifierPreprocessedDataset]
) -> Type[BaseClassifierPreprocessedDataset]:
    name = getattr(cls, "dataset_name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a dataset_name")
    DATASET_REGISTRY[name] = cls
    return cls


@register_dataset
class QuebecTreesClassifierDataset(QuebecTreesDataset):
    pipeline_outputs_root: Optional[Path] = None

    @staticmethod
    def _scan_pipeline_outputs(output_folder: Path) -> Dict[int, dict]:
        """Index a pipeline run's ``{id}_{name}/`` component folders, keyed by component id. Each entry
        holds the component ``name``, its ``folder``, and any output files found by naming convention
        (``coco``, ``gpkg``, ``pre_aggregated_gpkg``)."""
        components: Dict[int, dict] = {}
        for component_path in output_folder.iterdir():
            if not component_path.is_dir():
                continue
            try:
                component_id_str, component_name = component_path.name.split('_', 1)
                component_id = int(component_id_str)
            except ValueError:
                continue

            entry = {'name': component_name, 'folder': component_path}
            for coco_file in component_path.glob('*.json'):
                if '_coco_' in coco_file.name:
                    entry['coco'] = coco_file
            for gpkg_file in component_path.glob('*.gpkg'):
                file_type = 'pre_aggregated_gpkg' if 'notaggregated' in gpkg_file.name else 'gpkg'
                entry[file_type] = gpkg_file
            components[component_id] = entry
        return components

    def _resolve_inputs_from_pipeline_output(
            self,
            pipeline_output_folder: Path,
            product_name: Optional[str] = None,
    ) -> Tuple[Path, Path, Optional[Path]]:
        # NOTE (classifier-on-v3 TODO): this scavenges a run folder for the tilerizer's product
        # tiles/ dir and a geodataset-named infer COCO. The v3 grid tilerizer writes tiles to disk but
        # no COCO (it's export("coco")-on-demand), so this lookup must be revisited when the classifier
        # benchmark is ported to consume v3 runs. Behavior is unchanged from the v1 DataState version.
        components = self._scan_pipeline_outputs(pipeline_output_folder)

        tilerizer_ids = [cid for cid, entry in components.items() if entry['name'] == 'tilerizer']
        if not tilerizer_ids:
            raise FileNotFoundError(
                f"No tilerizer component folder found in {pipeline_output_folder}."
            )
        tilerizer_id = max(tilerizer_ids)
        tilerizer_folder = components[tilerizer_id]['folder']

        product_folders = [
            p for p in tilerizer_folder.iterdir()
            if p.is_dir() and (p / 'tiles').exists()
        ]
        if not product_folders:
            raise FileNotFoundError(
                f"No product folder with a tiles/ directory found in "
                f"{tilerizer_folder}."
            )

        # TODO: multi-product run support.
        # For now, pipeline_output_folder is expected to correspond to a
        # single-product run
        # (one tilerizer product folder containing tiles/).
        # When running multiple products in a single pipeline output folder,
        # we should select the correct product folder
        # deterministically (and likely validate that we found an exact match).
        if len(product_folders) == 1:
            product_folder = product_folders[0]
        else:
            product_folder = None
            if product_name is not None:
                for p in product_folders:
                    if product_name in p.name:
                        product_folder = p
                        break
            if product_folder is None:
                product_folder = sorted(product_folders)[0]
        tiles_path = product_folder

        coco_candidates = sorted(product_folder.glob('*_coco_*infer*.json'))
        if not coco_candidates:
            coco_candidates = sorted(product_folder.glob('*_coco_*.json'))
        if not coco_candidates:
            raise FileNotFoundError(f"No tilerizer COCO file found in {product_folder}.")
        input_coco = coco_candidates[0]

        aggregator_ids = [
            cid for cid, entry in components.items()
            if entry['name'] == 'aggregator' and cid < tilerizer_id
        ]
        input_gpkg = None
        if aggregator_ids:
            input_gpkg = components[max(aggregator_ids)].get('gpkg')

        return tiles_path, input_coco, input_gpkg

    def _find_infer_coco(
            self,
            product_root: Path,
            product_name: str,
    ) -> Optional[Path]:
        try_names = []
        try:
            try_names.append(CocoNameConvention.create_name(
                product_name=product_name,
                fold='infer',
                ground_resolution=self.ground_resolution,
            ))
        except (TypeError, ValueError):
            pass

        if getattr(self, 'scale_factor', None) is not None:
            try:
                try_names.append(CocoNameConvention.create_name(
                    product_name=product_name,
                    fold='infer',
                    scale_factor=self.scale_factor,
                ))
            except (TypeError, ValueError):
                pass

        for name in try_names:
            candidate = product_root / name
            if candidate.exists():
                return candidate

        candidates = sorted(product_root.glob(f"*{product_name}*infer*.json"))
        if candidates:
            return candidates[0]

        candidates = sorted(product_root.glob("*infer*.json"))
        if candidates:
            return candidates[0]

        return None

    def _find_infer_gpkg(
            self,
            product_root: Path,
            product_name: str,
    ) -> Optional[Path]:
        candidates = sorted(product_root.glob(f"*{product_name}*infer*.gpkg"))
        if candidates:
            return candidates[0]

        candidates = sorted(product_root.glob("*infer*.gpkg"))
        if candidates:
            return candidates[0]

        return None

    def iter_fold_classifier(
            self,
            root_output_path: Union[str, Path],
            fold: str,
            hf_root: str = "CanopyRS"
    ) -> Iterator[Tuple[
        str,
        str,
        Path,
        Optional[str],
        Optional[str],
        str,
    ]]:
        root = Path(root_output_path)

        for (location, product_name, tile_dir, _aoi_gpkg_path,
             gt_gpkg_path, gt_coco_path) in super().iter_fold(
            root_output_path=root,
            fold=fold,
            hf_root=hf_root,
        ):
            input_coco_path = None
            input_gpkg_path = None
            tiles_path = tile_dir

            if self.pipeline_outputs_root is not None:
                pipeline_outputs_root = Path(self.pipeline_outputs_root)
                if (pipeline_outputs_root / '0_tilerizer').exists():
                    pipeline_output_folder = pipeline_outputs_root
                elif (pipeline_outputs_root / product_name).exists():
                    pipeline_output_folder = pipeline_outputs_root / product_name
                else:
                    pipeline_output_folder = pipeline_outputs_root

                tiles_path, input_coco_path, input_gpkg_path = (
                    self._resolve_inputs_from_pipeline_output(
                        pipeline_output_folder,
                        product_name=product_name,
                    )
                )
            else:
                product_root = root / location / product_name
                input_coco_path = self._find_infer_coco(
                    product_root,
                    product_name,
                )
                input_gpkg_path = self._find_infer_gpkg(
                    product_root,
                    product_name,
                )

                if input_coco_path is None:
                    input_coco_path = Path(gt_coco_path)

                if input_gpkg_path is None and gt_gpkg_path is not None:
                    input_gpkg_path = Path(gt_gpkg_path)

            yield (
                location,
                product_name,
                tiles_path,
                str(input_gpkg_path) if input_gpkg_path is not None else None,
                str(input_coco_path) if input_coco_path is not None else None,
                str(gt_coco_path),
            )
