```python
from dataset.detection.public_datasets import BCI50haDataset, ReforesTreeDataset, NeonTreeEvaluationDataset,
    OamTcdDataset

raw_output = '/network/scratch/h/hugo.baudchon/data/raw'
cross_validation = False
tile_size = 1333
tile_overlap = 0.5
ground_resolution = 0.025
scale_factor = None

folds = (
    'train',
    'valid',
    'test'
)

tilerized_output = f'/network/scratch/h/hugo.baudchon/data/tilerized2/tilerized'
                   f'_{str(tile_size)}_{str(tile_overlap).replace(".", "p")}'
                   f'_{str(ground_resolution).replace(".", "p")}_{str(scale_factor).replace(".", "p")}'

annotation_types = ['box', 'mask']

bci50ha = BCI50haDataset()
# bci50ha.download(output_path=raw_output)
bci50ha.tilerize(raw_path=raw_output,
                 output_path=tilerized_output,
                 cross_validation=cross_validation,
                 tile_size=tile_size,
                 tile_overlap=tile_overlap,
                 ground_resolution=ground_resolution,
                 scale_factor=scale_factor)
#
refores_tree = ReforesTreeDataset()
# refores_tree.download(output_path=raw_output)
refores_tree.tilerize(raw_path=raw_output,
                      output_path=tilerized_output,
                      cross_validation=cross_validation,
                      tile_size=tile_size,
                      tile_overlap=tile_overlap,
                      ground_resolution=ground_resolution,
                      scale_factor=scale_factor)

neon_tree_evaluation = NeonTreeEvaluationDataset()
neon_tree_evaluation.download(output_path=raw_output)
neon_tree_evaluation.tilerize(raw_path=raw_output,
                              output_path=tilerized_output,
                              cross_validation=cross_validation,
                              tile_size=tile_size,
                              tile_overlap=tile_overlap,
                              ground_resolution=ground_resolution,
                              scale_factor=scale_factor)

oam_tcd = OamTcdDataset()
# oam_tcd.download(output_path=raw_output)
oam_tcd.tilerize(raw_path=raw_output,
                 output_path=tilerized_output,
                 remove_tree_group_annotations=True,
                 cross_validation=cross_validation,
                 tile_size=tile_size,
                 tile_overlap=tile_overlap,
                 ground_resolution=ground_resolution,
                 scale_factor=scale_factor)
```
