from abc import ABC, abstractmethod
from pathlib import Path

from canopyrs.data.detection.raw_datasets.utils import download_and_unzip


class BasePublicZipDataset(ABC):
    name = None
    zip_url = None

    def download(self, output_path: str or Path):
        output_path = Path(output_path) / self.name
        download_and_unzip(self.zip_url, output_path, self.name)
        self._parse(output_path)

    @abstractmethod
    def _parse(self, path: str or Path):
        pass

    @abstractmethod
    def tilerize(self,
                 raw_path: str or Path,
                 output_path: str or Path,
                 ground_resolution: float,
                 scale_factor: float,
                 tile_size: int,
                 tile_overlap: float,
                 binary_category: bool,
                 **kwargs):
        pass
