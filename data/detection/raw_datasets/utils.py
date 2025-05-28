import os
import tarfile
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


def download_with_progress(url, output_file):
    """
    Downloads a file from a URL with a progress bar.

    :param url: URL to download the file from.
    :param output_file: Path to save the downloaded file.
    """
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get the total file size from headers
    total_size = int(response.headers.get("content-length", 0))

    # Download the file with a progress bar
    with open(output_file, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(output_file)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            pbar.update(len(chunk))


def uncompress_with_progress(archive_file, output_dir):
    """
    Extracts a ZIP or .tar.gz file with a progress bar and recursively extracts first-level child ZIP files.

    :param archive_file: Path to the archive file to extract.
    :param output_dir: Directory to extract the contents to.
    """
    def extract_zip(file, target_dir):
        """
        Extracts a single ZIP file with progress.
        """
        with zipfile.ZipFile(file, "r") as zf:
            file_list = zf.namelist()
            with tqdm(
                desc=f"Extracting {Path(file).name}",
                total=len(file_list),
                unit="file",
            ) as pbar:
                for item in file_list:
                    zf.extract(item, target_dir)
                    pbar.update(1)

    def extract_tar(file, target_dir):
        """
        Extracts a .tar.gz file with progress.
        """

        with tarfile.open(file, mode="r:gz") as tf:
            file_list = tf.getnames()
            with tqdm(
                desc=f"Extracting {Path(file).name}",
                total=len(file_list),
                unit="file",
            ) as pbar:
                for item in file_list:
                    tf.extract(item, target_dir)
                    pbar.update(1)

        Path(file.parent/file.stem).rename(file.parent/Path(file.stem).stem)      # removing the .tar extension

    # Determine file type and extract accordingly
    if zipfile.is_zipfile(archive_file):
        extract_zip(archive_file, output_dir)
    elif tarfile.is_tarfile(archive_file):
        extract_tar(archive_file, output_dir)
    else:
        raise ValueError("Unsupported file format. Only .zip and .tar.gz are supported.")

    os.remove(archive_file)  # Remove the archive file after extraction

    # Track nested ZIP files for cleanup
    extracted_path = Path(output_dir)
    nested_archives = list(extracted_path.glob("**/*.zip")) + list(extracted_path.glob("**/*.tar.gz"))

    for nested_archive in nested_archives:
        print(f"Found nested archive: {nested_archive}, extracting...")
        nested_output_dir = nested_archive.parent / nested_archive.stem
        nested_output_dir.mkdir(exist_ok=True)
        if zipfile.is_zipfile(nested_archive):
            extract_zip(nested_archive, nested_output_dir)
        elif tarfile.is_tarfile(nested_archive):
            extract_tar(nested_archive, nested_output_dir)
        nested_archive.unlink()  # Remove the nested archive after extraction
        print(f"Extracted and removed nested archive: {nested_archive}")


def download_and_unzip(url, output_dir, dataset_name):
    """
    Downloads a ZIP file from a URL and extracts it to a specific directory with progress bars.

    :param url: The URL of the ZIP file to download.
    :param output_dir: The directory to save and extract the ZIP file.
    :param dataset_name: The name of the dataset for logging purposes.
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # File path for the downloaded ZIP
    zip_file = os.path.join(output_dir, f"{dataset_name}.zip")

    # Download the ZIP file
    download_with_progress(url, zip_file)

    # Unzip the contents
    uncompress_with_progress(zip_file, output_dir)
