import os
import shutil
from config import DATA_DIR, HDF5_FILES, KAGGLE_DATASET


def list_local_files(target_dir: str = DATA_DIR):
    """Print which HDF5 files are already downloaded."""
    print(f"\nLocal HDF5 files in '{target_dir}':")
    total_gb = 0.0
    for fname in HDF5_FILES:
        path = os.path.join(target_dir, fname)
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / 1e9
            total_gb += size_gb
            print(f"  [OK]  {fname}  ({size_gb:.2f} GB)")
        else:
            print(f"  [--]  {fname}  (not downloaded)")
    print(f"  Total: {total_gb:.2f} GB\n")


def download_single_file(
    filename: str,
    target_dir: str = DATA_DIR,
    cache_dir: str = None,
):
    """
    Download one HDF5 file from Kaggle using kagglehub.
    cache_dir: where kagglehub stores temp files (set to a drive with space).
    """
    out_path = os.path.join(target_dir, filename)

    if os.path.exists(out_path):
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"[kaggle] '{filename}' already exists ({size_gb:.2f} GB). Skipping.")
        return

    # Set cache dir to avoid C: drive space issues
    if cache_dir:
        os.environ["KAGGLE_CACHE_DIR"] = cache_dir
    else:
        # Default cache next to data dir
        os.environ["KAGGLE_CACHE_DIR"] = os.path.join(
            os.path.dirname(target_dir), "kaggle_cache"
        )

    import kagglehub

    print(f"[kaggle] Downloading: {filename}")
    print(f"[kaggle] Cache dir  : {os.environ['KAGGLE_CACHE_DIR']}")
    print(f"[kaggle] Target dir : {os.path.abspath(target_dir)}\n")

    os.makedirs(target_dir, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        KAGGLE_DATASET,
        path=filename,
    )
    print(f"\n[kaggle] Downloaded to cache: {downloaded_path}")

    # Copy from cache to data dir
    shutil.copy2(downloaded_path, out_path)

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[kaggle] Copied to: {out_path}  ({size_gb:.2f} GB)")


def download_dataset(
    target_dir: str = DATA_DIR,
    cache_dir: str = None,
):
    """Download all 10 HDF5 files. Use after download_single_file works."""
    for fname in HDF5_FILES:
        print(f"\n{'='*50}")
        download_single_file(fname, target_dir, cache_dir)

    list_local_files(target_dir)
