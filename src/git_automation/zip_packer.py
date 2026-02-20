import os
import zipfile
import glob
from typing import Optional, List


class ZipPacker:
    """
    Compresses run artifacts into a zip file for efficient git tracking.
    """

    def __init__(self, run_dir: str, include_checkpoints: bool = True):
        self.run_dir = run_dir
        self.include_checkpoints = include_checkpoints
        self.exclude_patterns = [
            "*.pyc",
            "__pycache__",
            ".pytest_cache",
            "*.log",
            "visuals/*",
        ]
        if not include_checkpoints:
            self.exclude_patterns.extend(["*.pth", "*.pt", "*.ckpt"])

    def _should_exclude(self, filepath: str) -> bool:
        """Check if a file matches any of the exclude patterns."""
        import fnmatch

        filename = os.path.basename(filepath)

        # Also exclude the zip files themselves
        if filename.endswith(".zip"):
            return True

        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
            # Handle directory patterns
            if "*/*" in pattern and pattern.split("/")[0] in filepath:
                return True
        return False

    def create_zip(
        self, epoch: int, total_epochs: int, output_dir: str = "artifacts"
    ) -> str:
        """
        Zip the run directory contents into a single file.

        Returns:
            str: Path to the created zip file
        """
        os.makedirs(output_dir, exist_ok=True)
        run_name = os.path.basename(os.path.abspath(self.run_dir))

        zip_filename = f"{run_name}_epoch_{epoch}_of_{total_epochs}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        print(f"[Packer] Creating artifact archive: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.run_dir):
                # Optionally exclude directories if they match patterns
                # This could be more robust, but sufficient for now
                if any(
                    excl.replace("/*", "") in root
                    for excl in self.exclude_patterns
                    if "/*" in excl
                ):
                    continue

                for file in files:
                    filepath = os.path.join(root, file)
                    if not self._should_exclude(filepath):
                        # Add to zip with relative path corresponding to run_dir
                        arcname = os.path.relpath(filepath, start=self.run_dir)
                        # We put everything inside a folder named after the run
                        zipf.write(filepath, arcname=os.path.join(run_name, arcname))

        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[Packer] Archive created ({size_mb:.2f} MB)")
        return zip_path

    def get_latest_zip(self, output_dir: str = "artifacts") -> Optional[str]:
        """Find the latest zip artifact for this run."""
        run_name = os.path.basename(os.path.abspath(self.run_dir))
        zips = glob.glob(os.path.join(output_dir, f"{run_name}_epoch_*.zip"))

        if not zips:
            return None

        # Parse epochs from filenames to find the max
        # Filename format: {run_name}_epoch_{epoch}_of_{total_epochs}.zip
        def get_epoch(path):
            try:
                # Extracts the {epoch} number
                return int(os.path.basename(path).split("_epoch_")[1].split("_of_")[0])
            except (IndexError, ValueError):
                return -1

        latest_zip = max(zips, key=get_epoch)
        return latest_zip
