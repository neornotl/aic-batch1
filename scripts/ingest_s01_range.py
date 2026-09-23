#!/usr/bin/env python3
"""
Ingest Video_S01 to Google Drive one video at a time using HTTP Range requests.
Immune to disk space exhaustion.
"""

import os
import sys
import subprocess
from pathlib import Path
from aic_pipeline.range_zip import RangeZip

def main():
    url = "https://aic-data.ledo.io.vn/Video_S01.zip"
    print(f"Connecting to {url} via RangeZip...")
    r = RangeZip(url)
    print(f"Archive parsed. Total entries: {len(r.entries)}")

    temp_dir = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "s01_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dest_drive = "drive:01_Videos_Batch2/Video_S01"

    # Check which files already exist on Drive to avoid redundant uploads
    try:
        res = subprocess.run(["rclone", "lsf", dest_drive], capture_output=True, text=True, check=True)
        existing = set(res.stdout.strip().splitlines())
    except Exception as ex:
        print(f"Could not list existing drive files: {ex}")
        existing = set()

    print(f"Existing files on Drive ({len(existing)}): {existing}")

    for name in sorted(r.entries.keys()):
        filename = Path(name).name
        if filename in existing:
            print(f"[=] Skipping {filename} (already on Drive)")
            continue

        size_gb = round(r.entries[name].uncompressed_size / 1e9, 2)
        local_file = temp_dir / filename
        print(f"[*] Downloading {filename} ({size_gb} GB) via parallel byte ranges...")
        r.download_parallel(name, local_file, workers=8)

        print(f"[*] Uploading {filename} to {dest_drive} ...")
        subprocess.run([
            "rclone", "copy", str(local_file), dest_drive,
            "--transfers", "4",
            "--checkers", "8",
            "--stats", "20s",
            "-v"
        ], check=True)

        local_file.unlink(missing_ok=True)
        print(f"[✓] {filename} successfully uploaded and cleaned up!")

    print("[SUCCESS] All Video_S01 videos ingested successfully!")

if __name__ == "__main__":
    main()
