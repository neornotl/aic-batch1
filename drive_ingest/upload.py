"""Low-request official BTC video ingestion.

Videos remain individually viewable in Drive.  The official keyframe archive
and the timestamp index are uploaded once per logical batch, avoiding one
Drive API upload per frame (177k+ frames).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

try:
    from .remote_zip import RemoteFile
except ImportError:  # script execution from the repository root
    from remote_zip import RemoteFile

BASE = "https://aic-data.ledo.io.vn/"
VIDEO_ARCHIVES = [
    "Videos_L21_a.zip", "Videos_L22_a.zip", "Videos_L23_a.zip",
    "Videos_L24_a.zip", "Videos_L25_a.zip", "Videos_L26_a.zip",
    "Videos_L26_b.zip", "Videos_L26_c.zip", "Videos_L26_d.zip",
    "Videos_L26_e.zip", "Videos_L27_a.zip", "Videos_L28_a.zip",
    "Videos_L29_a.zip", "Videos_L30_a.zip",
]
KEYFRAME_ARCHIVES = [
    "Keyframes_L21.zip", "Keyframes_L22.zip", "Keyframes_L23.zip",
    "Keyframes_L24.zip", "Keyframes_L25.zip", "Keyframes_L26_a.zip",
    "Keyframes_L26_b.zip", "Keyframes_L26_c.zip", "Keyframes_L26_d.zip",
    "Keyframes_L26_e.zip", "Keyframes_L27.zip", "Keyframes_L28.zip",
    "Keyframes_L29.zip", "Keyframes_L30.zip",
]
LOGICAL_BATCHES = {
    **{f"L{i}": [f"Videos_L{i}_a.zip", f"Keyframes_L{i}.zip"]
       for i in [21, 22, 23, 24, 25, 27, 28, 29, 30]},
    **{f"L26_{p}": [f"Videos_L26_{p}.zip", f"Keyframes_L26_{p}.zip"]
       for p in "abcde"},
}

# One transfer at a time and large resumable chunks reduce Drive API calls.
RCLONE_PACING = [
    "--tpslimit", "1", "--tpslimit-burst", "1",
    "--drive-pacer-min-sleep", "10s", "--drive-pacer-burst", "1",
    "--transfers", "1", "--checkers", "1",
    "--drive-chunk-size", "256M",
    "--retries", "6", "--low-level-retries", "20", "--retries-sleep", "30s",
]


def rclone(*args):
    return ["rclone", *RCLONE_PACING, *args]


def stream_url_to_drive(url, destination):
    """Upload a remote archive as one file without storing it locally."""
    print(f"streaming archive -> {destination}", flush=True)
    proc = subprocess.Popen(rclone("rcat", destination), stdin=subprocess.PIPE)
    try:
        with urllib.request.urlopen(url, timeout=120) as src:
            while True:
                block = src.read(8 * 1024 * 1024)
                if not block:
                    break
                proc.stdin.write(block)
        proc.stdin.close()
        code = proc.wait()
    except Exception:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise
    if code:
        raise SystemExit(f"archive upload failed: {destination}")


def load_timelines(path):
    timelines = {}
    with open(path, encoding="utf8") as mapping:
        for line in mapping:
            try:
                row = json.loads(line)
                timelines.setdefault(row.get("video_id"), []).append(row)
            except Exception:
                continue
    return timelines


def write_batch_timestamps(destination, video_ids, timelines):
    fd, tmp = tempfile.mkstemp(prefix="timestamps-", suffix=".jsonl")
    os.close(fd)
    with open(tmp, "w", encoding="utf8") as out:
        for vid in sorted(video_ids):
            for row in timelines.get(vid, []):
                out.write(json.dumps({
                    k: row.get(k) for k in
                    ("video_id", "keyframe", "keyframe_number", "frame_id", "timestamp_s", "fps")
                }, separators=(",", ":")) + "\n")
    subprocess.run(rclone("copyto", tmp, destination), check=True)
    os.unlink(tmp)


def upload_videos(archive, folder):
    catalog = []
    z = zipfile.ZipFile(RemoteFile(BASE + archive))
    for info in z.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".mp4"):
            continue
        match = re.search(r"(L\d+_V\d+)", info.filename)
        if not match:
            continue
        vid = match.group(1)
        dest = f"{folder}/{vid[:3]}/{vid}/{vid}.mp4"
        print(f"uploading {vid}", flush=True)
        proc = subprocess.Popen(rclone("rcat", dest), stdin=subprocess.PIPE)
        try:
            with z.open(info) as src:
                while block := src.read(8 * 1024 * 1024):
                    proc.stdin.write(block)
            proc.stdin.close()
            code = proc.wait()
        except Exception:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.kill()
            proc.wait()
            raise
        if code:
            raise SystemExit(f"video upload failed: {vid}")
        catalog.append({"video_id": vid, "archive": archive,
                        "member": info.filename, "drive_path": dest})
    return catalog


def process_batch(batch, video_archive, keyframe_archive, timelines):
    folder = "drive:01_Videos_Original"
    batch_root = f"{folder}/{batch}"
    catalog = upload_videos(video_archive, folder)
    timestamp_dest = f"{batch_root}/timestamps_{batch}.jsonl"
    write_batch_timestamps(timestamp_dest, [row["video_id"] for row in catalog], timelines)
    keyframe_dest = f"{batch_root}/keyframes/keyframes_{batch}.zip"
    stream_url_to_drive(BASE + keyframe_archive, keyframe_dest)

    fd, catalog_path = tempfile.mkstemp(prefix=f"catalog-{batch}-", suffix=".jsonl")
    os.close(fd)
    with open(catalog_path, "w", encoding="utf8") as out:
        for row in catalog:
            row.update({"timestamp_index": timestamp_dest,
                        "keyframe_archive": keyframe_dest})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    subprocess.run(rclone("copyto", catalog_path, f"drive:catalog_{batch}.jsonl"), check=True)
    os.unlink(catalog_path)
    print(f"completed {batch}: {len(catalog)} videos", flush=True)


def main(name):
    if name == "all":
        batches = list(LOGICAL_BATCHES)
    elif name in LOGICAL_BATCHES:
        batches = [name]
    else:
        raise SystemExit("archive must be L21..L30, L26_a..L26_e, or all")
    map_path = os.environ.get("TIMESTAMP_MAP")
    if not map_path:
        raise SystemExit("TIMESTAMP_MAP must point to the exact BTC timestamp map")
    timelines = load_timelines(map_path)
    for batch in batches:
        video_archive, keyframe_archive = LOGICAL_BATCHES[batch]
        process_batch(batch, video_archive, keyframe_archive, timelines)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
