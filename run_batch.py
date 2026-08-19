import base64
import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FAILED = Path(sys.argv[3])
LOCK = threading.Lock()

KEYS = [os.environ["AIC2"], os.environ["AIC3"]]

import itertools

# AIC2/AIC3 only (AIC1 exhausted). Retries cycle through AIC2/AIC3
CYCLE = [1, 0, 1, 0, 1]

KEY_COUNTER = itertools.count()


def ask(image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    start = next(KEY_COUNTER) % len(CYCLE)

    payload = {
        "model": "gpt-5.6-luna",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this AIC keyframe. "
                        "Return valid JSON only:\n"
                        "{\n"
                        '  "scene_description": "detailed description",\n'
                        '  "on_screen_text": "all visible text, empty string if none",\n'
                        '  "objects": ["important objects"],\n'
                        '  "actions": ["important actions"],\n'
                        '  "location": "scene location"\n'
                        "}\n"
                        "Do not use markdown."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded,
                    },
                },
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 700,
    }

    for attempt in range(5):
        key = KEYS[CYCLE[(start + attempt) % len(CYCLE)]]
        try:
            response = requests.post(
                "https://api.pateway.ai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                return answer, None

            print(
                image_path.name,
                "HTTP",
                response.status_code,
                "retry",
                attempt + 1,
                flush=True,
            )
        except Exception as error:
            print(
                image_path.name,
                error,
                "retry",
                attempt + 1,
                flush=True,
            )

        time.sleep(min(5 * (attempt + 1), 30))

    return None, "failed_after_5_retries"


def load_failed():
    """Load list of failed images to retry"""
    failed_to_retry = set()
    if FAILED.exists():
        try:
            with FAILED.open(encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        failed_to_retry.add(item)
        except Exception as e:
            print(f"Warning: Could not load failed list: {e}", flush=True)
    return failed_to_retry


def load_done():
    """Load list of already processed images"""
    done = set()
    if OUT.exists():
        try:
            with OUT.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        done.add(json.loads(line)["path"])
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: Could not load done list: {e}", flush=True)
    return done


# Load failed images (images to retry)
failed_to_retry = load_failed()

# Load done images (already processed)
done = load_done()

# Determine which images to process
if failed_to_retry:
    # Mode 1: Retry mode - only process failed images
    images = sorted([
        path for path in ROOT.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and str(path.relative_to(ROOT)) in failed_to_retry
        and str(path.relative_to(ROOT)) not in done
    ])
    mode = "RETRY"
else:
    # Mode 2: Full mode - process all images except done ones
    all_images = sorted([
        path for path in ROOT.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    images = [
        path for path in all_images
        if str(path.relative_to(ROOT)) not in done
    ]
    mode = "FULL"

failed = set()

print(f"Mode: {mode}", flush=True)
print("Already done:", len(done), flush=True)
print("Remaining:", len(images), flush=True)

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(ask, path): path
        for path in images
    }

    for number, future in enumerate(as_completed(futures), 1):
        path = futures[future]
        relative_path = str(path.relative_to(ROOT))

        try:
            answer, error = future.result()
        except Exception as exception:
            answer, error = None, str(exception)

        if answer is not None:
            record = {
                "path": relative_path,
                "file": path.name,
                "response": answer,
            }
            with LOCK:
                with OUT.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            failed.add(relative_path)

        if number % 10 == 0:
            FAILED.write_text(
                json.dumps(sorted(failed), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"Progress {number}/{len(images)} | failed={len(failed)}",
                flush=True,
            )

FAILED.write_text(
    json.dumps(sorted(failed), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Finished", flush=True)
print("Results:", OUT, flush=True)
print("Failed:", FAILED, flush=True)
