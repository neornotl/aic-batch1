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

KEYS = [os.environ["AIC1"], os.environ["AIC3"]]


def ask(image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode()

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
        key = KEYS[attempt % len(KEYS)]
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


def load_done():
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as file:
            for line in file:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:
                    pass
    return done


images = sorted([
    path for path in ROOT.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

done = load_done()
todo = [
    path for path in images
    if str(path.relative_to(ROOT)) not in done
]

failed = set()

print("Total images:", len(images), flush=True)
print("Already done:", len(done), flush=True)
print("Remaining:", len(todo), flush=True)

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(ask, path): path
        for path in todo
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
                f"Progress {number}/{len(todo)} | failed={len(failed)}",
                flush=True,
            )

FAILED.write_text(
    json.dumps(sorted(failed), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Finished", flush=True)
print("Results:", OUT, flush=True)
print("Failed:", FAILED, flush=True)
