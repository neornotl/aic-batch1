import json
from pathlib import Path

missing_videos = {'L26_V114', 'L26_V151', 'L26_V477', 'L26_V276'}

for i in range(1, 11):
    path = Path(f'work/aic_pipeline/overnight/shard_{i}.json')
    data = json.loads(path.read_text())
    original = len(data['video_ids'])
    data['video_ids'] = [v for v in data['video_ids'] if v not in missing_videos]
    path.write_text(json.dumps(data, ensure_ascii=False))
    print(f'Shard {i}: {original} -> {len(data["video_ids"])} videos')