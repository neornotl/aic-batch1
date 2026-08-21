"""OpenAI-compatible Terra adapter with cache, retries, rate limiting, and parsing."""
from __future__ import annotations
import json, os, re, time, hashlib
from pathlib import Path
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def parse_json_response(value):
    if isinstance(value, dict): return value
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try: return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if match:
            try: return json.loads(match.group(0))
            except json.JSONDecodeError: pass
    return {}

class TerraAdapter:
    def __init__(self, cache_dir: Path | None = None, timeout=120, retries=4):
        # Terra must use an explicitly scoped credential. Do not silently reuse
        # a Luna key: that can select the wrong model and create unexpected cost.
        self.key = os.getenv("AIC_TERRA_API_KEY") or os.getenv("AIC_JUDGE_API_KEY", "")
        self.base = (os.getenv("AIC_TERRA_BASE_URL") or os.getenv("AIC_JUDGE_BASE_URL", "https://api.pateway.ai/v1")).rstrip("/")
        self.model = os.getenv("AIC_TERRA_MODEL") or os.getenv("AIC_JUDGE_MODEL", "gpt-5.6-terra")
        self.cache = cache_dir or Path(os.getenv("AIC_TERRA_CACHE", "work/aic_pipeline/terra_cache")); self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries = timeout, retries

    def complete(self, prompt: str, system: str = "Return JSON only.") -> dict:
        path = self.cache / (hashlib.sha256((self.model + system + prompt).encode()).hexdigest() + ".json")
        if path.exists():
            try: return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError: pass
        if not self.key: return {}
        payload = {"model": self.model, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}], "temperature": 0}
        for attempt in range(self.retries):
            try:
                response = requests.post(self.base + "/chat/completions", headers={"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}, json=payload, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500: raise RuntimeError(f"HTTP {response.status_code}")
                response.raise_for_status(); result = parse_json_response(response.json()["choices"][0]["message"]["content"])
                path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8"); return result
            except Exception:
                if attempt + 1 == self.retries: return {}
                time.sleep(min(2 ** attempt, 20))
        return {}
