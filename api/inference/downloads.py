from __future__ import annotations
import hashlib
import time
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse
import os

_env_cache = os.environ.get("CACHE_DIR")
if _env_cache:
    CACHE_DIR = Path(_env_cache).resolve()
else:
    azure_home = Path("/home/site/wwwroot")
    if azure_home.exists():
        CACHE_DIR = (azure_home / "cache").resolve()
    else:
        CACHE_DIR = Path("/tmp/cache").resolve()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_to_cache(url: str, max_retries: int = 3, timeout: int = 30) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    suffix = Path(urlparse(url).path).suffix or ""
    local_name = f"{url_hash}{suffix}"
    local_path = CACHE_DIR / local_name
    if local_path.exists():
        return local_path

    tmp_path = CACHE_DIR / (local_name + ".tmp")

    attempt = 0
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with tmp_path.open("wb") as f:
                    chunk_size = 1024 * 64
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)

            tmp_path.replace(local_path)
            return local_path

        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise
            last_exc = e
        except Exception as e:
            last_exc = e

        time.sleep(2 ** attempt)

    raise last_exc
