from __future__ import annotations
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_env_cache = os.environ.get("CACHE_DIR")
if _env_cache:
    CACHE_DIR = Path(_env_cache).resolve()
else:
    CACHE_DIR = Path("/tmp/cache").resolve()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

ProgressCallback = Callable[[int, int], None]

def _build_session(max_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "anime-recommender/1.0"})
    return session

def download_file(
    url: str,
    destination_folder: str = ".",
    filename: Optional[str] = None,
    timeout: Union[int, Tuple[int, int]] = (5, 30),
    checksum: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    chunk_size: int = 64 * 1024,
    session: Optional[requests.Session] = None,
) -> Path:

    dest_dir = Path(destination_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = Path(url.split("/")[-1]).name or "download"

    final_path = dest_dir / Path(filename).name

    try:
        if final_path.exists() and final_path.stat().st_size > 0:
            return final_path
    except Exception:
        pass

    sess = session or _build_session()

    with sess.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length")) if resp.headers.get("Content-Length") else -1

        fd, tmp_name = tempfile.mkstemp(prefix=final_path.name + ".", dir=str(dest_dir))
        os.close(fd)
        tmp_path = Path(tmp_name)

        hash_obj = hashlib.sha256() if checksum else None
        downloaded = 0
        try:
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if hash_obj:
                        hash_obj.update(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        try:
                            progress_cb(downloaded, total)
                        except Exception:
                            pass
                f.flush()
                os.fsync(f.fileno())

            if checksum and hash_obj:
                if hash_obj.hexdigest().lower() != checksum.lower():
                    tmp_path.unlink(missing_ok=True)
                    raise ValueError("checksum mismatch for downloaded file")

            if total != -1 and tmp_path.stat().st_size != total:
                tmp_path.unlink(missing_ok=True)
                raise IOError("downloaded file size does not match Content-Length")

            os.replace(str(tmp_path), str(final_path))
            return final_path
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

def download_to_cache(url: str, **kwargs) -> Path:
    filename = Path(url.split("/")[-1]).name or "download"
    return download_file(url, destination_folder=str(CACHE_DIR), filename=filename, **kwargs)
