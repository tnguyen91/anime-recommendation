"""Jikan API collector for fetching anime data from MyAnimeList."""
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

from data_pipeline.config import (
    JIKAN_BASE_URL,
    JIKAN_REQUEST_DELAY,
    JIKAN_CACHE_EXPIRATION_DAYS,
    CACHE_DIR,
    JIKAN_DATA_DIR,
)

logger = logging.getLogger(__name__)

@dataclass
class AnimeData:
    """Structured representation of anime metadata from Jikan."""
    mal_id: int
    title: str
    title_english: Optional[str] = None
    title_japanese: Optional[str] = None
    synopsis: Optional[str] = None
    genres: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    demographics: list[str] = field(default_factory=list)
    type: Optional[str] = None
    episodes: Optional[int] = None
    status: Optional[str] = None
    score: Optional[float] = None
    scored_by: Optional[int] = None
    members: Optional[int] = None
    popularity: Optional[int] = None
    aired_from: Optional[str] = None
    aired_to: Optional[str] = None
    season: Optional[str] = None
    year: Optional[int] = None
    studios: list[str] = field(default_factory=list)
    source: Optional[str] = None
    rating: Optional[str] = None
    image_url: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_jikan_response(cls, data: dict) -> "AnimeData":
        """Create an AnimeData instance from Jikan API response."""
        aired = data.get("aired", {}) or {}
        aired_from = aired.get("from")
        aired_to = aired.get("to")

        images = data.get("images", {}) or {}
        jpg_images = images.get("jpg", {}) or {}
        image_url = jpg_images.get("large_image_url") or jpg_images.get("image_url")

        def extract_names(items: list) -> list[str]:
            if not items:
                return []
            return [item.get("name") for item in items if item.get("name")]

        return cls(
            mal_id=data.get("mal_id"),
            title=data.get("title"),
            title_english=data.get("title_english"),
            title_japanese=data.get("title_japanese"),
            synopsis=data.get("synopsis"),
            genres=extract_names(data.get("genres", [])),
            themes=extract_names(data.get("themes", [])),
            demographics=extract_names(data.get("demographics", [])),
            type=data.get("type"),
            episodes=data.get("episodes"),
            status=data.get("status"),
            score=data.get("score"),
            scored_by=data.get("scored_by"),
            members=data.get("members"),
            popularity=data.get("popularity"),
            aired_from=aired_from,
            aired_to=aired_to,
            season=data.get("season"),
            year=data.get("year"),
            studios=extract_names(data.get("studios", [])),
            source=data.get("source"),
            rating=data.get("rating"),
            image_url=image_url,
        )

class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, min_interval: float = JIKAN_REQUEST_DELAY):
        self.min_interval = min_interval
        self.last_request_time: Optional[float] = None

    def wait(self):
        """Wait until it's safe to make another request."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

        self.last_request_time = time.time()

class CacheManager:
    """Manages caching of API responses to disk."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        expiration_days: int = JIKAN_CACHE_EXPIRATION_DAYS
    ):
        self.cache_dir = Path(cache_dir)
        self.expiration_days = expiration_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        if len(key) < 100 and key.replace("_", "").replace("-", "").isalnum():
            return self.cache_dir / f"{key}.json"
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[dict]:
        """Retrieve a cached response if it exists and hasn't expired."""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            expires_at = datetime.fromisoformat(cached["expires_at"])
            if datetime.utcnow() > expires_at:
                logger.debug(f"Cache expired for key: {key}")
                cache_path.unlink()
                return None
            logger.debug(f"Cache hit for key: {key}")
            return cached["data"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache file {cache_path}: {e}")
            cache_path.unlink()
            return None

    def set(self, key: str, data: dict):
        """Store a response in the cache."""
        cache_path = self._get_cache_path(key)

        cached_at = datetime.utcnow()
        expires_at = cached_at + timedelta(days=self.expiration_days)

        cache_entry = {
            "cached_at": cached_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "data": data,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)

        logger.debug(f"Cached response for key: {key}")

    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")

class JikanCollector:
    """Collector for fetching anime data from the Jikan API."""

    def __init__(
        self,
        base_url: str = JIKAN_BASE_URL,
        use_cache: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limiter = RateLimiter()
        self.cache = CacheManager() if use_cache else None
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": "AnimeRecommendationSystem/1.0 (Educational Project)",
                "Accept": "application/json",
            },
        )
        logger.info(f"JikanCollector initialized (cache={'enabled' if use_cache else 'disabled'})")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a rate-limited request to the Jikan API."""
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                response = self.client.get(url, params=params)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s before retry...")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise
        raise RuntimeError("Unexpected error in request loop")

    def _get_or_fetch(self, cache_key: str, endpoint: str, params: Optional[dict] = None) -> dict:
        """Get data from cache or fetch from API."""
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        response = self._make_request(endpoint, params)
        if self.cache:
            self.cache.set(cache_key, response)
        return response

    def fetch_anime_by_id(self, mal_id: int) -> Optional[AnimeData]:
        """Fetch a single anime by its MyAnimeList ID."""
        cache_key = f"anime_{mal_id}"
        try:
            response = self._get_or_fetch(cache_key, f"/anime/{mal_id}")
            data = response.get("data")
            if data:
                return AnimeData.from_jikan_response(data)
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Anime {mal_id} not found")
                return None
            raise

    def fetch_seasonal_anime(
        self,
        year: int,
        season: str,
        limit: Optional[int] = None
    ) -> list[AnimeData]:
        """Fetch anime from a specific season."""
        season = season.lower()
        if season not in ("winter", "spring", "summer", "fall"):
            raise ValueError(f"Invalid season: {season}. Must be winter, spring, summer, or fall.")

        anime_list = []
        page = 1

        while True:
            cache_key = f"seasonal_{year}_{season}_{page}"

            response = self._get_or_fetch(
                cache_key,
                f"/seasons/{year}/{season}",
                params={"page": page}
            )

            data = response.get("data", [])
            if not data:
                break

            for item in data:
                anime = AnimeData.from_jikan_response(item)
                anime_list.append(anime)

                if limit and len(anime_list) >= limit:
                    logger.info(f"Fetched {len(anime_list)} anime from {season} {year}")
                    return anime_list

            pagination = response.get("pagination", {})
            if not pagination.get("has_next_page", False):
                break

            page += 1

        logger.info(f"Fetched {len(anime_list)} anime from {season} {year}")
        return anime_list

    def fetch_current_season(self, limit: Optional[int] = None) -> list[AnimeData]:
        """Fetch anime from the current airing season."""
        anime_list = []
        page = 1

        while True:
            cache_key = f"season_now_{page}"

            response = self._get_or_fetch(
                cache_key,
                "/seasons/now",
                params={"page": page}
            )

            data = response.get("data", [])
            if not data:
                break

            for item in data:
                anime = AnimeData.from_jikan_response(item)
                anime_list.append(anime)

                if limit and len(anime_list) >= limit:
                    logger.info(f"Fetched {len(anime_list)} currently airing anime")
                    return anime_list

            pagination = response.get("pagination", {})
            if not pagination.get("has_next_page", False):
                break

            page += 1

        logger.info(f"Fetched {len(anime_list)} currently airing anime")
        return anime_list

    def fetch_top_anime(
        self,
        limit: int = 100,
        filter_type: Optional[str] = None
    ) -> list[AnimeData]:
        """Fetch top-ranked anime from MAL."""
        anime_list = []
        page = 1
        while len(anime_list) < limit:
            cache_key = f"top_{filter_type or 'all'}_{page}"

            params = {"page": page}
            if filter_type:
                params["filter"] = filter_type

            response = self._get_or_fetch(cache_key, "/top/anime", params=params)

            data = response.get("data", [])
            if not data:
                break

            for item in data:
                anime = AnimeData.from_jikan_response(item)
                anime_list.append(anime)

                if len(anime_list) >= limit:
                    break

            pagination = response.get("pagination", {})
            if not pagination.get("has_next_page", False):
                break

            page += 1

        logger.info(f"Fetched {len(anime_list)} top anime")
        return anime_list[:limit]

    def fetch_anime_batch(self, mal_ids: list[int]) -> list[AnimeData]:
        """Fetch multiple anime by their MAL IDs."""
        anime_list = []
        for i, mal_id in enumerate(mal_ids):
            logger.debug(f"Fetching anime {i+1}/{len(mal_ids)}: {mal_id}")
            anime = self.fetch_anime_by_id(mal_id)
            if anime:
                anime_list.append(anime)
        logger.info(f"Fetched {len(anime_list)}/{len(mal_ids)} anime")
        return anime_list

    def save_to_file(self, anime_list: list[AnimeData], filename: str):
        """Save a list of anime to a JSON file."""
        JIKAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = JIKAN_DATA_DIR / filename
        data = [anime.to_dict() for anime in anime_list]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(anime_list)} anime to {output_path}")

    @staticmethod
    def load_from_file(filepath: Path) -> list[AnimeData]:
        """Load anime data from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        anime_list = []
        for item in data:
            if "fetched_at" in item:
                anime = AnimeData(**item)
            else:
                anime = AnimeData.from_jikan_response(item)
            anime_list.append(anime)
        logger.info(f"Loaded {len(anime_list)} anime from {filepath}")
        return anime_list

def collect_recent_anime(save: bool = True) -> list[AnimeData]:
    """Collect current season and top 100 anime."""
    with JikanCollector() as collector:
        logger.info("Fetching current season anime...")
        current_season = collector.fetch_current_season()
        logger.info("Fetching top 100 anime...")
        top_anime = collector.fetch_top_anime(limit=100)

        seen_ids = set()
        combined = []
        for anime in current_season + top_anime:
            if anime.mal_id not in seen_ids:
                seen_ids.add(anime.mal_id)
                combined.append(anime)

        logger.info(f"Total unique anime collected: {len(combined)}")
        if save:
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            collector.save_to_file(combined, f"jikan_update_{timestamp}.json")
        return combined

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print("=" * 60)
    print("Jikan API Collector - Test Run")
    print("=" * 60)
    try:
        anime_list = collect_recent_anime(save=True)
        print(f"\nCollected {len(anime_list)} unique anime!")
        print("\nSample (first 5):")
        for anime in anime_list[:5]:
            print(f"  - {anime.title} (ID: {anime.mal_id}, Score: {anime.score})")
        print("\nData saved to data/raw/jikan/")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
