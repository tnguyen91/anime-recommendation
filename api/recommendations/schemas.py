from pydantic import BaseModel, Field

from api.config import DEFAULT_TOP_N


class RecommendRequest(BaseModel):
    liked_anime: list[str] = Field(
        ..., min_length=1, description="List of anime names the user likes"
    )
    top_n: int = Field(
        DEFAULT_TOP_N, ge=1, le=50, description="Number of recommendations to return"
    )
    exclude_ids: list[int] = Field(
        default_factory=list, max_length=200, description="Anime IDs to exclude"
    )


class AnimeResult(BaseModel):
    anime_id: int
    name: str
    title_english: str | None = None
    title_japanese: str | None = None
    image_url: str | None = None
    genre: list[str] = Field(default_factory=list)
    synopsis: str | None = None


class RecommendResponse(BaseModel):
    recommendations: list[AnimeResult]
    request_id: str | None = None


class SearchResponse(BaseModel):
    results: list[AnimeResult]
    total: int
    limit: int
    offset: int
