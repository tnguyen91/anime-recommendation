"""Favorites API endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import User, UserFavorite
from api.auth.dependencies import get_current_user
from api.favorites.schemas import (
    FavoriteCreate,
    FavoriteResponse,
    FavoriteListResponse,
    MessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/favorites", tags=["Favorites"])


def get_anime_info(anime_id: int) -> dict:
    """Get anime info from loaded metadata."""
    try:
        from api.main import anime_metadata, anime_df
        
        info = {}
        metadata = anime_metadata.get(str(anime_id)) or anime_metadata.get(anime_id, {})
        if metadata:
            info["name"] = metadata.get("title") or metadata.get("name")
            info["title_english"] = metadata.get("title_english")
            info["image_url"] = metadata.get("image_url")
        
        if not info.get("name") and anime_df is not None:
            match = anime_df[anime_df["anime_id"] == anime_id]
            if not match.empty:
                row = match.iloc[0]
                info["name"] = row.get("name")
                info["title_english"] = row.get("title_english")
        
        return info
    except Exception:
        return {}


@router.get("", response_model=FavoriteListResponse)
async def list_favorites(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all favorite anime for the current user."""
    favorites = (
        db.query(UserFavorite)
        .filter(UserFavorite.user_id == current_user.id)
        .order_by(UserFavorite.added_at.desc())
        .all()
    )
    
    enriched_favorites = []
    for fav in favorites:
        anime_info = get_anime_info(fav.anime_id)
        enriched_favorites.append(
            FavoriteResponse(
                id=fav.id,
                anime_id=fav.anime_id,
                added_at=fav.added_at,
                name=anime_info.get("name"),
                title_english=anime_info.get("title_english"),
                image_url=anime_info.get("image_url"),
            )
        )
    
    return FavoriteListResponse(
        favorites=enriched_favorites,
        total=len(enriched_favorites)
    )


@router.post("", response_model=FavoriteResponse, status_code=status.HTTP_201_CREATED)
async def add_favorite(
    favorite_data: FavoriteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add an anime to the user's favorites."""
    existing = (
        db.query(UserFavorite)
        .filter(
            UserFavorite.user_id == current_user.id,
            UserFavorite.anime_id == favorite_data.anime_id
        )
        .first()
    )
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anime already in favorites"
        )
    
    new_favorite = UserFavorite(
        user_id=current_user.id,
        anime_id=favorite_data.anime_id
    )
    
    db.add(new_favorite)
    db.commit()
    db.refresh(new_favorite)
    
    logger.info(f"User {current_user.id} added anime {favorite_data.anime_id} to favorites")
    
    anime_info = get_anime_info(new_favorite.anime_id)
    
    return FavoriteResponse(
        id=new_favorite.id,
        anime_id=new_favorite.anime_id,
        added_at=new_favorite.added_at,
        name=anime_info.get("name"),
        title_english=anime_info.get("title_english"),
        image_url=anime_info.get("image_url"),
    )


@router.delete("/{favorite_id}", response_model=MessageResponse)
async def remove_favorite(
    favorite_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove an anime from the user's favorites by favorite ID.
    
    Requires authentication.
    """
    favorite = (
        db.query(UserFavorite)
        .filter(
            UserFavorite.id == favorite_id,
            UserFavorite.user_id == current_user.id  # Ensure user owns this favorite
        )
        .first()
    )
    
    if not favorite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Favorite not found"
        )
    
    anime_id = favorite.anime_id
    db.delete(favorite)
    db.commit()
    
    logger.info(f"User {current_user.id} removed anime {anime_id} from favorites")
    
    return MessageResponse(message="Favorite removed successfully")


@router.delete("/anime/{anime_id}", response_model=MessageResponse)
async def remove_favorite_by_anime(
    anime_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove an anime from the user's favorites by anime ID.
    
    This is useful when you know the anime_id but not the favorite record ID.
    
    Requires authentication.
    """
    favorite = (
        db.query(UserFavorite)
        .filter(
            UserFavorite.anime_id == anime_id,
            UserFavorite.user_id == current_user.id
        )
        .first()
    )
    
    if not favorite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Anime not in favorites"
        )
    
    db.delete(favorite)
    db.commit()
    
    logger.info(f"User {current_user.id} removed anime {anime_id} from favorites")
    
    return MessageResponse(message="Favorite removed successfully")


@router.get("/check/{anime_id}")
async def check_favorite(
    anime_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Check if an anime is in the user's favorites.
    
    Returns {"is_favorite": true/false}
    
    Requires authentication.
    """
    exists = (
        db.query(UserFavorite)
        .filter(
            UserFavorite.anime_id == anime_id,
            UserFavorite.user_id == current_user.id
        )
        .first()
    ) is not None
    
    return {"is_favorite": exists, "anime_id": anime_id}
