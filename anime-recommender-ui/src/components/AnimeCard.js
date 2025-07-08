import React from 'react';

function AnimeCard({ anime, onClick, selected, expanded, expandable }) {
  return (
    <div
      className={`anime-card ${selected ? 'selected' : ''} ${expanded ? 'expanded' : ''}`}
      onClick={onClick}
      title={anime.name}
    >
      {selected && <div className="star-icon">★</div>}

      {anime.image_url && (
        <div className="anime-thumb-wrapper">
          <img
            src={anime.image_url}
            alt={anime.name}
            className="anime-thumb"
            loading="lazy"
          />
          <div className="overlay-title">
            <span className="overlay-text">
              <strong>{anime.name}</strong>
            </span>
          </div>
        </div>
      )}

      {expandable && expanded && (
        <div className="card-info">
          {anime.genre.length > 0 && (
            <p className="genres">
              <strong>Genres:</strong> {anime.genre.join(", ")}
            </p>
          )}
          {anime.synopsis && (
            <p className="synopsis">
              <strong>Synopsis:</strong> {anime.synopsis}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default AnimeCard;