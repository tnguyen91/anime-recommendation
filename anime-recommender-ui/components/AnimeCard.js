import React from 'react';

function AnimeCard({ anime, onClick, selected, expanded, expandable }) {
  const hasImage = Boolean(anime?.image_url);
  const genres = Array.isArray(anime?.genre) ? anime.genre : [];
  const synopsis = anime?.synopsis;

  const buttonProps = expandable
    ? { 'aria-expanded': expanded }
    : { 'aria-pressed': Boolean(selected) };

  return (
    <article className={`anime-card ${selected ? 'is-selected' : ''} ${expanded ? 'is-expanded' : ''}`}>
      <button
        type="button"
        className="anime-card__button"
        onClick={onClick}
        title={anime?.name}
        {...buttonProps}
      >
        <div className="anime-card__media">
          {hasImage ? (
            <img
              src={anime.image_url}
              alt={anime?.name || 'Anime cover art'}
              loading="lazy"
            />
          ) : (
            <div className="anime-card__placeholder" aria-hidden="true">
              <span>{anime?.name?.charAt(0)}</span>
            </div>
          )}
          <span className="anime-card__gradient" aria-hidden="true" />
          {selected && <span className="anime-card__badge">Selected</span>}
        </div>

        <div className="anime-card__body">
          <h3 className="anime-card__title">{anime?.name}</h3>
          {genres.length > 0 && (
            <div className="anime-card__badges" aria-label="Genres">
              {genres.slice(0, 3).map((genre) => (
                <span key={genre} className="badge">
                  {genre}
                </span>
              ))}
              {genres.length > 3 && <span className="badge badge--muted">+{genres.length - 3}</span>}
            </div>
          )}
          {expandable && (
            <p className="anime-card__cta">Tap to {expanded ? 'hide' : 'view'} synopsis</p>
          )}
        </div>
      </button>

      {expandable && expanded && (
        <div className="anime-card__drawer">
          {genres.length > 0 && (
            <p className="anime-card__meta">
              <strong>Genres:</strong> {genres.join(', ')}
            </p>
          )}
          {synopsis && <p className="anime-card__synopsis">{synopsis}</p>}
        </div>
      )}
    </article>
  );
}

export default AnimeCard;
