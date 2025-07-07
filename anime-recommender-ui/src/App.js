import React, { useState } from "react";
import "./App.css";

function App() {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedAnime, setSelectedAnime] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [expandedRecommendations, setExpandedRecommendations] = useState(new Set());


  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      const response = await fetch(
        `http://localhost:5000/search-anime?query=${encodeURIComponent(searchQuery)}`
      );
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (err) {
      console.error("Search failed:", err);
    }
  };

  const handleClick = (anime) => {
    setSelectedAnime((prev) => {
      const exists = prev.some((a) => a.name === anime.name);
      return exists
        ? prev.filter((a) => a.name !== anime.name)
        : [...prev, anime];
    });
  };

  const getRecommendations = async () => {
    try {
      const response = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ liked_anime: selectedAnime.map((a) => a.name) }),
      });

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error("Recommendation fetch failed:", err);
    }
  };

  return (
    <div>
      <div className="search-bar">
        <input
          className="search-input"
          placeholder="Search your favorite anime"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <button className="button" onClick={handleSearch}>
          Search
        </button>
      </div>
      
      <div className="recommend-button-wrapper">
        <button
          className="button"
          onClick={getRecommendations}
          disabled={selectedAnime.length === 0}
        >
          Get Recommendations
        </button>
      </div>

      <div className="card-grid">
        {selectedAnime.map((anime) => (
          <div
            key={anime.name}
            className="anime-card selected"
            onClick={() => handleClick(anime)}
            title={anime.name}
          >
            <div className="star-icon">★</div>
            {anime.image_url && (
              <div className="anime-thumb-wrapper">
                <img src={anime.image_url} alt={anime.name} className="anime-thumb" />
                <div className="overlay-title">
                  <strong>{anime.name}</strong>
                </div>
              </div>
            )}
          </div>
        ))}
        {/* Search Results */}
        {searchResults
          .filter((anime) => !selectedAnime.some((a) => a.name === anime.name))
          .map((anime) => (
            <div
              key={anime.anime_id}
              className="anime-card"
              onClick={() => handleClick(anime)}
              title={anime.name}
            >
              {anime.image_url && (
                <div className="anime-thumb-wrapper">
                  <img src={anime.image_url} alt={anime.name} className="anime-thumb" />
                  <div className="overlay-title">
                    <strong>{anime.name}</strong>
                  </div>
                </div>
              )}
            </div>
        ))}
      </div>


      {recommendations.length > 0 && (
        <div className="recommendation-section">
          <h3 className="subtitle">Recommended Anime</h3>
          <div className="card-grid">
            {recommendations.map((anime, index) => {
              const isExpanded = expandedRecommendations.has(index);
              return (
                <div
                  className={`anime-card ${isExpanded ? 'selected' : ''}`}
                  key={index}
                  onClick={() => {
                    const newSet = new Set(expandedRecommendations);
                    if (isExpanded) {
                      newSet.delete(index);
                    } else {
                      newSet.add(index);
                    }
                    setExpandedRecommendations(newSet);
                  }}
                >
                  {anime.image_url && (
                    <img
                      src={anime.image_url}
                      alt={anime.name}
                      className="anime-thumb"
                    />
                  )}

                  <div className="overlay-title">
                    <strong>{anime.name}</strong>
                  </div>

                  {isExpanded && (
                    <div className="card-info">
                      {anime.genre.length > 0 && (
                        <p className="genres"><strong>Genres:</strong> {anime.genre.join(", ")}</p>
                      )}
                      {anime.synopsis && (
                        <p className="synopsis"><strong>Synopsis:</strong> {anime.synopsis}</p>
                      )}
                    </div>
                  )}
                </div>
              );
            })}

          </div>
        </div>
      )}
    </div>
  );
}

export default App;