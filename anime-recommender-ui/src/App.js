import React, { useState } from "react";
import "./App.css";

function App() {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedAnime, setSelectedAnime] = useState([]);
  const [recommendations, setRecommendations] = useState([]);

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

  const handleClick = (animeName) => {
    setSelectedAnime((prev) =>
      prev.includes(animeName)
        ? prev.filter((name) => name !== animeName)
        : [...prev, animeName]
    );
  };

  const getRecommendations = async () => {
    try {
      const response = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ liked_anime: selectedAnime }),
      });

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error("Recommendation fetch failed:", err);
    }
  };

  return (
    <div>
      <h1 className="title">Anime Recommender</h1>
      <h2 className="subtitle">Search and select your favorite anime</h2>

      <input
        className="text"
        placeholder="Type anime name..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        style={{ padding: "8px", width: "250px", marginRight: "10px" }}
      />
      <button className="button" onClick={handleSearch}>
        Search
      </button>

      <ul style={{ listStyleType: "none", padding: 0, marginTop: "20px" }}>
        {/* Show selected items on top */}
        {selectedAnime
          .filter(
            (selected) => !searchResults.some((anime) => anime.Name === selected)
          )
          .map((name) => (
            <li
              key={name}
              onClick={() => handleClick(name)}
              className="anime-list-item selected"
              style={{
                cursor: "pointer",
                padding: "10px",
                border: "1px solid #ccc",
                borderRadius: "5px",
                marginBottom: "10px",
                backgroundColor: "#d1e7f0",
              }}
            >
              <strong>{name}</strong>
              <span style={{ color: "#888", marginLeft: "8px" }}>(Selected)</span>
            </li>
          ))}

        {/* Show search results */}
        {searchResults.map((anime) => {
          const isSelected = selectedAnime.includes(anime.Name);
          const engName = anime["English name"];
          const showEng = engName && engName !== "Unknown" && engName !== anime.Name;

          return (
            <li
              key={anime.MAL_ID}
              onClick={() => handleClick(anime.Name)}
              className={`anime-list-item ${isSelected ? "selected" : ""}`}
              style={{
                cursor: "pointer",
                padding: "10px",
                border: "1px solid #ccc",
                borderRadius: "5px",
                marginBottom: "10px",
                backgroundColor: isSelected ? "#d1e7f0" : "#f9f9f9",
              }}
            >
              <strong>{anime.Name}</strong>
              {showEng && (
                <span style={{ color: "#888", marginLeft: "8px" }}>
                  ({engName})
                </span>
              )}
            </li>
          );
        })}
      </ul>

      <button
        className="button"
        onClick={getRecommendations}
        disabled={selectedAnime.length === 0}
      >
        Get Recommendations
      </button>

      {recommendations.length > 0 && (
        <div className="text" style={{ marginTop: "30px" }}>
          <h3>Recommended Anime:</h3>
          <ul>
            {recommendations.map((anime, index) => (
              <li key={index}>
                <strong>{anime.Name}</strong>{" "}
                <span style={{ color: "#888" }}>(Score: {anime.score})</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;