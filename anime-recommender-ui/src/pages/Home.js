import React, { useState, useRef } from 'react';
import SearchBar from '../components/SearchBar';
import AnimeCard from '../components/AnimeCard';
import { fetchAnimeResults, fetchRecommendations } from '../utils/api';

function Home() {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedAnime, setSelectedAnime] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [expandedRecommendations, setExpandedRecommendations] = useState(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const recommendationRef = useRef(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsLoading(true);
    try {
      const results = await fetchAnimeResults(searchQuery);
      setSearchResults(results);
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClick = (anime) => {
    setSelectedAnime((prev) => {
      const exists = prev.some((a) => a.name === anime.name);
      return exists ? prev.filter((a) => a.name !== anime.name) : [...prev, anime];
    });
  };

  const handleGetRecommendations = async () => {
    try {
      const recs = await fetchRecommendations(selectedAnime.map((a) => a.name));
      setRecommendations(recs);
      setTimeout(() => {
        recommendationRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      console.error("Recommendation fetch failed:", err);
    }
  };

  return (
    <div>
      <SearchBar
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        onSearch={handleSearch}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
      />

      <div className="recommend-button-wrapper">
        <button className="button" onClick={handleGetRecommendations} disabled={selectedAnime.length === 0}>
          Get Recommendations
        </button>
      </div>

      {isLoading && <div className="subtitle">Loading...</div>}

      <div className="card-grid">
        {selectedAnime.map((anime) => (
          <AnimeCard key={anime.name} anime={anime} selected onClick={() => handleClick(anime)} />
        ))}

        {searchResults
          .filter((anime) => !selectedAnime.some((a) => a.name === anime.name))
          .map((anime) => (
            <AnimeCard key={anime.anime_id} anime={anime} onClick={() => handleClick(anime)} />
          ))}
      </div>

      {recommendations.length > 0 && (
        <div className="recommendation-section" ref={recommendationRef}>
          <h3 className="subtitle">Recommended Anime</h3>
          <div className="card-grid">
            {recommendations.map((anime, index) => {
              const isExpanded = expandedRecommendations.has(index);
              return (
                <AnimeCard
                  key={index}
                  anime={anime}
                  expanded={isExpanded}
                  expandable
                  onClick={() => {
                    const newSet = new Set(expandedRecommendations);
                    isExpanded ? newSet.delete(index) : newSet.add(index);
                    setExpandedRecommendations(newSet);
                  }}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;