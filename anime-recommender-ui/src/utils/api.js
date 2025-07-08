const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export const fetchAnimeResults = async (query) => {
  const response = await fetch(`${API_URL}/search-anime?query=${encodeURIComponent(query)}`);
  const data = await response.json();
  return data.results || [];
};

export const fetchRecommendations = async (likedAnime) => {
  const response = await fetch(`${API_URL}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked_anime: likedAnime }),
  });
  const data = await response.json();
  return data.recommendations || [];
};