import React, { useEffect, useMemo, useRef, useState } from 'react';
import Loading from './Loading';
import SearchBar from './SearchBar';
import AnimeCard from './AnimeCard';
import { fetchAnimeResults, fetchRecommendations } from '../utils/api';

const SUGGESTED_QUERIES = ['Naruto', 'Your Name', 'Attack on Titan'];

function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedAnime, setSelectedAnime] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [expandedRecommendations, setExpandedRecommendations] = useState(new Set());
  const [isSearching, setIsSearching] = useState(false);
  const [isFetchingRecommendations, setIsFetchingRecommendations] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [recommendationError, setRecommendationError] = useState(null);
  const recommendationRef = useRef(null);

  const filteredResults = useMemo(
    () =>
      searchResults.filter(
        (anime) => !selectedAnime.some((selected) => selected.name === anime.name)
      ),
    [searchResults, selectedAnime]
  );

  useEffect(() => {
    setExpandedRecommendations(new Set());
  }, [recommendations]);

  const handleSearch = async (queryValue) => {
    const query = typeof queryValue === 'string' ? queryValue : searchQuery;
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    setSearchError(null);
    setIsSearching(true);
    try {
      const results = await fetchAnimeResults(trimmedQuery);
      setSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchError('We ran into an issue fetching search results. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelect = (anime) => {
    setSelectedAnime((previous) => {
      const exists = previous.some((item) => item.name === anime.name);
      return exists ? previous.filter((item) => item.name !== anime.name) : [...previous, anime];
    });
  };

  const handleClearSelections = () => {
    setSelectedAnime([]);
  };

  const handleGetRecommendations = async () => {
    if (selectedAnime.length === 0) return;

    setRecommendationError(null);
    setIsFetchingRecommendations(true);
    try {
      const liked = selectedAnime.map((anime) => anime.name);
      const recs = await fetchRecommendations(liked);
      setRecommendations(recs);
      setTimeout(() => {
        recommendationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 120);
    } catch (error) {
      console.error('Recommendation fetch failed:', error);
      setRecommendationError('Unable to generate recommendations right now. Give it another go soon.');
    } finally {
      setIsFetchingRecommendations(false);
    }
  };

  const hasSelections = selectedAnime.length > 0;
  const searchHasNoMatches =
    !isSearching && searchQuery.trim() && filteredResults.length === 0 && !searchError;

  return (
    <div className="page-shell">
      <section className="hero container">
        <div className="hero__content">
          <span className="hero__eyebrow">Personalized anime discovery</span>
          <h1 className="hero__title">Find your next binge-worthy series</h1>
          <p className="hero__subtitle">
            Tell us which shows you love and we&apos;ll curate tailored recommendations powered by
            collaborative filtering.
          </p>
        </div>

        <div className="hero__search">
          <SearchBar
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            onSearch={() => handleSearch()}
            isLoading={isSearching}
          />
          <div className="hero__hints">
            <span>Try:</span>
            <div className="hero__suggestions">
              {SUGGESTED_QUERIES.map((query) => (
                <button
                  key={query}
                  type="button"
                  className="chip"
                  onClick={() => {
                    setSearchQuery(query);
                    handleSearch(query);
                  }}
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="hero__actions">
          <button
            className="btn btn-primary btn-lg"
            onClick={handleGetRecommendations}
            disabled={!hasSelections || isFetchingRecommendations}
          >
            {isFetchingRecommendations ? 'Generating recommendations…' : 'Get recommendations'}
          </button>
          <p className="hero__meta">Select a few favorites to unlock personalized picks.</p>
        </div>
      </section>

      <main className="content container">
        <section className="panel">
          <header className="panel__header">
            <div>
              <h2>Your favorites</h2>
              <p className="panel__description">
                Add or remove titles to fine-tune what the recommender learns about your taste.
              </p>
            </div>
            <div className="panel__actions">
              <span className="panel__counter">{selectedAnime.length} selected</span>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={handleClearSelections}
                disabled={!hasSelections}
              >
                Clear all
              </button>
            </div>
          </header>

          {hasSelections ? (
            <>
              <div className="selected-list" aria-label="Selected anime">
                {selectedAnime.map((anime) => (
                  <button
                    key={`pill-${anime.anime_id ?? anime.name}`}
                    type="button"
                    className="selected-pill"
                    onClick={() => handleSelect(anime)}
                  >
                    <span>{anime.name}</span>
                    <span className="selected-pill__icon" aria-hidden="true">×</span>
                    <span className="sr-only">Remove {anime.name}</span>
                  </button>
                ))}
              </div>

              <div className="card-grid">
                {selectedAnime.map((anime) => (
                  <AnimeCard
                    key={`selected-card-${anime.anime_id ?? anime.name}`}
                    anime={anime}
                    selected
                    onClick={() => handleSelect(anime)}
                  />
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <h3>No favorites yet</h3>
              <p>Search for a show you love and tap it to add it to your favorites.</p>
            </div>
          )}
        </section>

        <section className="panel">
          <header className="panel__header">
            <div>
              <h2>Search results</h2>
              <p className="panel__description">
                Explore the catalog and tap a title to add it to your favorites list.
              </p>
            </div>
            {filteredResults.length > 0 && (
              <span className="panel__counter">{filteredResults.length} matches</span>
            )}
          </header>

          {searchError && <div className="alert alert--error">{searchError}</div>}

          {isSearching ? (
            <Loading message="Searching anime…" />
          ) : searchHasNoMatches ? (
            <div className="empty-state">
              <h3>No titles found</h3>
              <p>Try tweaking the spelling or search for another keyword.</p>
            </div>
          ) : (
            <div className="card-grid">
              {filteredResults.map((anime) => (
                <AnimeCard
                  key={anime.anime_id ?? anime.name}
                  anime={anime}
                  onClick={() => handleSelect(anime)}
                />
              ))}
            </div>
          )}
        </section>

        <section className="panel" ref={recommendationRef}>
          <header className="panel__header">
            <div>
              <h2>Recommended for you</h2>
              <p className="panel__description">
                Generated based on the shows you selected. Expand a card to read the synopsis.
              </p>
            </div>
            {recommendations.length > 0 && (
              <span className="panel__counter">{recommendations.length} suggestions</span>
            )}
          </header>

          {recommendationError && <div className="alert alert--error">{recommendationError}</div>}

          {isFetchingRecommendations && recommendations.length === 0 ? (
            <Loading message="Crafting recommendations…" />
          ) : recommendations.length > 0 ? (
            <div className="card-grid card-grid--recommendations">
              {recommendations.map((anime, index) => {
                const isExpanded = expandedRecommendations.has(index);
                return (
                  <AnimeCard
                    key={`recommend-${anime.anime_id ?? index}`}
                    anime={anime}
                    expanded={isExpanded}
                    expandable
                    onClick={() => {
                      const nextSet = new Set(expandedRecommendations);
                      if (isExpanded) {
                        nextSet.delete(index);
                      } else {
                        nextSet.add(index);
                      }
                      setExpandedRecommendations(nextSet);
                    }}
                  />
                );
              })}
            </div>
          ) : (
            <div className="empty-state">
              <h3>No recommendations yet</h3>
              <p>Select at least one show and generate recommendations to see curated picks.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default Home;
