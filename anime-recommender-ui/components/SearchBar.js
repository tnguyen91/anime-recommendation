import React from 'react';

function SearchBar({ value, onChange, onSearch, isLoading }) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSearch?.();
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit} role="search">
      <label htmlFor="anime-search" className="sr-only">
        Search anime titles
      </label>
      <div className="search-field">
        <span className="search-icon" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="7" />
            <line x1="20" y1="20" x2="16.65" y2="16.65" />
          </svg>
        </span>
        <input
          id="anime-search"
          className="search-input"
          placeholder="Search by title, studio, or keyword"
          value={value}
          onChange={onChange}
          autoComplete="off"
        />
        <button type="submit" className="btn btn-primary" disabled={isLoading}>
          {isLoading ? 'Searching…' : 'Search'}
        </button>
      </div>
    </form>
  );
}

export default SearchBar;
