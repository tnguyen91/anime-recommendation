import React from 'react';

function SearchBar({ value, onChange, onSearch, onKeyDown }) {
  return (
    <div className="search-bar">
      <input
        className="search-input"
        placeholder="Search your favorite anime"
        value={value}
        onChange={onChange}
        onKeyDown={onKeyDown}
      />
      <button className="button" onClick={onSearch}>
        Search
      </button>
    </div>
  );
}

export default SearchBar;