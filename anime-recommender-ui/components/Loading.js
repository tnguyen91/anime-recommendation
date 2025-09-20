import React from 'react';

const Loading = ({ message = 'Loading…' }) => (
  <div className="loading-state" role="status" aria-live="polite">
    <span className="loading-spinner" aria-hidden="true" />
    <span className="loading-text">{message}</span>
  </div>
);

export default Loading;
