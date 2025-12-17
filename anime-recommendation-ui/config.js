window.APP_CONFIG = {
  // Base URL for the API server (without /api/v1 - that's added in app.js)
  API_BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://animereco-api-725392014501.us-west1.run.app'
};
