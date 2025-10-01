# Anime Recommendation UI

A modern, responsive web interface for the anime recommendation system.

## Features

- **Search Anime**: Real-time search with autocomplete functionality
- **Favorites Management**: Add/remove anime to/from your favorites list
- **Recommendations**: Get personalized recommendations based on your favorites
- **Anime Details**: Click on any anime to view detailed information including synopsis and genres
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Offline Support**: Service worker for improved performance and caching

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom CSS with modern design principles
- **Icons**: Font Awesome 6
- **Fonts**: Inter (Google Fonts)
- **API**: FastAPI backend with anime recommendation engine

## Quick Start

### Option 1: Docker Compose (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Run the complete stack:
   ```bash
   docker-compose up
   ```
3. Open your browser and navigate to:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

### Option 2: Local Development

1. Serve the website folder using any HTTP server:
   ```bash
   # Using Python
   cd website
   python -m http.server 3000
   
   # Using Node.js
   npx serve website -l 3000
   
   # Using PHP
   cd website
   php -S localhost:3000
   ```

2. Open your browser and navigate to http://localhost:3000

## Usage

1. **Search for Anime**: Type in the search box to find anime titles
2. **Add to Favorites**: Click the heart button next to search results
3. **View Favorites**: See your favorited anime in the "My Favorites" section
4. **Get Recommendations**: Click "Get Recommendations" to receive personalized suggestions
5. **View Details**: Click on any anime card to see detailed information

## Features Overview

### Search Functionality
- Real-time search with 300ms debounce
- Results show anime title, alternative titles, and genres
- Image thumbnails when available
- One-click add to favorites

### Favorites Management
- Persistent storage using localStorage
- Visual feedback for favorite status
- Bulk clear option
- Click to view details

### Recommendations
- Based on collaborative filtering using RBM (Restricted Boltzmann Machine)
- Shows recommended anime with images, genres, and synopsis
- Click any recommendation for full details

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimized layouts
- Touch-friendly interface
- Accessible design patterns

## API Integration

The UI integrates with the following API endpoints:

- `GET /search-anime?query=<query>` - Search for anime titles
- `POST /recommend` - Get recommendations based on liked anime

API Base URL: `https://animereco-api-725392014501.us-west1.run.app`

## Browser Compatibility

- Chrome/Chromium 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Performance Features

- Image lazy loading and error handling
- Request caching for search results
- Service worker for offline support
- Debounced search to reduce API calls
- CSS animations with hardware acceleration

## Customization

### Changing API Endpoint

Edit `script.js` and update the `apiBaseUrl` property:

```javascript
constructor() {
    this.apiBaseUrl = 'your-api-endpoint-here';
    // ... rest of constructor
}
```

### Styling

All styles are in `styles.css`. Key variables for theming:

- Primary color: `#ff6b6b` (red/pink)
- Background gradient: `#667eea` to `#764ba2` (blue to purple)
- Font family: Inter

### Adding Features

The `AnimeRecommendationApp` class in `script.js` is modular and can be extended with additional features.

## Troubleshooting

### CORS Issues
If you encounter CORS errors, make sure your API server has CORS properly configured for your domain.

### Images Not Loading
Some anime images may not load due to CORS restrictions or broken URLs. The UI handles this gracefully by hiding broken images.

### Search Not Working
Verify that the API endpoint is accessible and the search endpoint is working by testing it directly in your browser.

## License

This UI is part of the anime recommendation project. Check the main project README for license information.