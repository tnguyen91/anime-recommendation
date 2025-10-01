/**
 * Anime Recommendation Application
 * Industrial-grade implementation following best practices
 * @version 1.0.0
 * @author Anime Recommendation Team
 */

'use strict';

class AnimeRecommendationApp {
    // Configuration constants
    static CONFIG = {
        API_BASE_URL: 'https://animereco-api-725392014501.us-west1.run.app',
        SEARCH_DEBOUNCE_MS: 300,
        MIN_SEARCH_LENGTH: 1,
        CACHE_EXPIRY_MS: 5 * 60 * 1000, // 5 minutes
        TOAST_DURATION_MS: 3000,
        MAX_SEARCH_QUERY_LENGTH: 100,
        STORAGE_KEYS: {
            FAVORITES: 'anime_favorites',
            CACHE: 'anime_cache'
        }
    };

    // UI Element selectors
    static SELECTORS = {
        SEARCH_INPUT: '#searchInput',
        SEARCH_RESULTS: '#searchResults',
        CLEAR_SEARCH: '#clearSearch',
        FAVORITES_LIST: '#favoritesList',
        FAVORITES_COUNT: '#favoritesCount',
        CLEAR_FAVORITES: '#clearFavorites',
        GET_RECOMMENDATIONS: '#getRecommendations',
        RECOMMENDATIONS_SECTION: '#recommendationsSection',
        RECOMMENDATIONS_LIST: '#recommendationsList',
        LOADING_SPINNER: '#loadingSpinner',
        ANIME_MODAL: '#animeModal',
        ANIME_DETAILS: '#animeDetails',
        TOAST_CONTAINER: '#toastContainer'
    };

    constructor() {
        this.favorites = new Set();
        this.searchTimeout = null;
        this.cache = new Map();
        this.abortController = null;
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        try {
            this.bindEvents();
            this.loadFavoritesFromStorage();
            this.loadCacheFromStorage();
            this.updateFavoritesUI();
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showToast('Failed to load application. Please refresh the page.', 'error');
        }
    }

    /**
     * Bind all event listeners
     */
    bindEvents() {
        const searchInput = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_INPUT);
        const clearSearch = document.querySelector(AnimeRecommendationApp.SELECTORS.CLEAR_SEARCH);

        if (searchInput) {
            searchInput.addEventListener('input', this.handleSearchInput.bind(this));
            searchInput.addEventListener('keydown', this.handleSearchKeydown.bind(this));
        }

        if (clearSearch) {
            clearSearch.addEventListener('click', this.handleClearSearch.bind(this));
        }

        document.addEventListener('click', this.handleGlobalClick.bind(this));

        const clearFavorites = document.querySelector(AnimeRecommendationApp.SELECTORS.CLEAR_FAVORITES);
        const getRecommendations = document.querySelector(AnimeRecommendationApp.SELECTORS.GET_RECOMMENDATIONS);

        if (clearFavorites) {
            clearFavorites.addEventListener('click', this.handleClearFavorites.bind(this));
        }

        if (getRecommendations) {
            getRecommendations.addEventListener('click', this.handleGetRecommendations.bind(this));
        }

        const modal = document.querySelector(AnimeRecommendationApp.SELECTORS.ANIME_MODAL);
        if (modal) {
            const closeModal = modal.querySelector('.close');
            if (closeModal) {
                closeModal.addEventListener('click', this.hideModal.bind(this));
            }
            modal.addEventListener('click', this.handleModalClick.bind(this));
        }

        document.addEventListener('keydown', this.handleGlobalKeydown.bind(this));
        document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
    }

    handleSearchInput(e) {
        const query = e.target.value.trim();
        const clearButton = document.querySelector(AnimeRecommendationApp.SELECTORS.CLEAR_SEARCH);
        
        if (query.length > 0) {
            if (clearButton) clearButton.style.display = 'block';
            if (query.length >= AnimeRecommendationApp.CONFIG.MIN_SEARCH_LENGTH) {
                this.debounceSearch(query);
            }
        } else {
            if (clearButton) clearButton.style.display = 'none';
            this.hideSearchResults();
        }
    }

    handleSearchKeydown(e) {
        if (e.key === 'Escape') {
            this.hideSearchResults();
            e.target.blur();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            const query = e.target.value.trim();
            if (query.length >= AnimeRecommendationApp.CONFIG.MIN_SEARCH_LENGTH) {
                this.searchAnime(query);
            }
        }
    }

    handleClearSearch() {
        const searchInput = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_INPUT);
        const clearButton = document.querySelector(AnimeRecommendationApp.SELECTORS.CLEAR_SEARCH);
        
        if (searchInput) {
            searchInput.value = '';
            searchInput.focus();
        }
        if (clearButton) clearButton.style.display = 'none';
        
        this.hideSearchResults();
        this.cancelPendingSearch();
    }

    handleGlobalClick(e) {
        if (!e.target.closest('.search-container')) {
            this.hideSearchResults();
        }
    }

    handleClearFavorites() {
        if (confirm('Are you sure you want to clear all favorites?')) {
            this.clearAllFavorites();
        }
    }

    handleGetRecommendations() {
        this.getRecommendations();
    }

    handleModalClick(e) {
        const modal = document.querySelector(AnimeRecommendationApp.SELECTORS.ANIME_MODAL);
        if (e.target === modal) {
            this.hideModal();
        }
    }

    handleGlobalKeydown(e) {
        if (e.key === 'Escape') {
            this.hideModal();
            this.hideSearchResults();
        } else if (e.key === '/' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            const searchInput = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_INPUT);
            if (searchInput) searchInput.focus();
        }
    }

    handleVisibilityChange() {
        if (document.hidden) {
            this.cancelPendingSearch();
        } else {
            this.cleanupExpiredCache();
        }
    }

    handleBeforeUnload() {
        this.cancelPendingSearch();
        this.saveCacheToStorage();
    }

    cancelPendingSearch() {
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = null;
        }
        
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
    }

    debounceSearch(query) {
        this.cancelPendingSearch();
        
        this.searchTimeout = setTimeout(() => {
            if (query.length >= AnimeRecommendationApp.CONFIG.MIN_SEARCH_LENGTH) {
                this.searchAnime(query);
            }
        }, AnimeRecommendationApp.CONFIG.SEARCH_DEBOUNCE_MS);
    }

    async searchAnime(query) {
        if (!query || query.trim().length < AnimeRecommendationApp.CONFIG.MIN_SEARCH_LENGTH) {
            this.hideSearchResults();
            return;
        }

        const trimmedQuery = query.trim();
        
        if (trimmedQuery.length > AnimeRecommendationApp.CONFIG.MAX_SEARCH_QUERY_LENGTH) {
            this.showToast('Search query too long. Please shorten your search.', 'error');
            return;
        }

        const sanitizedQuery = this.sanitizeSearchQuery(trimmedQuery);
        if (!sanitizedQuery) {
            this.showToast('Invalid search query. Please use only alphanumeric characters.', 'error');
            return;
        }

        try {
            this.showLoading();
            
            const cacheKey = `search_${sanitizedQuery.toLowerCase()}`;
            const cachedResult = this.getCachedResult(cacheKey);
            
            if (cachedResult) {
                this.displaySearchResults(cachedResult);
                this.hideLoading();
                return;
            }

            this.abortController = new AbortController();

            const response = await fetch(
                `${AnimeRecommendationApp.CONFIG.API_BASE_URL}/search-anime?query=${encodeURIComponent(sanitizedQuery)}`,
                {
                    signal: this.abortController.signal,
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (!this.validateSearchResponse(data)) {
                throw new Error('Invalid response format from server');
            }
            
            this.setCachedResult(cacheKey, data.results);
            this.displaySearchResults(data.results);
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Search request was cancelled');
                return;
            }
            
            console.error('Search error:', error);
            this.handleSearchError(error);
            this.hideSearchResults();
        } finally {
            this.hideLoading();
            this.abortController = null;
        }
    }

    sanitizeSearchQuery(query) {
        const sanitized = query.replace(/[<>\"'%;()&+]/g, '');
        
        if (sanitized.length === 0 || sanitized.length < AnimeRecommendationApp.CONFIG.MIN_SEARCH_LENGTH) {
            return null;
        }
        
        return sanitized;
    }

    validateSearchResponse(data) {
        return data && 
               typeof data === 'object' && 
               Array.isArray(data.results);
    }

    handleSearchError(error) {
        let message = 'Search failed. Please try again.';
        
        if (error.message.includes('Failed to fetch')) {
            message = 'Network error. Please check your connection and try again.';
        } else if (error.message.includes('HTTP 429')) {
            message = 'Too many requests. Please wait a moment before searching again.';
        } else if (error.message.includes('HTTP 5')) {
            message = 'Server error. Please try again later.';
        }
        
        this.showToast(message, 'error');
    }

    getCachedResult(key) {
        const cached = this.cache.get(key);
        if (!cached) return null;
        
        const now = Date.now();
        if (now - cached.timestamp > AnimeRecommendationApp.CONFIG.CACHE_EXPIRY_MS) {
            this.cache.delete(key);
            return null;
        }
        
        return cached.data;
    }

    setCachedResult(key, data) {
        this.cache.set(key, {
            data: data,
            timestamp: Date.now()
        });
        
        if (this.cache.size > 100) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }

    cleanupExpiredCache() {
        const now = Date.now();
        const expiredKeys = [];
        
        for (const [key, value] of this.cache.entries()) {
            if (now - value.timestamp > AnimeRecommendationApp.CONFIG.CACHE_EXPIRY_MS) {
                expiredKeys.push(key);
            }
        }
        
        expiredKeys.forEach(key => this.cache.delete(key));
    }

    saveCacheToStorage() {
        try {
            const recentCache = {};
            const now = Date.now();
            
            for (const [key, value] of this.cache.entries()) {
                if (now - value.timestamp < AnimeRecommendationApp.CONFIG.CACHE_EXPIRY_MS) {
                    recentCache[key] = value;
                }
            }
            
            localStorage.setItem(AnimeRecommendationApp.CONFIG.STORAGE_KEYS.CACHE, JSON.stringify(recentCache));
        } catch (error) {
            console.warn('Failed to save cache to localStorage:', error);
        }
    }

    loadCacheFromStorage() {
        try {
            const savedCache = localStorage.getItem(AnimeRecommendationApp.CONFIG.STORAGE_KEYS.CACHE);
            if (!savedCache) return;
            
            const parsedCache = JSON.parse(savedCache);
            const now = Date.now();
            
            for (const [key, value] of Object.entries(parsedCache)) {
                if (value && value.timestamp && now - value.timestamp < AnimeRecommendationApp.CONFIG.CACHE_EXPIRY_MS) {
                    this.cache.set(key, value);
                }
            }
        } catch (error) {
            console.warn('Failed to load cache from localStorage:', error);
        }
    }

    displaySearchResults(results) {
        const searchResults = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_RESULTS);
        if (!searchResults) return;
        
        if (!results || results.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
            searchResults.style.display = 'block';
            return;
        }

        const html = results.map(anime => `
            <div class="search-result-item" data-anime-id="${anime.anime_id}">
                ${anime.image_url ? `<img src="${anime.image_url}" alt="${this.escapeHtml(anime.name)}" onerror="this.style.display='none'">` : '<div class="placeholder-image"></div>'}
                <div class="search-result-info">
                    <div class="search-result-title">${this.escapeHtml(anime.name)}</div>
                    ${anime.title_english && anime.title_english !== anime.name ? `<div class="search-result-alt-title">${this.escapeHtml(anime.title_english)}</div>` : ''}
                    ${anime.title_japanese && anime.title_japanese !== anime.name ? `<div class="search-result-alt-title">${this.escapeHtml(anime.title_japanese)}</div>` : ''}
                    <div class="search-result-genres">${anime.genre && anime.genre.length > 0 ? anime.genre.slice(0, 3).join(', ') : 'No genres available'}</div>
                </div>
                <button class="add-favorite-btn" data-anime-name="${this.escapeHtml(anime.name)}" data-anime-id="${anime.anime_id}" ${this.favorites.has(anime.name) ? 'disabled' : ''}>
                    <i class="fas fa-heart"></i>
                    ${this.favorites.has(anime.name) ? 'Added' : 'Add'}
                </button>
            </div>
        `).join('');

        searchResults.innerHTML = html;
        searchResults.style.display = 'block';

        searchResults.querySelectorAll('.search-result-item').forEach(item => {
            const addButton = item.querySelector('.add-favorite-btn');
            if (addButton) {
                addButton.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const animeName = addButton.dataset.animeName;
                    const animeId = addButton.dataset.animeId;
                    this.addToFavorites(animeName, parseInt(animeId));
                });
            }

            item.addEventListener('click', (e) => {
                if (!e.target.closest('.add-favorite-btn')) {
                    const animeId = item.dataset.animeId;
                    const anime = results.find(a => a.anime_id == animeId);
                    if (anime) {
                        this.showAnimeDetails(anime);
                    }
                }
            });
        });
    }

    hideSearchResults() {
        const searchResults = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_RESULTS);
        if (searchResults) {
            searchResults.style.display = 'none';
        }
    }

    addToFavorites(animeName, animeId) {
        if (!animeName || typeof animeName !== 'string') {
            this.showToast('Invalid anime name!', 'error');
            return;
        }

        if (this.favorites.has(animeName)) {
            this.showToast('Anime already in favorites!', 'error');
            return;
        }

        this.favorites.add(animeName);
        this.saveFavoritesToStorage();
        this.updateFavoritesUI();
        this.showToast(`Added "${animeName}" to favorites!`, 'success');

        const searchResults = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_RESULTS);
        if (searchResults) {
            const button = searchResults.querySelector(`[data-anime-name="${this.escapeHtml(animeName)}"]`);
            if (button) {
                button.disabled = true;
                button.innerHTML = '<i class="fas fa-heart"></i> Added';
            }
        }
    }

    removeFromFavorites(animeName) {
        if (!this.favorites.has(animeName)) {
            return;
        }

        this.favorites.delete(animeName);
        this.saveFavoritesToStorage();
        this.updateFavoritesUI();
        this.showToast(`Removed "${animeName}" from favorites!`, 'success');

        const searchResults = document.querySelector(AnimeRecommendationApp.SELECTORS.SEARCH_RESULTS);
        if (searchResults && searchResults.style.display === 'block') {
            const button = searchResults.querySelector(`[data-anime-name="${this.escapeHtml(animeName)}"]`);
            if (button) {
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-heart"></i> Add';
            }
        }
    }

    clearAllFavorites() {
        this.favorites.clear();
        this.saveFavoritesToStorage();
        this.updateFavoritesUI();
        this.hideRecommendations();
        this.showToast('All favorites cleared!', 'success');
    }

    updateFavoritesUI() {
        const favoritesList = document.querySelector(AnimeRecommendationApp.SELECTORS.FAVORITES_LIST);
        const favoritesCount = document.querySelector(AnimeRecommendationApp.SELECTORS.FAVORITES_COUNT);
        const clearButton = document.querySelector(AnimeRecommendationApp.SELECTORS.CLEAR_FAVORITES);
        const getRecommendationsButton = document.querySelector(AnimeRecommendationApp.SELECTORS.GET_RECOMMENDATIONS);

        if (favoritesCount) {
            favoritesCount.textContent = this.favorites.size;
        }

        if (this.favorites.size === 0) {
            if (favoritesList) {
                favoritesList.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-heart-broken"></i>
                        <p>No favorites yet. Search and add some anime!</p>
                    </div>
                `;
            }
            if (clearButton) clearButton.style.display = 'none';
            if (getRecommendationsButton) getRecommendationsButton.style.display = 'none';
        } else {
            const html = Array.from(this.favorites).map(animeName => `
                <div class="favorite-item" data-anime-name="${this.escapeHtml(animeName)}">
                    <div class="favorite-header">
                        <div class="favorite-title">${this.escapeHtml(animeName)}</div>
                        <button class="remove-favorite-btn" data-anime-name="${this.escapeHtml(animeName)}">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `).join('');

            if (favoritesList) {
                favoritesList.innerHTML = html;

                favoritesList.querySelectorAll('.remove-favorite-btn').forEach(button => {
                    button.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const animeName = button.dataset.animeName;
                        this.removeFromFavorites(animeName);
                    });
                });

                favoritesList.querySelectorAll('.favorite-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const animeName = item.dataset.animeName;
                        this.searchAndShowDetails(animeName);
                    });
                });
            }

            if (clearButton) clearButton.style.display = 'block';
            if (getRecommendationsButton) getRecommendationsButton.style.display = 'block';
        }
    }

    async getRecommendations() {
        if (this.favorites.size === 0) {
            this.showToast('Add some favorites first!', 'error');
            return;
        }

        try {
            this.showLoading();

            const response = await fetch(`${AnimeRecommendationApp.CONFIG.API_BASE_URL}/recommend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    liked_anime: Array.from(this.favorites)
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (!data || !Array.isArray(data.recommendations)) {
                throw new Error('Invalid response format');
            }

            this.displayRecommendations(data.recommendations);
            
            const recommendationsSection = document.querySelector(AnimeRecommendationApp.SELECTORS.RECOMMENDATIONS_SECTION);
            if (recommendationsSection) {
                recommendationsSection.scrollIntoView({ behavior: 'smooth' });
            }

        } catch (error) {
            console.error('Recommendation error:', error);
            let message = 'Failed to get recommendations. Please try again.';
            
            if (error.message.includes('Failed to fetch')) {
                message = 'Network error. Please check your connection.';
            } else if (error.message.includes('HTTP 5')) {
                message = 'Server error. Please try again later.';
            }
            
            this.showToast(message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayRecommendations(recommendations) {
        const recommendationsSection = document.querySelector(AnimeRecommendationApp.SELECTORS.RECOMMENDATIONS_SECTION);
        const recommendationsList = document.querySelector(AnimeRecommendationApp.SELECTORS.RECOMMENDATIONS_LIST);

        if (!recommendationsSection || !recommendationsList) return;

        if (!recommendations || recommendations.length === 0) {
            recommendationsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>No recommendations found. Try adding more diverse anime to your favorites!</p>
                </div>
            `;
            recommendationsSection.style.display = 'block';
            return;
        }

        const html = recommendations.map(anime => `
            <div class="recommendation-item" data-anime='${JSON.stringify(anime).replace(/'/g, "&#39;")}'>
                ${anime.image_url ? `<img src="${anime.image_url}" alt="${this.escapeHtml(anime.name)}" class="recommendation-image" onerror="this.style.display='none'">` : '<div class="recommendation-image placeholder-image"></div>'}
                <div class="recommendation-content">
                    <div class="recommendation-title">${this.escapeHtml(anime.name)}</div>
                    <div class="recommendation-genres">
                        ${anime.genre && anime.genre.length > 0 ? 
                            anime.genre.slice(0, 4).map(genre => `<span class="genre-tag">${this.escapeHtml(genre)}</span>`).join('') : 
                            '<span class="genre-tag">Unknown</span>'
                        }
                    </div>
                    ${anime.synopsis ? `<div class="recommendation-synopsis">${this.escapeHtml(anime.synopsis)}</div>` : '<div class="recommendation-synopsis">No synopsis available.</div>'}
                </div>
            </div>
        `).join('');

        recommendationsList.innerHTML = html;
        recommendationsSection.style.display = 'block';

        recommendationsList.querySelectorAll('.recommendation-item').forEach(item => {
            item.addEventListener('click', () => {
                try {
                    const animeData = JSON.parse(item.dataset.anime);
                    this.showAnimeDetails(animeData);
                } catch (error) {
                    console.error('Failed to parse anime data:', error);
                }
            });
        });
    }

    hideRecommendations() {
        const recommendationsSection = document.querySelector(AnimeRecommendationApp.SELECTORS.RECOMMENDATIONS_SECTION);
        if (recommendationsSection) {
            recommendationsSection.style.display = 'none';
        }
    }

    async searchAndShowDetails(animeName) {
        try {
            this.showLoading();
            
            const response = await fetch(`${AnimeRecommendationApp.CONFIG.API_BASE_URL}/search-anime?query=${encodeURIComponent(animeName)}`);
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.results && data.results.length > 0) {
                const anime = data.results.find(a => a.name === animeName) || data.results[0];
                this.showAnimeDetails(anime);
            } else {
                this.showToast('Anime details not found.', 'error');
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showToast('Failed to load anime details.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    showAnimeDetails(anime) {
        if (!anime) return;

        const modal = document.querySelector(AnimeRecommendationApp.SELECTORS.ANIME_MODAL);
        const animeDetails = document.querySelector(AnimeRecommendationApp.SELECTORS.ANIME_DETAILS);

        if (!modal || !animeDetails) return;

        const html = `
            ${anime.image_url ? `<img src="${anime.image_url}" alt="${this.escapeHtml(anime.name)}" class="anime-detail-image" onerror="this.style.display='none'">` : ''}
            <div class="anime-detail-title">${this.escapeHtml(anime.name)}</div>
            ${anime.title_english && anime.title_english !== anime.name ? `<div class="anime-detail-subtitle">${this.escapeHtml(anime.title_english)}</div>` : ''}
            ${anime.title_japanese && anime.title_japanese !== anime.name ? `<div class="anime-detail-subtitle">${this.escapeHtml(anime.title_japanese)}</div>` : ''}
            ${anime.synopsis ? `<div class="anime-detail-synopsis">${this.escapeHtml(anime.synopsis)}</div>` : '<div class="anime-detail-synopsis">No synopsis available.</div>'}
            <div class="anime-detail-genres">
                ${anime.genre && anime.genre.length > 0 ? 
                    anime.genre.map(genre => `<span class="genre-tag">${this.escapeHtml(genre)}</span>`).join('') : 
                    '<span class="genre-tag">No genres available</span>'
                }
            </div>
        `;

        animeDetails.innerHTML = html;
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    hideModal() {
        const modal = document.querySelector(AnimeRecommendationApp.SELECTORS.ANIME_MODAL);
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    }

    showLoading() {
        const loadingSpinner = document.querySelector(AnimeRecommendationApp.SELECTORS.LOADING_SPINNER);
        if (loadingSpinner) {
            loadingSpinner.style.display = 'flex';
        }
    }

    hideLoading() {
        const loadingSpinner = document.querySelector(AnimeRecommendationApp.SELECTORS.LOADING_SPINNER);
        if (loadingSpinner) {
            loadingSpinner.style.display = 'none';
        }
    }

    showToast(message, type = 'success') {
        const toastContainer = document.querySelector(AnimeRecommendationApp.SELECTORS.TOAST_CONTAINER);
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        toastContainer.appendChild(toast);

        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, AnimeRecommendationApp.CONFIG.TOAST_DURATION_MS);
    }

    saveFavoritesToStorage() {
        try {
            localStorage.setItem(
                AnimeRecommendationApp.CONFIG.STORAGE_KEYS.FAVORITES, 
                JSON.stringify(Array.from(this.favorites))
            );
        } catch (error) {
            console.warn('Failed to save favorites to localStorage:', error);
        }
    }

    loadFavoritesFromStorage() {
        try {
            const saved = localStorage.getItem(AnimeRecommendationApp.CONFIG.STORAGE_KEYS.FAVORITES);
            if (saved) {
                const favoritesArray = JSON.parse(saved);
                if (Array.isArray(favoritesArray)) {
                    this.favorites = new Set(favoritesArray);
                }
            }
        } catch (error) {
            console.warn('Failed to load favorites from localStorage:', error);
            this.favorites = new Set();
        }
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Application initialization with error handling
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.app = new AnimeRecommendationApp();
    } catch (error) {
        console.error('Failed to initialize application:', error);
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #f44336; color: white; padding: 15px; border-radius: 5px; z-index: 10000;';
        errorDiv.textContent = 'Application failed to load. Please refresh the page.';
        document.body.appendChild(errorDiv);
    }
});

// Service Worker registration for better caching
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('Service Worker registered successfully:', registration);
            })
            .catch(registrationError => {
                console.log('Service Worker registration failed:', registrationError);
            });
    });
}