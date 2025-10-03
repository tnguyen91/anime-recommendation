const APP_CONFIG = {
  apiBaseUrl: window.API_BASE || 'https://animereco-api-725392014501.us-west1.run.app',
  localStorageKey: 'animeDiscovery_userFavorites',
  searchSettings: {
    maxSuggestions: 12,
    debounceDelayMs: 300,
    minQueryLength: 2
  },
  recommendationSettings: {
    maxRecommendations: 10
  }
};

const domElements = {
  searchForm: document.querySelector('#search-form'),
  searchInput: document.querySelector('#search-input'),
  searchSuggestionsList: document.querySelector('#search-suggestions'),
  
  favoritesContainer: document.querySelector('#favorite-list'),
  
  recommendationsButton: document.querySelector('#recommend-button'),
  recommendationsGrid: document.querySelector('#recommendation-list'),
  recommendationsSection: document.querySelector('#recommendations-section'),
  recommendationsEmptyState: document.querySelector('#recommendations-empty'),
  
  detailsModal: document.querySelector('#details-modal'),
  detailsModalCloseButton: document.querySelector('#details-modal-close'),
  detailsModalContent: document.querySelector('#details-modal-body')
};

const applicationState = {
  userFavorites: [],
  searchResults: [],
  currentRecommendations: [],
  activeSearchRequest: null,
  ui: {
    activeSearchIndex: -1,
    isSearching: false,
    isLoadingRecommendations: false
  }
};

class AnimeApiService {
  static async makeRequest(url, options = {}) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') {
        throw error;
      }
      console.error('API request failed:', error);
      throw new Error(`Network request failed: ${error.message}`);
    }
  }

  static async searchAnime(query, abortSignal) {
    if (!query?.trim()) return [];
    
    const searchUrl = `${APP_CONFIG.apiBaseUrl}/search-anime?query=${encodeURIComponent(query.trim())}`;
    
    try {
      const data = await this.makeRequest(searchUrl, { signal: abortSignal });
      return this.normalizeAnimeData(data);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      console.warn('Anime search failed:', error);
      return [];
    }
  }

  static async fetchRecommendations(favoriteAnimeList) {
    if (!favoriteAnimeList?.length) {
      throw new Error('No favorite anime provided for recommendations');
    }
    
    const animeNames = favoriteAnimeList.map(anime => anime.name).filter(Boolean);
    
    try {
      const response = await this.makeRequest(`${APP_CONFIG.apiBaseUrl}/recommend`, {
        method: 'POST',
        body: JSON.stringify({ liked_anime: animeNames })
      });
      
      return this.normalizeAnimeData(response);
    } catch (error) {
      throw new Error(`Failed to fetch recommendations: ${error.message}`);
    }
  }

  static normalizeAnimeData(apiResponse) {
    if (!apiResponse) return [];
    
    const items = Array.isArray(apiResponse) 
      ? apiResponse 
      : (apiResponse.results || apiResponse.recommendations || apiResponse.data || []);
      
    if (!Array.isArray(items)) return [];
    
    return items.map(item => ({
      id: item.anime_id || item.id || item.mal_id || item.slug || item._id || null,
      name: (item.name || item.title || item.title_english || item.title_japanese || 'Unknown Title').trim(),
      titleEnglish: item.title_english || null,
      titleJapanese: item.title_japanese || null,
      imageUrl: (item.image_url || item.image || item.images?.jpg?.image_url || item.picture || '').trim(),
      genres: this.extractGenres(item),
      synopsis: (item.synopsis || item.description || item.overview || '').trim()
    })).filter(anime => anime.name !== 'Unknown Title');
  }

  static extractGenres(item) {
    const genres = Array.isArray(item.genres) ? item.genres 
                 : Array.isArray(item.genre) ? item.genre 
                 : Array.isArray(item.tags) ? item.tags 
                 : [];
                 
    return genres.filter(genre => genre && typeof genre === 'string');
  }
}

class FavoritesService {
  static loadUserFavorites() {
    if (typeof localStorage === 'undefined') {
      console.warn('localStorage not available');
      return;
    }
    
    try {
      const storedFavorites = localStorage.getItem(APP_CONFIG.localStorageKey);
      const parsedFavorites = storedFavorites ? JSON.parse(storedFavorites) : [];
      
      if (!Array.isArray(parsedFavorites)) {
        throw new Error('Invalid favorites data format');
      }
      
      applicationState.userFavorites = parsedFavorites
        .filter(item => item && typeof item.name === 'string')
        .map(item => ({
          id: item.id || null,
          name: item.name.trim()
        }));
        
    } catch (error) {
      console.error('Failed to load favorites:', error);
      applicationState.userFavorites = [];
    }
  }

  static saveUserFavorites() {
    if (typeof localStorage === 'undefined') return;
    
    try {
      localStorage.setItem(APP_CONFIG.localStorageKey, JSON.stringify(applicationState.userFavorites));
    } catch (error) {
      console.error('Failed to save favorites:', error);
    }
  }

  static isAnimeFavorited(anime) {
    return applicationState.userFavorites.some(favorite => 
      (favorite.id && anime.id && favorite.id === anime.id) || 
      favorite.name === anime.name
    );
  }

  static addAnimeToFavorites(anime) {
    if (!anime?.name) {
      console.warn('Invalid anime object for favorites');
      return false;
    }
    
    if (this.isAnimeFavorited(anime)) {
      console.info('Anime already in favorites:', anime.name);
      return false;
    }

    const favoriteItem = {
      id: anime.id || null,
      name: anime.name.trim()
    };
    
    applicationState.userFavorites.push(favoriteItem);
    this.saveUserFavorites();
    FavoritesUI.renderFavoritesList();
    RecommendationsUI.updateRecommendButtonState();
    return true;
  }

  static removeAnimeFromFavorites(animeName) {
    if (!animeName) return false;
    
    const initialLength = applicationState.userFavorites.length;
    applicationState.userFavorites = applicationState.userFavorites.filter(
      favorite => favorite.name !== animeName
    );
    
    if (applicationState.userFavorites.length < initialLength) {
      this.saveUserFavorites();
      FavoritesUI.renderFavoritesList();
      RecommendationsUI.updateRecommendButtonState();
      return true;
    }
    
    return false;
  }

  static toggleAnimeFavoriteStatus(anime) {
    return this.isAnimeFavorited(anime) 
      ? this.removeAnimeFromFavorites(anime.name)
      : this.addAnimeToFavorites(anime);
  }
}

class SearchUI {
  static debounceTimer = null;

  static initialize() {
    this.bindEventListeners();
  }

  static bindEventListeners() {
    domElements.searchInput.addEventListener('input', (event) => {
      this.handleSearchInput(event.target.value);
    });

    domElements.searchInput.addEventListener('focus', () => {
      if (applicationState.searchResults.length > 0) {
        this.showSuggestions();
      }
    });

    domElements.searchForm.addEventListener('submit', (event) => {
      event.preventDefault();
      if (applicationState.searchResults.length > 0) {
        this.selectFirstSuggestion();
      }
    });

    document.addEventListener('click', (event) => {
      if (!domElements.searchForm.contains(event.target)) {
        this.hideSuggestions();
      }
    });
  }

  static handleSearchInput(query) {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    if (query.trim().length < APP_CONFIG.searchSettings.minQueryLength) {
      this.hideSuggestions();
      return;
    }

    this.debounceTimer = setTimeout(() => {
      this.performSearch(query);
    }, APP_CONFIG.searchSettings.debounceDelayMs);
  }

  static async performSearch(query) {
    try {
      if (applicationState.activeSearchRequest) {
        applicationState.activeSearchRequest.abort();
      }

      const searchController = new AbortController();
      applicationState.activeSearchRequest = searchController;
      applicationState.ui.isSearching = true;

      const searchResults = await AnimeApiService.searchAnime(query, searchController.signal);
      
      if (applicationState.activeSearchRequest === searchController) {
        applicationState.searchResults = searchResults.slice(0, APP_CONFIG.searchSettings.maxSuggestions);
        applicationState.ui.activeSearchIndex = -1;
        this.renderSearchSuggestions();
        this.showSuggestions();
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Search failed:', error);
        this.hideSuggestions();
      }
    } finally {
      applicationState.ui.isSearching = false;
    }
  }

  static renderSearchSuggestions() {
    domElements.searchSuggestionsList.innerHTML = '';

    if (!applicationState.searchResults.length) {
      this.hideSuggestions();
      return;
    }

    const suggestionTemplate = document.querySelector('#suggestion-item-template');
    
    applicationState.searchResults.forEach((anime, index) => {
      const suggestionElement = suggestionTemplate.content.cloneNode(true);
      const listItem = suggestionElement.querySelector('.search-suggestions__item');
      const image = suggestionElement.querySelector('.search-suggestions__image');
      const name = suggestionElement.querySelector('.search-suggestions__name');
      const subtitle = suggestionElement.querySelector('.search-suggestions__subtitle');
      const toggleButton = suggestionElement.querySelector('.search-suggestions__add-button');

      image.src = anime.imageUrl || '/assets/placeholder.svg';
      image.alt = `${anime.name} poster`;
      name.textContent = anime.name;
      subtitle.textContent = anime.genres.slice(0, 3).join(', ') || 'Anime';

      const isCurrentlyFavorited = FavoritesService.isAnimeFavorited(anime);
      toggleButton.textContent = isCurrentlyFavorited ? 'Remove' : 'Add';
      toggleButton.className = `search-suggestions__add-button ${isCurrentlyFavorited ? 'remove' : 'add'}`;

      listItem.addEventListener('click', (event) => {
        if (event.target === toggleButton) return;
        AnimeDetailsModal.showAnimeDetails(anime);
      });

      toggleButton.addEventListener('click', (event) => {
        event.stopPropagation();
        this.handleFavoriteToggle(anime, toggleButton);
      });

      listItem.setAttribute('data-index', index);
      domElements.searchSuggestionsList.appendChild(suggestionElement);
    });
  }

  static handleFavoriteToggle(anime, buttonElement) {
    const wasToggled = FavoritesService.toggleAnimeFavoriteStatus(anime);
    
    if (wasToggled) {
      const isNowFavorited = FavoritesService.isAnimeFavorited(anime);
      buttonElement.textContent = isNowFavorited ? 'Remove' : 'Add';
      buttonElement.className = `search-suggestions__add-button ${isNowFavorited ? 'remove' : 'add'}`;
    }
  }

  static selectFirstSuggestion() {
    if (applicationState.searchResults.length > 0) {
      const firstAnime = applicationState.searchResults[0];
      if (!FavoritesService.isAnimeFavorited(firstAnime)) {
        FavoritesService.addAnimeToFavorites(firstAnime);
      }
      this.hideSuggestions();
      domElements.searchInput.value = '';
    }
  }

  static showSuggestions() {
    domElements.searchSuggestionsList.hidden = false;
    domElements.searchInput.setAttribute('aria-expanded', 'true');
  }

  static hideSuggestions() {
    domElements.searchSuggestionsList.hidden = true;
    domElements.searchInput.setAttribute('aria-expanded', 'false');
    applicationState.ui.activeSearchIndex = -1;
  }
}

class FavoritesUI {
  static renderFavoritesList() {
    domElements.favoritesContainer.innerHTML = '';
    
    if (!applicationState.userFavorites.length) {
      domElements.favoritesContainer.dataset.empty = 'true';
      return;
    }

    domElements.favoritesContainer.dataset.empty = 'false';
    
    applicationState.userFavorites.forEach(favorite => {
      const favoriteChip = this.createFavoriteChip(favorite);
      if (favoriteChip) {
        domElements.favoritesContainer.appendChild(favoriteChip);
      }
    });
  }

  static createFavoriteChip(favorite) {
    try {
      const chipElement = document.createElement('button');
      chipElement.type = 'button';
      chipElement.className = 'favorites-list__chip';
      chipElement.textContent = favorite.name;
      chipElement.setAttribute('role', 'listitem');
      chipElement.setAttribute('aria-label', `Remove ${favorite.name} from favorites`);
      chipElement.setAttribute('title', 'Click to remove from favorites');
      
      chipElement.addEventListener('click', () => {
        FavoritesService.removeAnimeFromFavorites(favorite.name);
      });
      
      chipElement.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          FavoritesService.removeAnimeFromFavorites(favorite.name);
        }
      });
      
      return chipElement;
    } catch (error) {
      console.error('Failed to create favorite chip:', error);
      return null;
    }
  }
}

class RecommendationsUI {
  static initialize() {
    this.bindEventListeners();
  }

  static bindEventListeners() {
    domElements.recommendationsButton.addEventListener('click', () => {
      this.handleRecommendationRequest();
    });
  }

  static async handleRecommendationRequest() {
    if (applicationState.userFavorites.length === 0) {
      console.warn('No favorites selected for recommendations');
      return;
    }

    try {
      this.setLoadingState(true);
      
      const recommendations = await AnimeApiService.fetchRecommendations(applicationState.userFavorites);
      
      applicationState.currentRecommendations = recommendations.slice(0, APP_CONFIG.recommendationSettings.maxRecommendations);
      this.renderRecommendations();
      this.scrollToRecommendationsSection();
      
    } catch (error) {
      console.error('Failed to fetch recommendations:', error);
      this.showErrorState('Failed to fetch recommendations. Please try again.');
    } finally {
      this.setLoadingState(false);
    }
  }

  static renderRecommendations() {
    domElements.recommendationsGrid.innerHTML = '';
    
    if (!applicationState.currentRecommendations.length) {
      domElements.recommendationsEmptyState.hidden = false;
      domElements.recommendationsGrid.setAttribute('aria-busy', 'false');
      return;
    }

    domElements.recommendationsEmptyState.hidden = true;
    
    applicationState.currentRecommendations.forEach((anime, index) => {
      const recommendationCard = this.createRecommendationCard(anime, index);
      domElements.recommendationsGrid.appendChild(recommendationCard);
    });

    domElements.recommendationsGrid.setAttribute('aria-busy', 'false');
  }

  static createRecommendationCard(anime, index) {
    const cardElement = document.createElement('div');
    cardElement.className = 'recommendation-card';
    cardElement.setAttribute('role', 'button');
    cardElement.setAttribute('tabindex', '0');
    cardElement.setAttribute('aria-label', `View details for ${anime.name}`);
    cardElement.setAttribute('data-index', index);

    const posterImage = document.createElement('img');
    posterImage.className = 'recommendation-card__poster';
    posterImage.src = anime.imageUrl || '/assets/placeholder.svg';
    posterImage.alt = `${anime.name} poster`;
    posterImage.loading = 'lazy';

    posterImage.onerror = () => {
      posterImage.src = '/assets/placeholder.svg';
    };

    const overlay = document.createElement('div');
    overlay.className = 'recommendation-card__overlay';
    overlay.textContent = anime.name;

    cardElement.appendChild(posterImage);
    cardElement.appendChild(overlay);

    cardElement.addEventListener('click', (event) => {
      event.stopPropagation();
      AnimeDetailsModal.showAnimeDetails(anime);
    });

    cardElement.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        AnimeDetailsModal.showAnimeDetails(anime);
      }
    });

    return cardElement;
  }

  static scrollToRecommendationsSection() {
    if (domElements.recommendationsSection) {
      domElements.recommendationsSection.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
      });
    }
  }

  static setLoadingState(isLoading) {
    domElements.recommendationsButton.dataset.loading = String(isLoading);
    applicationState.ui.isLoadingRecommendations = isLoading;
    
    if (isLoading) {
      domElements.recommendationsButton.classList.add('loading');
      domElements.recommendationsButton.disabled = true;
      domElements.recommendationsGrid.setAttribute('aria-busy', 'true');
    } else {
      domElements.recommendationsButton.classList.remove('loading');
      domElements.recommendationsButton.disabled = applicationState.userFavorites.length === 0;
      domElements.recommendationsGrid.setAttribute('aria-busy', 'false');
    }
  }

  static updateRecommendButtonState() {
    if (applicationState.ui.isLoadingRecommendations) return;
    
    const hasEnoughFavorites = applicationState.userFavorites.length > 0;
    domElements.recommendationsButton.disabled = !hasEnoughFavorites;
    domElements.recommendationsButton.setAttribute('aria-disabled', String(!hasEnoughFavorites));
  }

  static showErrorState(errorMessage) {
    domElements.recommendationsGrid.innerHTML = `
      <div class="error-message" style="
        grid-column: 1 / -1;
        text-align: center;
        padding: 40px 20px;
        color: var(--color-danger);
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        margin: 20px 0;
      ">
        <h3 style="margin: 0 0 8px; font-size: 18px;">Error</h3>
        <p style="margin: 0;">${errorMessage}</p>
      </div>
    `;
    
    domElements.recommendationsEmptyState.hidden = true;
  }
}

class AnimeDetailsModal {
  static initialize() {
    this.bindEventListeners();
  }

  static bindEventListeners() {
    if (domElements.detailsModalCloseButton) {
      domElements.detailsModalCloseButton.addEventListener('click', () => {
        this.closeModal();
      });
    }

    if (domElements.detailsModal) {
      domElements.detailsModal.addEventListener('click', (event) => {
        if (event.target === domElements.detailsModal) {
          this.closeModal();
        }
      });

      domElements.detailsModal.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
          this.closeModal();
        }
      });
    }
  }

  static showAnimeDetails(anime) {
    if (!domElements.detailsModal || !domElements.detailsModalContent) {
      console.warn('Modal elements not found');
      return;
    }

    this.renderModalContent(anime);
    domElements.detailsModal.showModal();
    domElements.detailsModalCloseButton.focus();
    document.body.style.overflow = 'hidden';
  }

  static closeModal() {
    if (domElements.detailsModal) {
      domElements.detailsModal.close();
      document.body.style.overflow = '';
      domElements.detailsModalContent.innerHTML = '';
    }
  }

  static renderModalContent(anime) {
    const layoutContainer = document.createElement('div');
    layoutContainer.className = 'details-modal__layout';

    if (anime.imageUrl) {
      const posterImage = document.createElement('img');
      posterImage.className = 'details-modal__image';
      posterImage.src = anime.imageUrl;
      posterImage.alt = `${anime.name} poster`;
      posterImage.loading = 'lazy';
      
      posterImage.onerror = () => {
        posterImage.src = '/assets/placeholder.svg';
      };

      layoutContainer.appendChild(posterImage);
    }

    const metadataContainer = document.createElement('div');
    metadataContainer.className = 'details-modal__meta';

    const titleElement = document.createElement('h3');
    titleElement.textContent = anime.name;
    metadataContainer.appendChild(titleElement);

    if (anime.titleEnglish && anime.titleEnglish !== anime.name) {
      const englishTitleElement = document.createElement('p');
      englishTitleElement.className = 'text-muted';
      englishTitleElement.innerHTML = `<strong>English:</strong> ${anime.titleEnglish}`;
      metadataContainer.appendChild(englishTitleElement);
    }

    if (anime.titleJapanese && anime.titleJapanese !== anime.name) {
      const japaneseTitleElement = document.createElement('p');
      japaneseTitleElement.className = 'text-muted';
      japaneseTitleElement.innerHTML = `<strong>Japanese:</strong> ${anime.titleJapanese}`;
      metadataContainer.appendChild(japaneseTitleElement);
    }

    if (anime.genres && anime.genres.length > 0) {
      const genresElement = document.createElement('p');
      genresElement.className = 'text-muted';
      genresElement.innerHTML = `<strong>Genres:</strong> ${anime.genres.join(', ')}`;
      metadataContainer.appendChild(genresElement);
    }

    const favoriteSection = document.createElement('div');
    favoriteSection.style.marginTop = '16px';
    
    const favoriteButton = document.createElement('button');
    favoriteButton.className = 'button-primary';
    favoriteButton.style.marginRight = '8px';
    
    const isCurrentlyFavorited = FavoritesService.isAnimeFavorited(anime);
    favoriteButton.textContent = isCurrentlyFavorited ? 'Remove from Favorites' : 'Add to Favorites';
    favoriteButton.style.background = isCurrentlyFavorited ? '#ef4444' : 'var(--color-accent-primary)';
    
    favoriteButton.addEventListener('click', () => {
      const wasToggled = FavoritesService.toggleAnimeFavoriteStatus(anime);
      if (wasToggled) {
        const isNowFavorited = FavoritesService.isAnimeFavorited(anime);
        favoriteButton.textContent = isNowFavorited ? 'Remove from Favorites' : 'Add to Favorites';
        favoriteButton.style.background = isNowFavorited ? '#ef4444' : 'var(--color-accent-primary)';
      }
    });
    
    favoriteSection.appendChild(favoriteButton);
    metadataContainer.appendChild(favoriteSection);

    if (anime.synopsis) {
      const synopsisHeader = document.createElement('h4');
      synopsisHeader.textContent = 'Synopsis';
      synopsisHeader.style.marginTop = '20px';
      synopsisHeader.style.marginBottom = '8px';
      metadataContainer.appendChild(synopsisHeader);

      const synopsisContent = document.createElement('p');
      synopsisContent.textContent = anime.synopsis;
      synopsisContent.style.lineHeight = '1.6';
      metadataContainer.appendChild(synopsisContent);
    }

    layoutContainer.appendChild(metadataContainer);
    domElements.detailsModalContent.appendChild(layoutContainer);
  }
}

class AnimeDiscoveryApp {
  static initialize() {
    try {
      if (!this.validateDomElements()) {
        throw new Error('Essential DOM elements not found');
      }
      
      FavoritesService.loadUserFavorites();
      
      SearchUI.initialize();
      RecommendationsUI.initialize();
      AnimeDetailsModal.initialize();
      
      FavoritesUI.renderFavoritesList();
      domElements.recommendationsEmptyState.hidden = false;
      RecommendationsUI.updateRecommendButtonState();
      
      console.info('Anime Discovery App initialized successfully');
    } catch (error) {
      console.error('Failed to initialize application:', error);
    }
  }

  static validateDomElements() {
    const requiredElements = [
      'searchForm', 'searchInput', 'searchSuggestionsList',
      'favoritesContainer', 'recommendationsButton', 'recommendationsGrid'
    ];
    
    return requiredElements.every(elementKey => {
      const element = domElements[elementKey];
      if (!element) {
        console.error(`Required DOM element not found: ${elementKey}`);
        return false;
      }
      return true;
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    AnimeDiscoveryApp.initialize();
  });
} else {
  AnimeDiscoveryApp.initialize();
}