/**
 * Anime Recommendation Frontend Application
 *
 * Single-page application for browsing and getting anime recommendations.
 * Communicates with the FastAPI backend to search anime, manage favorites,
 * and get personalized recommendations powered by a Restricted Boltzmann Machine.
 *
 * Features:
 * - Debounced anime search with autocomplete suggestions
 * - Local storage persistence for favorites (works without authentication)
 * - Personalized recommendations based on liked anime
 * - Infinite scroll with exclusion of previously shown recommendations
 * - Accessible modal dialogs with focus trapping
 * - Keyboard shortcuts (Ctrl+K to search, Ctrl+Enter to get recommendations)
 */

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = window.APP_CONFIG?.API_BASE_URL || 'http://localhost:8000';
const IS_DEV = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const DISPLAY_COUNT = 10;

/** Development-only logger that silences logs in production */
const logger = {
  error: (...args) => console.error(...args),
  warn: (...args) => IS_DEV && console.warn(...args),
  info: (...args) => IS_DEV && console.info(...args),
  log: (...args) => IS_DEV && console.log(...args)
};

// =============================================================================
// Utility Functions
// =============================================================================

/** Query selector shorthand */
const $ = (sel, root=document) => root.querySelector(sel);
/** Query selector all shorthand (returns array) */
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));

/**
 * Maps API response to normalized anime item format.
 * @param {Object} x - Raw API response item
 * @returns {Object} Normalized anime item
 */
function mapAnimeResult(x) {
  return {
    id: x.anime_id,
    name: x.name,
    title: x.name,
    title_english: x.title_english ?? null,
    title_japanese: x.title_japanese ?? null,
    image: x.image_url ?? './assets/placeholder.svg',
    year: x.year ?? null,
    genres: x.genre ?? [],
    synopsis: x.synopsis ?? null,
  };
}

/**
 * Smoothly scrolls to a target element, accounting for fixed header.
 * @param {string} targetId - ID of the target element (without #)
 * @param {number} extraOffset - Additional offset in pixels (default: 0)
 */
function scrollToSection(targetId, extraOffset = 0) {
  const targetEl = document.getElementById(targetId);
  if (targetEl) {
    const header = document.querySelector('.site-header');
    const headerHeight = header ? header.offsetHeight : 0;
    const targetPosition = targetEl.getBoundingClientRect().top + window.scrollY - headerHeight - extraOffset;
    window.scrollTo({ top: targetPosition, behavior: 'smooth' });
  }
}

// =============================================================================
// Application State
// =============================================================================

const state = {
  favorites: loadFavorites(),
  recommendations: [],
  shownIds: new Set(),  // Track shown IDs to exclude from future requests
  lastQuery: '',
  aborter: null,
  scrolling: false,
};

/** Cached DOM element references for performance */
const els = {
  suggestions: $("#suggestions"),
  searchInput: $("#search-input"),
  clearSearch: $("#clear-search"),
  favoritesList: $("#favorites-list"),
  getRecsBtn: $("#get-recommendations"),
  grid: $("#recommendations-grid"),
  modal: $("#modal"),
  modalContent: $("#modal-content"),
};

// =============================================================================
// Local Storage (Favorites Persistence)
// =============================================================================

/** Load favorites from localStorage with fallback to empty object */
function loadFavorites(){
  try {
    const raw = localStorage.getItem("favorites");
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch(_){ return {}; }
}

/** Persist current favorites to localStorage */
function persistFavorites(){
  localStorage.setItem("favorites", JSON.stringify(state.favorites));
  updateCTA();
}

// =============================================================================
// Network & Performance Utilities
// =============================================================================

/**
 * Creates a debounced version of a function.
 * @param {Function} fn - Function to debounce
 * @param {number} ms - Delay in milliseconds
 */
function debounce(fn, ms){
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  }
}

/**
 * Fetch wrapper with automatic timeout.
 * @param {string} resource - URL to fetch
 * @param {Object} options - Fetch options
 * @param {number} ms - Timeout in milliseconds (default: 15000)
 */
function timeoutFetch(resource, options={}, ms=15000){
  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), ms);
    fetch(resource, {...options, signal: controller.signal})
      .then(res => resolve(res))
      .catch(err => reject(err))
      .finally(() => clearTimeout(t));
  });
}

// =============================================================================
// UI Helpers
// =============================================================================

/** Set loading state on an element with ARIA attributes */
function setLoading(el, isLoading){
  if(isLoading){
    el.setAttribute('data-loading', 'true');
    el.setAttribute('aria-busy', 'true');
  } else {
    el.removeAttribute('data-loading');
    el.removeAttribute('aria-busy');
  }
}

/**
 * Create DOM element with attributes and children.
 * Supports class, dataset, event handlers (onX), and standard attributes.
 */
function createEl(tag, attrs={}, children=[]){
  const el = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)){
    if(k === 'class') el.className = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  for (const child of [].concat(children)){
    if (child == null) continue;
    el.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return el;
}

// =============================================================================
// Modal & Accessibility
// =============================================================================

let focusableElements = [];
let firstFocusable = null;
let lastFocusable = null;

/** Trap focus within modal for accessibility (keyboard navigation) */
function trapFocus(e) {
  if (e.key !== 'Tab') return;
  if (e.shiftKey) {
    if (document.activeElement === firstFocusable) {
      e.preventDefault();
      lastFocusable?.focus();
    }
  } else {
    if (document.activeElement === lastFocusable) {
      e.preventDefault();
      firstFocusable?.focus();
    }
  }
}

/** Update focus trap boundaries when modal content changes */
function updateFocusTrap() {
  const focusableSelectors = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
  focusableElements = Array.from(els.modal.querySelectorAll(focusableSelectors))
    .filter(el => !el.hasAttribute('disabled'));
  firstFocusable = focusableElements[0];
  lastFocusable = focusableElements[focusableElements.length - 1];
}

// =============================================================================
// Rendering Functions
// =============================================================================

/** Render the favorites list in the sidebar */
function renderFavorites(){
  els.favoritesList.innerHTML = '';
  const entries = Object.values(state.favorites);
  if (!entries.length){
    els.favoritesList.innerHTML = '<li class="muted">No favorites yet. Search for anime above and click "Add" to build your list!</li>';
    return;
  }
  for (const fav of entries){
    const li = createEl('li', {tabindex: '0', role: 'button', 'aria-label': `Remove ${fav.title} from favorites`}, fav.title);
    li.addEventListener('click', () => toggleFavorite(fav));
    li.addEventListener('keydown', (e) => { if(e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleFavorite(fav);} });
    els.favoritesList.appendChild(li);
  }
}

/** Enable/disable the Get Recommendations button based on favorites count */
function updateCTA(){
  const hasFavs = Object.keys(state.favorites).length > 0;
  els.getRecsBtn.disabled = !hasFavs;
}

/** Render search suggestion cards */
function renderSuggestions(items){
  els.suggestions.innerHTML = '';
  if (!items || !items.length) return;
  for (const item of items){
    const key = item.name || item.title;
    const isFav = !!state.favorites[key];
    const card = createEl('div', {class: 'suggestion-card', role:'option', tabindex:'0', 'aria-label': `${item.title}`});
    const img = createEl('img', {src: item.image || './assets/placeholder.svg', alt: `${item.title} cover`});
    const meta = createEl('div', {class:'suggestion-meta'});
    const title = createEl('div', {class:'suggestion-title'}, item.title);
    const btn = createEl('button', {class:`add-btn ${isFav?'remove':'add'}`, 'aria-label': isFav ? 'Remove from favorites' : 'Add to favorites'}, isFav ? 'Remove' : 'Add');

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleFavorite(item);
      const nowFav = !!state.favorites[key];
      btn.textContent = nowFav ? 'Remove' : 'Add';
      btn.classList.toggle('remove', nowFav);
      btn.classList.toggle('add', !nowFav);
    });

    meta.appendChild(title);
    card.appendChild(img);
    card.appendChild(meta);
    card.appendChild(btn);
    card.addEventListener('click', () => showDetails(item));
    card.addEventListener('keydown', (e) => { if(e.key==='Enter') showDetails(item); });
    els.suggestions.appendChild(card);
  }
}

/** Render recommendation cards in the main grid */
function renderRecommendations(items){
  els.grid.innerHTML = '';
  if (!items || !items.length){
    els.grid.innerHTML = '<p class="muted">No recommendations yet. Add at least one favorite anime above, then click "Get Recommendations" to discover new shows you\'ll love!</p>';
    return;
  }
  state.recommendations = items.slice(0, 10);
  state.recommendations.forEach((item, index) => {
    const key = item.name || item.title;
    const isFav = !!state.favorites[key];
    const card = createEl('article', {class:'card-lg', tabindex:'0', 'aria-label': `${item.title}`});
    const img = createEl('img', {src: item.image || './assets/placeholder.svg', alt: `${item.title} cover`});
    const inner = createEl('div', {class:'p'});
    const title = createEl('h3', {}, item.title);
    const sub = createEl('div', {class:'sub'}, item.year ? String(item.year) : (item.genres && item.genres.length ? item.genres.slice(0,2).join(' • ') : ''));
    
    const favBtn = createEl('button', {
      class: `rec-btn rec-btn-fav-corner ${isFav ? 'is-fav' : ''}`,
      'aria-label': isFav ? 'Remove from favorites' : 'Add to favorites',
      title: isFav ? 'Remove from favorites' : 'Add to favorites'
    }, '♥');
    
    const removeBtn = createEl('button', {
      class: 'rec-btn rec-btn-remove-corner',
      'aria-label': 'Not interested',
      title: 'Not interested'
    }, '×');
    
    favBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleFavorite(item);
      const nowFav = !!state.favorites[key];
      favBtn.classList.toggle('is-fav', nowFav);
      favBtn.setAttribute('aria-label', nowFav ? 'Remove from favorites' : 'Add to favorites');
      favBtn.setAttribute('title', nowFav ? 'Remove from favorites' : 'Add to favorites');
    });
    
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeRecommendation(index);
    });
    
    inner.appendChild(title);
    inner.appendChild(sub);
    card.appendChild(img);
    card.appendChild(favBtn);
    card.appendChild(removeBtn);
    card.appendChild(inner);
    card.addEventListener('click', () => showDetails(item));
    card.addEventListener('keydown', (e) => { if(e.key==='Enter') showDetails(item); });
    els.grid.appendChild(card);
  });
}

// =============================================================================
// Recommendation Logic
// =============================================================================

/** Remove a recommendation and fetch more if needed to maintain display count */
async function removeRecommendation(index){
  state.recommendations.splice(index, 1);

  if (state.recommendations.length < DISPLAY_COUNT) {
    await fetchMoreRecommendations();
  }

  renderRecommendations(state.recommendations);
}

/** Fetch additional recommendations, excluding previously shown anime */
async function fetchMoreRecommendations(){
  const likedAnime = Object.keys(state.favorites);
  if (!likedAnime.length) return;

  const needed = DISPLAY_COUNT - state.recommendations.length;
  if (needed <= 0) return;

  try {
    const res = await timeoutFetch(`${API_BASE_URL}/api/v1/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({
        liked_anime: likedAnime,
        top_n: needed,
        exclude_ids: [...state.shownIds]
      })
    });

    if (!res.ok) return;

    const data = await res.json();
    const newItems = (data.recommendations || []).map(mapAnimeResult);

    // Add new items to shown set and recommendations
    newItems.forEach(item => state.shownIds.add(item.id));
    state.recommendations.push(...newItems);
    renderRecommendations(state.recommendations);
  } catch (err) {
    logger.error('Error fetching more recommendations:', err);
  }
}

// =============================================================================
// User Actions
// =============================================================================

/** Toggle an anime's favorite status */
function toggleFavorite(item){
  const key = item.name || item.title;
  if (state.favorites[key]){
    delete state.favorites[key];
  } else {
    state.favorites[key] = { name: key, title: key };
  }
  renderFavorites();
  persistFavorites();
}

/** Show anime details in modal dialog */
async function showDetails(item){
  try {
    setLoading(document.body, true);
    openModal(renderDetailContent(item));
  } catch(err){
    logger.error('Details error:', err);
    openModal(createEl('div', {}, [
      createEl('h3', {}, 'Unable to Load Details'),
      createEl('p', {}, 'We couldn\'t load this anime\'s details at this time. Please try again later.'),
    ]));
  } finally {
    setLoading(document.body, false);
  }
}

/** Build modal content for anime details */
function renderDetailContent(data){
  const body = createEl('div', {class:'modal-body'});
  const img = createEl('img', {src: data.image || './assets/placeholder.svg', alt: `${data.title || 'Anime'} cover`});
  const meta = createEl('div');
  meta.appendChild(createEl('h3', {}, data.title || 'Untitled'));
  if (data.title_english) meta.appendChild(createEl('div', {class:'muted'}, data.title_english));
  if (data.title_japanese) meta.appendChild(createEl('div', {class:'muted'}, data.title_japanese));
  if (data.year) meta.appendChild(createEl('div', {class:'muted'}, `Year: ${data.year}`));
  if (Array.isArray(data.genres) && data.genres.length) meta.appendChild(createEl('div', {class:'muted'}, `Genres: ${data.genres.join(', ')}`));
  if (data.synopsis) meta.appendChild(createEl('p', {}, data.synopsis));
  body.appendChild(img);
  body.appendChild(meta);
  return body;
}

/** Open modal with given content and set up focus trapping */
function openModal(content){
  els.modalContent.innerHTML = '';
  els.modalContent.appendChild(content);
  els.modal.setAttribute('aria-hidden', 'false');
  updateFocusTrap();
  els.modal.addEventListener('keydown', trapFocus);
  requestAnimationFrame(() => {
    const closeBtn = els.modal.querySelector('.modal-close');
    closeBtn?.focus();
  });
}

/** Close modal and clean up event listeners */
function closeModal(){
  els.modal.setAttribute('aria-hidden', 'true');
  els.modal.removeEventListener('keydown', trapFocus);
}

// =============================================================================
// Event Listeners & Initialization
// =============================================================================

// Modal close handlers
$$('[data-close-modal]').forEach(el => el.addEventListener('click', closeModal));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && els.modal.getAttribute('aria-hidden') === 'false') closeModal();
});

/** Debounced search handler - triggers after 300ms of no input */
const doSearch = debounce(async (q) => {
  if (!q || q.trim().length < 2){
    renderSuggestions([]);
    return;
  }
  state.lastQuery = q;
  setLoading(els.suggestions, true);
  try {
    const res = await timeoutFetch(`${API_BASE_URL}/api/v1/search-anime?query=${encodeURIComponent(q)}`, { headers: { "Accept": "application/json" } });
    if (!res.ok) {
      if (res.status === 404) {
        els.suggestions.innerHTML = '<p class="muted">No anime found matching your search.</p>';
        return;
      }
      throw new Error(`Search failed (${res.status})`);
    }
    const data = await res.json();
    const items = (data.results || []).map(mapAnimeResult);
    renderSuggestions(items);
  } catch (err){
    logger.error('Search error:', err);
    const errorMsg = err.name === 'AbortError'
      ? 'Search timed out. Please try again.'
      : 'Unable to search. Please check your connection.';
    els.suggestions.innerHTML = `<p class="muted">${errorMsg}</p>`;
  } finally {
    setLoading(els.suggestions, false);
  }
}, 300);

// Search input handlers
els.searchInput.addEventListener('input', (e) => {
  doSearch(e.target.value);
});
els.clearSearch.addEventListener('click', () => {
  els.searchInput.value = '';
  renderSuggestions([]);
  els.searchInput.focus();
});

// Get Recommendations button handler
els.getRecsBtn.addEventListener('click', async () => {
  const likedAnime = Object.keys(state.favorites);
  if (!likedAnime.length) return;
  setLoading(els.grid, true);

  // Reset state for fresh recommendations
  state.shownIds.clear();
  state.recommendations = [];

  try{
    const res = await timeoutFetch(`${API_BASE_URL}/api/v1/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept':'application/json' },
      body: JSON.stringify({ liked_anime: likedAnime, top_n: DISPLAY_COUNT })
    });
    if (!res.ok) {
      const errorMsg = res.status === 500
        ? 'Server error. Please try again later.'
        : `Recommendation failed (${res.status})`;
      throw new Error(errorMsg);
    }
    const data = await res.json();
    const items = (data.recommendations || []).map(mapAnimeResult);

    // Track shown IDs for future exclusion
    items.forEach(item => state.shownIds.add(item.id));
    renderRecommendations(items);
    requestAnimationFrame(() => scrollToSection('recommended'));
  } catch(err){
    logger.error('Recommendation error:', err);
    const errorMsg = err.name === 'AbortError'
      ? 'Request timed out. Please try again.'
      : err.message || 'Unable to generate recommendations. Please try again.';
    els.grid.innerHTML = `<p class="muted">${errorMsg}</p>`;
  } finally {
    setLoading(els.grid, false);
  }
});

// Initialize UI on page load
renderFavorites();
updateCTA();

// Lazy loading for images (performance optimization)
if ('IntersectionObserver' in window) {
  const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        if (img.dataset.src) {
          img.src = img.dataset.src;
          img.removeAttribute('data-src');
          imageObserver.unobserve(img);
        }
      }
    });
  }, { rootMargin: '50px' });
  window.lazyLoadImage = (img) => imageObserver.observe(img);
}

// Smooth scroll for navigation links
$$('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const targetId = link.getAttribute('href').substring(1);
    scrollToSection(targetId, 12);
  });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  // Ctrl+K: Focus search input
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    els.searchInput?.focus();
  }
  // Ctrl+Enter: Get recommendations
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    if (!els.getRecsBtn.disabled) {
      e.preventDefault();
      els.getRecsBtn.click();
    }
  }
});

// Network status monitoring
window.addEventListener('online', () => {
  logger.info('Connection restored');
});
window.addEventListener('offline', () => {
  logger.warn('No internet connection');
});

// Global error handling
window.addEventListener('error', (e) => {
  logger.error('Uncaught error:', e.error);
});
window.addEventListener('unhandledrejection', (e) => {
  logger.error('Unhandled promise rejection:', e.reason);
});
