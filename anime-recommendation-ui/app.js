const API_BASE_URL = window.APP_CONFIG?.API_BASE_URL || 'http://localhost:8000';
const IS_DEV = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const logger = {
  error: (...args) => console.error(...args),
  warn: (...args) => IS_DEV && console.warn(...args),
  info: (...args) => IS_DEV && console.info(...args),
  log: (...args) => IS_DEV && console.log(...args)
};

const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const DISPLAY_COUNT = 10;
const FETCH_COUNT = 20;

const state = {
  favorites: loadFavorites(),
  recommendations: [],
  shownIds: new Set(),  // Track all shown recommendation IDs
  lastQuery: '',
  aborter: null,
  scrolling: false,
};

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

function loadFavorites(){
  try {
    const raw = localStorage.getItem("favorites");
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch(_){ return {}; }
}

function persistFavorites(){
  localStorage.setItem("favorites", JSON.stringify(state.favorites));
  updateCTA();
}

function debounce(fn, ms){
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  }
}

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

function setLoading(el, isLoading){
  if(isLoading){
    el.setAttribute('data-loading', 'true');
    el.setAttribute('aria-busy', 'true');
  } else {
    el.removeAttribute('data-loading');
    el.removeAttribute('aria-busy');
  }
}

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

let focusableElements = [];
let firstFocusable = null;
let lastFocusable = null;

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

function updateFocusTrap() {
  const focusableSelectors = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
  focusableElements = Array.from(els.modal.querySelectorAll(focusableSelectors))
    .filter(el => !el.hasAttribute('disabled'));
  firstFocusable = focusableElements[0];
  lastFocusable = focusableElements[focusableElements.length - 1];
}

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

function updateCTA(){
  const hasFavs = Object.keys(state.favorites).length > 0;
  els.getRecsBtn.disabled = !hasFavs;
}

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

async function removeRecommendation(index){
  state.recommendations.splice(index, 1);

  if (state.recommendations.length < DISPLAY_COUNT) {
    await fetchMoreRecommendations();
  }

  renderRecommendations(state.recommendations);
}

async function fetchMoreRecommendations(){
  const likedAnime = Object.keys(state.favorites);
  if (!likedAnime.length) return;

  const needed = DISPLAY_COUNT - state.recommendations.length;
  if (needed <= 0) return;

  try {
    const res = await timeoutFetch(`${API_BASE_URL}/recommend`, {
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
    const newItems = (data.recommendations || []).map(x => ({
      id: x.anime_id,
      name: x.name,
      title: x.name,
      title_english: x.title_english ?? null,
      title_japanese: x.title_japanese ?? null,
      image: x.image_url ?? './assets/placeholder.svg',
      year: x.year ?? null,
      genres: x.genre ?? [],
      synopsis: x.synopsis ?? null,
    }));

    // Add new items to shown set and recommendations
    newItems.forEach(item => state.shownIds.add(item.id));
    state.recommendations.push(...newItems);
    renderRecommendations(state.recommendations);
  } catch (err) {
    logger.error('Error fetching more recommendations:', err);
  }
}

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

function closeModal(){
  els.modal.setAttribute('aria-hidden', 'true');
  els.modal.removeEventListener('keydown', trapFocus);
}

$$('[data-close-modal]').forEach(el => el.addEventListener('click', closeModal));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && els.modal.getAttribute('aria-hidden') === 'false') closeModal();
});

const doSearch = debounce(async (q) => {
  if (!q || q.trim().length < 2){
    renderSuggestions([]);
    return;
  }
  state.lastQuery = q;
  setLoading(els.suggestions, true);
  try {
    const res = await timeoutFetch(`${API_BASE_URL}/search-anime?query=${encodeURIComponent(q)}`, { headers: { "Accept": "application/json" } });
    if (!res.ok) {
      if (res.status === 404) {
        els.suggestions.innerHTML = '<p class="muted">No anime found matching your search.</p>';
        return;
      }
      throw new Error(`Search failed (${res.status})`);
    }
    const data = await res.json();
    const items = (data.results || [])
      .map(x => ({
        id: x.anime_id,
        name: x.name,
        title: x.name,
        title_english: x.title_english ?? null,
        title_japanese: x.title_japanese ?? null,
        image: x.image_url ?? './assets/placeholder.svg',
        year: x.year ?? null,
        genres: x.genre ?? [],
        synopsis: x.synopsis ?? null,
      }));
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

els.searchInput.addEventListener('input', (e) => {
  doSearch(e.target.value);
});
els.clearSearch.addEventListener('click', () => {
  els.searchInput.value = '';
  renderSuggestions([]);
  els.searchInput.focus();
});

els.getRecsBtn.addEventListener('click', async () => {
  const likedAnime = Object.keys(state.favorites);
  if (!likedAnime.length) return;
  setLoading(els.grid, true);

  // Reset state for fresh recommendations
  state.shownIds.clear();
  state.recommendations = [];

  try{
    const res = await timeoutFetch(`${API_BASE_URL}/recommend`, {
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
    const items = (data.recommendations || [])
      .map(x => ({
        id: x.anime_id,
        name: x.name,
        title: x.name,
        title_english: x.title_english ?? null,
        title_japanese: x.title_japanese ?? null,
        image: x.image_url ?? './assets/placeholder.svg',
        year: x.year ?? null,
        genres: x.genre ?? [],
        synopsis: x.synopsis ?? null,
      }));

    // Track shown IDs for future exclusion
    items.forEach(item => state.shownIds.add(item.id));
    renderRecommendations(items);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const targetEl = document.querySelector('#recommended');
        if (targetEl) {
          const header = document.querySelector('.site-header');
          const headerHeight = header ? header.offsetHeight : 0;
          const targetPosition = targetEl.getBoundingClientRect().top + window.scrollY - headerHeight;
          window.scrollTo({ top: targetPosition, behavior: 'smooth' });
        }
      });
    });
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

renderFavorites();
updateCTA();

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

$$('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const targetId = link.getAttribute('href').substring(1);
    const targetEl = document.getElementById(targetId);
    if (targetEl) {
      const header = document.querySelector('.site-header');
      const headerHeight = header ? header.offsetHeight : 0;
      const targetPosition = targetEl.getBoundingClientRect().top + window.scrollY - headerHeight - 12;
      window.scrollTo({ top: targetPosition, behavior: 'smooth' });
    }
  });
});

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    els.searchInput?.focus();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    if (!els.getRecsBtn.disabled) {
      e.preventDefault();
      els.getRecsBtn.click();
    }
  }
});

window.addEventListener('online', () => {
  logger.info('Connection restored');
});
window.addEventListener('offline', () => {
  logger.warn('No internet connection');
});
window.addEventListener('error', (e) => {
  logger.error('Uncaught error:', e.error);
});
window.addEventListener('unhandledrejection', (e) => {
  logger.error('Unhandled promise rejection:', e.reason);
});
