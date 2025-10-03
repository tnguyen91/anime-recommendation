/* AnimeReco - Production-ready Vanilla JS SPA
   Notes:
   - Expects API endpoints:
     • GET   /search-anime?query=...           -> returns { results: [{ anime_id, name, image_url, synopsis? }] }
     • POST  /recommend  body: { liked_anime:[names] } -> returns { recommendations: [{ anime_id, name, image_url, synopsis? }, ...] } (10 items)
   - All network calls are wrapped with robust error handling and timeouts.
   - Favorites are persisted in localStorage.
*/

const API_BASE_URL = 'https://animereco-api-725392014501.us-west1.run.app';
const IS_DEV = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const logger = {
  error: (...args) => console.error(...args),
  warn: (...args) => IS_DEV && console.warn(...args),
  info: (...args) => IS_DEV && console.info(...args),
  log: (...args) => IS_DEV && console.log(...args)
};

const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const state = {
  favorites: loadFavorites(), // Map of name -> { name, title }
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

// Utilities
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
function timeoutFetch(resource, options={}, ms=12000){
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

/** Rendering **/
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
  if (!items || !items.length){
    // No suggestions, don't render placeholders to reduce noise
    return;
  }
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
      // Update button state in place
      const nowFav = !!state.favorites[key];
      btn.textContent = nowFav ? 'Remove' : 'Add';
      btn.classList.toggle('remove', nowFav);
      btn.classList.toggle('add', !nowFav);
    });

    meta.appendChild(title);
    card.appendChild(img);
    card.appendChild(meta);
    card.appendChild(btn);

    // Click anywhere else to open details
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
  // Force exactly two rows of five if 10 results; otherwise responsive grid
  items.slice(0, 10).forEach(item => {
    const card = createEl('article', {class:'card-lg', tabindex:'0', role:'button', 'aria-label': `${item.title}`});
    const img = createEl('img', {src: item.image || './assets/placeholder.svg', alt: `${item.title} cover`});
    const inner = createEl('div', {class:'p'});
    const title = createEl('h3', {}, item.title);
    const sub = createEl('div', {class:'sub'}, item.year ? String(item.year) : (item.genres && item.genres.length ? item.genres.slice(0,2).join(' • ') : ''));

    inner.appendChild(title);
    inner.appendChild(sub);
    card.appendChild(img);
    card.appendChild(inner);

    card.addEventListener('click', () => showDetails(item));
    card.addEventListener('keydown', (e) => { if(e.key==='Enter') showDetails(item); });

    els.grid.appendChild(card);
  });
}

/** Favorites **/
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

/** Details Modal **/
async function showDetails(item){
  try {
    setLoading(document.body, true);
    // Note: API doesn't have /anime/:id endpoint, so we use the data we already have
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
  // Focus trap entry
  setTimeout(() => els.modalContent.focus(), 0);
}
function closeModal(){
  els.modal.setAttribute('aria-hidden', 'true');
}

// Modal events
$$('[data-close-modal]').forEach(el => el.addEventListener('click', closeModal));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && els.modal.getAttribute('aria-hidden') === 'false') closeModal();
});

/** Search **/
const doSearch = debounce(async (q) => {
  if (!q || q.trim().length < 2){
    renderSuggestions([]);
    return;
  }
  // cancel previous (handled via timeoutFetch abort on timeout)
  state.lastQuery = q;
  setLoading(els.suggestions, true);
  try {
    const res = await timeoutFetch(`${API_BASE_URL}/search-anime?query=${encodeURIComponent(q)}`, { headers: { "Accept": "application/json" } });
    if (!res.ok) throw new Error(`Search failed (${res.status})`);
    const data = await res.json();
    // Normalize
    const items = (data.results || [])
      .map(x => ({
        id: x.anime_id,
        name: x.name,
        title: x.name,
        image: x.image_url ?? './assets/placeholder.svg',
        year: x.year ?? null,
        genres: x.genre ?? [],
        synopsis: x.synopsis ?? null,
      }));
    renderSuggestions(items);
  } catch (err){
    logger.error('Search error:', err);
    els.suggestions.innerHTML = '<p class="muted">Unable to search at this time. Please check your connection and try again.</p>';
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

/** Get Recommendations **/
els.getRecsBtn.addEventListener('click', async () => {
  const likedAnime = Object.keys(state.favorites);
  if (!likedAnime.length) return;
  setLoading(els.grid, true);
  try{
    const res = await timeoutFetch(`${API_BASE_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept':'application/json' },
      body: JSON.stringify({ liked_anime: likedAnime })
    });
    if (!res.ok) throw new Error(`Recommend failed (${res.status})`);
    const data = await res.json();
    const items = (data.recommendations || [])
      .map(x => ({
        id: x.anime_id,
        name: x.name,
        title: x.name,
        image: x.image_url ?? './assets/placeholder.svg',
        year: x.year ?? null,
        genres: x.genre ?? [],
        synopsis: x.synopsis ?? null,
      }));
    renderRecommendations(items);
    // Smooth scroll to section
    document.querySelector('#recommended').scrollIntoView({ behavior: 'smooth', block:'start' });
  } catch(err){
    logger.error('Recommendation error:', err);
    els.grid.innerHTML = '<p class="muted">Unable to generate recommendations at this time. Please check your connection and try again.</p>';
  } finally {
    setLoading(els.grid, false);
  }
});

/** Init **/
renderFavorites();
updateCTA();
$("#year").textContent = new Date().getFullYear();

// Click on favorite chip removes it
// (handled in renderFavorites)

// Accessibility: mark loading state via [data-loading] if needed
const observer = new MutationObserver(() => {
  // could add aria-busy indicators here per element if desired
});
observer.observe(document.body, { attributes:true, subtree:true, attributeFilter:['data-loading'] });
