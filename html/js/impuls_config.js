// ============================================
// IMPULS Configuration & API Helper
// ============================================

const API_URL = 'http://impuls-aina.sirisacademic.com:8000';
const API_KEY = 'wV3oLPmhsYCfEp2nuOYqrflEKHdPvKqY';

// Authenticated fetch wrapper
async function apiFetch(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        ...(options.headers || {})
    };
    return fetch(url, { ...options, headers });
}

// Global state for search
let currentExpansionTerms = { aliases: [], parents: [], excluded: new Set() };
let lastSearchState = {
    queryUsed: '',
    filters: null,
    aliasLevels: [],
    parentLevels: [],
    k: 50,
    expansionResult: null
};

// Health check
async function checkHealth() {
    try {
        const response = await apiFetch(`${API_URL}/health`);
        const data = await response.json();
        const badge = document.getElementById('status-badge');
        if (data.status === 'ok') {
            badge.className = 'status-badge online';
            badge.textContent = data.index_size > 0 ? `Online · ${data.index_size} docs` : 'Online';
        } else {
            badge.className = 'status-badge offline';
            badge.textContent = 'Offline';
        }
        return data;
    } catch (error) {
        document.getElementById('status-badge').className = 'status-badge offline';
        document.getElementById('status-badge').textContent = 'Offline';
        return null;
    }
}

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');
    if (tabName === 'stats') loadStats();
}

// UI Helpers
function toggleCheckbox(id) {
    const cb = document.getElementById(id);
    cb.checked = !cb.checked;
    cb.parentElement.classList.toggle('active', cb.checked);
    if (id === 'use-expansion') {
        document.getElementById('expansion-controls').classList.toggle('show', cb.checked);
    }
}

function toggleLevel(element, type, level) {
    const cb = element.querySelector('input');
    cb.checked = !cb.checked;
    element.classList.toggle('checked', cb.checked);
}

function getSelectedLevels() {
    const aliasLevels = [], parentLevels = [];
    document.querySelectorAll('.expansion-section').forEach((section, sectionIdx) => {
        section.querySelectorAll('.level-checkbox').forEach((cb, levelIdx) => {
            if (cb.classList.contains('checked')) {
                if (sectionIdx === 0) aliasLevels.push(levelIdx + 1);
                else parentLevels.push(levelIdx + 1);
            }
        });
    });
    return { aliasLevels, parentLevels };
}

function toggleFilters() {
    const content = document.getElementById('filters-content');
    const arrow = document.getElementById('filters-arrow');
    content.classList.toggle('show');
    arrow.textContent = content.classList.contains('show') ? '▲' : '▼';
}

function clearFilters() {
    ['filter-framework', 'filter-instrument', 'filter-year-from', 'filter-year-to',
     'filter-country', 'filter-region', 'filter-province', 'filter-org-type'].forEach(id => {
        const el = document.getElementById(id);
        el.value = '';
        el.classList.remove('populated');
    });
}

function toggleDebug() {
    const content = document.getElementById('debug-content');
    const icon = document.getElementById('debug-icon');
    content.classList.toggle('show');
    icon.classList.toggle('open');
}

function toggleAbstract(id, btn) {
    const el = document.getElementById(id);
    if (el.classList.contains('truncated')) {
        el.classList.remove('truncated');
        el.classList.add('expanded');
        btn.textContent = 'Llegir menys ▲';
    } else {
        el.classList.remove('expanded');
        el.classList.add('truncated');
        btn.textContent = 'Llegir més ▼';
    }
}

// Stats
async function loadStats() {
    try {
        const data = await checkHealth();
        if (data) displayStats(data);
    } catch (error) {
        document.getElementById('stats-results').innerHTML = 
            `<div class="error">Error carregant informació: ${error.message}</div>`;
    }
}

function displayStats(data) {
    document.getElementById('stats-results').innerHTML = `
        <div class="stats">
            <div class="stat-card"><div class="stat-value">${data.index_size || 0}</div><div class="stat-label">Documents Indexats</div></div>
            <div class="stat-card"><div class="stat-value">${data.projects_metadata_loaded || 0}</div><div class="stat-label">Projectes amb Metadades</div></div>
            <div class="stat-card"><div class="stat-value">${data.kb_concepts || 0}</div><div class="stat-label">Conceptes KB</div></div>
            <div class="stat-card"><div class="stat-value">${data.parser_loaded ? '✓' : '✗'}</div><div class="stat-label">Parser Intel·ligent</div></div>
        </div>
        <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-bottom: 15px;">🛠️ Components del Sistema</h3>
            <ul style="list-style: none; line-height: 2;">
                <li>📊 <strong>Embeddings:</strong> mRoBERTA (BSC)</li>
                <li>🗄️ <strong>Index Vectorial:</strong> HNSW</li>
                <li>🤖 <strong>Parser:</strong> Salamandra-7B fine-tuned</li>
                <li>📚 <strong>Base de Coneixement:</strong> Wikidata (${data.kb_indexed || 0} conceptes indexats)</li>
                <li>🔗 <strong>Expansió:</strong> Termes equivalents multilingües + Conceptes amplis</li>
            </ul>
        </div>`;
}
