// ============================================
// IMPULS Search Functions
// ============================================

function buildFilters() {
    const filters = {};
    const fw = document.getElementById('filter-framework').value.trim();
    const inst = document.getElementById('filter-instrument').value.trim();
    const country = document.getElementById('filter-country').value.trim();
    const region = document.getElementById('filter-region').value.trim();
    const province = document.getElementById('filter-province').value.trim();
    const orgType = document.getElementById('filter-org-type').value.trim();
    const yf = document.getElementById('filter-year-from').value;
    const yt = document.getElementById('filter-year-to').value;
    
    if (fw) filters.framework = fw.split(',').map(s => s.trim());
    if (inst) filters.instrument = inst;
    if (country) filters.country = country.split(',').map(s => s.trim());
    if (region) filters.region = region.split(',').map(s => s.trim());
    if (province) filters.province = province.split(',').map(s => s.trim());
    if (orgType) filters.organization_type = orgType.split(',').map(s => s.trim());
    if (yf) filters.year_from = parseInt(yf);
    if (yt) filters.year_to = parseInt(yt);
    
    return Object.keys(filters).length > 0 ? filters : null;
}

function populateFiltersFromParsed(filters) {
    if (!filters) return;
    const mappings = {
        'filter-framework': filters.framework ? filters.framework.join(', ') : '',
        'filter-instrument': filters.instrument || '',
        'filter-year-from': filters.year_from || '',
        'filter-year-to': filters.year_to || '',
        'filter-country': filters.country ? filters.country.join(', ') : '',
        'filter-region': filters.region ? filters.region.join(', ') : '',
        'filter-province': filters.province ? filters.province.join(', ') : '',
        'filter-org-type': filters.organization_type ? filters.organization_type.join(', ') : ''
    };
    for (const [id, value] of Object.entries(mappings)) {
        if (value) {
            const el = document.getElementById(id);
            el.value = value;
            el.classList.add('populated');
        }
    }
}

function toggleExpansionTerm(term, type) {
    const key = `${type}:${term}`;
    if (currentExpansionTerms.excluded.has(key)) {
        currentExpansionTerms.excluded.delete(key);
    } else {
        currentExpansionTerms.excluded.add(key);
    }
    updateExpansionTagsDisplay();
    performRefinementSearch();
}

function updateExpansionTagsDisplay() {
    const aliasContainer = document.getElementById('alias-tags');
    const parentContainer = document.getElementById('parent-tags');
    
    if (aliasContainer) {
        aliasContainer.innerHTML = currentExpansionTerms.aliases.map(term => {
            const excluded = currentExpansionTerms.excluded.has(`alias:${term}`);
            return `<span class="tag synonym ${excluded ? 'excluded' : ''}" onclick="toggleExpansionTerm('${term.replace(/'/g, "\\'")}', 'alias')">${term} <span class="tag-icon">${excluded ? '✕' : '✓'}</span></span>`;
        }).join('');
    }
    
    if (parentContainer) {
        parentContainer.innerHTML = currentExpansionTerms.parents.map(term => {
            const excluded = currentExpansionTerms.excluded.has(`parent:${term}`);
            return `<span class="tag parent ${excluded ? 'excluded' : ''}" onclick="toggleExpansionTerm('${term.replace(/'/g, "\\'")}', 'parent')">${term} <span class="tag-icon">${excluded ? '✕' : '✓'}</span></span>`;
        }).join('');
    }
}

function updateResultsCount(count) {
    const feedbackGrid = document.querySelector('.feedback-grid');
    if (feedbackGrid) {
        const labels = feedbackGrid.querySelectorAll('.feedback-label');
        labels.forEach((label, idx) => {
            if (label.textContent.includes('Resultats trobats')) {
                const valueSpan = feedbackGrid.querySelectorAll('.feedback-value')[idx];
                if (valueSpan) valueSpan.textContent = `${count} projectes`;
            }
        });
    }
}

async function performRefinementSearch() {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div class="loading">Actualitzant resultats</div>';
    const excludedTerms = Array.from(currentExpansionTerms.excluded).map(key => key.split(':')[1]);
    
    try {
        const requestBody = {
            query: lastSearchState.queryUsed,
            k: lastSearchState.k,
            k_factor: 5,
            filters: lastSearchState.filters,
            use_parsing: false,
            expansion: {
                enabled: true,
                alias_levels: lastSearchState.aliasLevels,
                parent_levels: lastSearchState.parentLevels,
                excluded_terms: excludedTerms,
                return_details: false
            }
        };
        
        const response = await apiFetch(`${API_URL}/search`, {
            method: 'POST',
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error en la cerca');
        }
        
        const data = await response.json();
        displayResults(data);
        updateResultsCount(data.total_matching);
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

async function performSearch(isReSearch = false) {
    const query = document.getElementById('search-input').value;
    const k = parseInt(document.getElementById('top-k').value);
    const useParsing = document.getElementById('use-parsing').checked;
    const useExpansion = document.getElementById('use-expansion').checked;
    
    const resultsDiv = document.getElementById('results');
    const feedbackDiv = document.getElementById('feedback-section');
    
    if (!isReSearch) {
        resultsDiv.innerHTML = '<div class="loading">Cercant</div>';
        feedbackDiv.classList.remove('show');
        currentExpansionTerms = { aliases: [], parents: [], excluded: new Set() };
        document.querySelectorAll('.filter-group input.populated').forEach(el => el.classList.remove('populated'));
    }
    
    const filters = buildFilters();
    const { aliasLevels, parentLevels } = getSelectedLevels();
    
    try {
        const requestBody = { query, k, k_factor: 5, filters, use_parsing: useParsing };
        if (useExpansion) {
            requestBody.expansion = {
                enabled: true,
                alias_levels: aliasLevels,
                parent_levels: parentLevels,
                return_details: true
            };
        }
        
        const response = await apiFetch(`${API_URL}/search`, {
            method: 'POST',
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error en la cerca');
        }
        
        const data = await response.json();
        
        lastSearchState = {
            queryUsed: data.query_used,
            filters: data.filters,
            aliasLevels,
            parentLevels,
            k,
            expansionResult: data.expansion
        };
        
        if (data.filters && useParsing) {
            populateFiltersFromParsed(data.filters);
            const hasActiveFilters = Object.entries(data.filters).some(([k, v]) => v !== null && k !== 'organisations');
            if (hasActiveFilters) {
                document.getElementById('filters-content').classList.add('show');
                document.getElementById('filters-arrow').textContent = '▲';
            }
        }
        
        displayDebugInfo(requestBody, data);
        displayFeedback(data, !isReSearch);
        displayResults(data);
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        feedbackDiv.classList.remove('show');
    }
}

function displayDebugInfo(request, response) {
    const debugSection = document.getElementById('debug-section');
    const debugContent = document.getElementById('debug-content');
    debugContent.innerHTML = `
        <div class="debug-box"><h4>📤 Request</h4><pre>${JSON.stringify(request, null, 2)}</pre></div>
        <div class="debug-box"><h4>📥 Response</h4><pre>${JSON.stringify(response, null, 2)}</pre></div>`;
    debugSection.style.display = 'block';
}

function displayFeedback(data, updateTerms = true) {
    const feedbackDiv = document.getElementById('feedback-section');
    const hasFeedback = data.feedback || data.expansion || (data.filters && Object.values(data.filters).some(v => v !== null));
    
    if (!hasFeedback) {
        feedbackDiv.classList.remove('show');
        return;
    }
    
    let html = '<div class="feedback-title">📋 Interpretació de la Consulta</div><div class="feedback-grid">';
    
    if (data.feedback?.query_rewrite) {
        html += `<span class="feedback-label">Interpretació completa:</span><span class="feedback-value">"${data.feedback.query_rewrite}"</span>`;
    }
    
    html += `<span class="feedback-label">Consulta temàtica:</span><span class="feedback-value highlight">"${data.query_used}"</span>`;
    
    if (data.expansion?.query_language) {
        const langNames = { 'CA': 'Català', 'ES': 'Castellà', 'EN': 'Anglès' };
        html += `<span class="feedback-label">Idioma detectat:</span><span class="feedback-value">${langNames[data.expansion.query_language] || data.expansion.query_language}</span>`;
    }
    
    html += `<span class="feedback-label">Resultats trobats:</span><span class="feedback-value">${data.total_matching} projectes</span></div>`;
    
    // Active filters
    const hasActiveFilters = data.filters && Object.entries(data.filters).some(([k, v]) => v !== null && k !== 'organisations');
    if (hasActiveFilters) {
        html += '<div class="active-filters"><span class="active-filters-label">Filtres aplicats:</span>';
        const filterLabels = { framework: 'Framework', instrument: 'Instrument', year_from: 'Des de', year_to: 'Fins a', country: 'País', region: 'Regió', province: 'Província', organization_type: 'Tipus org.' };
        for (const [key, value] of Object.entries(data.filters)) {
            if (value !== null && filterLabels[key]) {
                const displayValue = Array.isArray(value) ? value.join(', ') : value;
                html += `<span class="active-filter"><span class="filter-key">${filterLabels[key]}:</span> ${displayValue}</span>`;
            }
        }
        html += '</div>';
    }
    
    // Expansion tags
    if (data.expansion && (data.expansion.alias_levels || data.expansion.parent_levels)) {
        if (updateTerms) {
            currentExpansionTerms.aliases = [];
            currentExpansionTerms.parents = [];
            if (data.expansion.alias_levels) {
                Object.values(data.expansion.alias_levels).forEach(level => {
                    if (level.representatives) currentExpansionTerms.aliases.push(...level.representatives);
                });
            }
            if (data.expansion.parent_levels) {
                Object.values(data.expansion.parent_levels).forEach(level => {
                    if (level.representatives) currentExpansionTerms.parents.push(...level.representatives);
                });
            }
        }
        
        if (currentExpansionTerms.aliases.length > 0 || currentExpansionTerms.parents.length > 0) {
            html += '<div class="expansion-tags"><div class="expansion-tags-title">🏷️ Termes d\'expansió (clica per incloure/excloure):</div>';
            
            if (currentExpansionTerms.aliases.length > 0) {
                html += `<div class="expansion-tags-row"><span class="expansion-tags-label synonyms">🌐 Termes equivalents:</span><div id="alias-tags" class="tags-container">`;
                html += currentExpansionTerms.aliases.map(term => {
                    const excluded = currentExpansionTerms.excluded.has(`alias:${term}`);
                    return `<span class="tag synonym ${excluded ? 'excluded' : ''}" onclick="toggleExpansionTerm('${term.replace(/'/g, "\\'")}', 'alias')">${term} <span class="tag-icon">${excluded ? '✕' : '✓'}</span></span>`;
                }).join('');
                html += '</div></div>';
            }
            
            if (currentExpansionTerms.parents.length > 0) {
                html += `<div class="expansion-tags-row"><span class="expansion-tags-label parents">🔼 Conceptes amplis:</span><div id="parent-tags" class="tags-container">`;
                html += currentExpansionTerms.parents.map(term => {
                    const excluded = currentExpansionTerms.excluded.has(`parent:${term}`);
                    return `<span class="tag parent ${excluded ? 'excluded' : ''}" onclick="toggleExpansionTerm('${term.replace(/'/g, "\\'")}', 'parent')">${term} <span class="tag-icon">${excluded ? '✕' : '✓'}</span></span>`;
                }).join('');
                html += '</div></div>';
            }
            html += '</div>';
        }
    }
    
    if (data.feedback?.warning) {
        html += `<div style="color: #856404; margin-top: 10px; padding: 8px; background: #fff3cd; border-radius: 5px;">⚠️ ${data.feedback.warning}</div>`;
    }
    
    feedbackDiv.innerHTML = html;
    feedbackDiv.classList.add('show');
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    if (!data.results || data.results.length === 0) {
        resultsDiv.innerHTML = '<div class="info-message">No s\'han trobat resultats. Prova amb altres termes o ajusta els filtres.</div>';
        return;
    }
    
    let html = `<div class="results-summary"><span class="results-count">Mostrant ${data.returned} de ${data.total_matching} projectes</span></div>`;
    
    data.results.forEach((result, idx) => {
        const meta = result.metadata || {};
        const title = result.title || meta.title || `Projecte ${result.id}`;
        const abstract = result.abstract || meta.abstract || '';
        const hasAbstract = abstract && abstract.trim().length > 0;
        const matchedBy = result.matched_by || [];
        
        html += `<div class="result-item">
            <div class="result-header">
                <div>
                    <div class="result-title ${!hasAbstract ? 'no-abstract' : ''}">${title}</div>
                    <div class="result-id">ID: ${result.id}</div>
                </div>
                <div class="result-score">#${idx + 1} · ${(result.score * 100).toFixed(1)}%</div>
            </div>`;
        
        if (hasAbstract) {
            const abstractId = `abstract-${idx}`;
            html += `<div id="${abstractId}" class="result-abstract truncated">${abstract}</div>
                <button class="read-more-btn" onclick="toggleAbstract('${abstractId}', this)">Llegir més ▼</button>`;
        } else {
            html += `<div class="result-abstract empty">Descripció no disponible</div>`;
        }
        
        if (matchedBy.length > 0) {
            html += `<div class="result-matched"><div class="result-matched-label">🔎 Coincidència:</div><div class="result-matched-tags">`;
            html += matchedBy.map(term => {
                const isQuery = term === 'query';
                const isParent = currentExpansionTerms.parents.includes(term);
                const isSynonym = currentExpansionTerms.aliases.includes(term);
                const cls = isQuery ? 'query' : (isParent ? 'parent' : (isSynonym ? 'synonym' : ''));
                return `<span class="matched-tag ${cls}">${term}</span>`;
            }).join('');
            html += '</div></div>';
        }
        
        html += `<div class="result-meta">`;
        if (meta.framework_name) html += `<span class="meta-item"><span class="meta-label">Framework:</span> ${meta.framework_name}</span>`;
        if (meta.instrument_name) html += `<span class="meta-item"><span class="meta-label">Instrument:</span> ${meta.instrument_name}</span>`;
        if (meta.year) html += `<span class="meta-item"><span class="meta-label">Any:</span> ${parseInt(meta.year)}</span>`;
        if (result.participants?.length > 0) {
            const orgName = result.participants[0].organization_name;
            if (orgName) html += `<span class="meta-item"><span class="meta-label">Org:</span> ${orgName}</span>`;
        }
        html += `</div></div>`;
    });
    
    resultsDiv.innerHTML = html;
}
