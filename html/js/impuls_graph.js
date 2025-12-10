// ============================================
// IMPULS Graph - Knowledge Base Explorer
// Fixed version with proper node/link management
// Uses single-click for context menu (not right-click)
// ============================================

// Graph state - use Map for O(1) lookups
let graphNodes = new Map();  // id -> node object
let graphLinks = [];         // array of {source: id, target: id, type}
let graphSimulation = null;
let selectedNodeId = null;
let contextMenuNodeId = null;

// Node helpers
function getNode(id) {
    return graphNodes.get(id);
}

function addNode(id, label, type, data) {
    if (graphNodes.has(id)) {
        return graphNodes.get(id);
    }
    const node = { id, label, type, data, x: null, y: null };
    graphNodes.set(id, node);
    return node;
}

function addLink(sourceId, targetId, type) {
    const exists = graphLinks.some(l => 
        (l.source === sourceId && l.target === targetId) ||
        (l.source === targetId && l.target === sourceId)
    );
    if (!exists) {
        graphLinks.push({ source: sourceId, target: targetId, type });
    }
}

function removeNode(id) {
    graphNodes.delete(id);
    graphLinks = graphLinks.filter(l => l.source !== id && l.target !== id);
}

function clearGraph() {
    graphNodes.clear();
    graphLinks = [];
    selectedNodeId = null;
    contextMenuNodeId = null;
    hideContextMenu();
}

function updateInfoBar() {
    const infoBar = document.getElementById('graph-info-bar');
    const countSpan = document.getElementById('graph-node-count');
    if (graphNodes.size > 0) {
        infoBar.style.display = 'block';
        countSpan.textContent = graphNodes.size;
    } else {
        infoBar.style.display = 'none';
    }
}

// KB Search
async function searchKBConcepts() {
    const query = document.getElementById('graph-search-input').value.trim();
    if (query.length < 2) return;
    
    const resultsDiv = document.getElementById('graph-search-results');
    resultsDiv.innerHTML = '<div style="padding: 20px; color: #888; text-align: center;">Cercant...</div>';
    
    try {
        const response = await apiFetch(`${API_URL}/kb/search?q=${encodeURIComponent(query)}&limit=20`);
        const results = await response.json();
        
        if (results.length === 0) {
            resultsDiv.innerHTML = '<div style="padding: 20px; color: #888; text-align: center;">Cap resultat trobat</div>';
            return;
        }
        
        resultsDiv.innerHTML = results.map(r => `
            <div class="graph-search-result" onclick="loadConceptToGraph('${r.wikidata_id}')">
                <div class="keyword">${r.keyword || r.label_en || r.wikidata_id}</div>
                <div class="labels">
                    ${r.label_ca ? `CA: ${r.label_ca}` : ''}
                    ${r.label_es ? ` · ES: ${r.label_es}` : ''}
                    ${r.label_en ? ` · EN: ${r.label_en}` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        resultsDiv.innerHTML = `<div style="padding: 20px; color: #c33;">Error: ${error.message}</div>`;
    }
}

// Load concept to graph
async function loadConceptToGraph(wikidataId, clearExisting = true) {
    try {
        const response = await apiFetch(`${API_URL}/kb/concept/${encodeURIComponent(wikidataId)}`);
        const concept = await response.json();
        
        if (clearExisting) clearGraph();
        
        const mainLabel = concept.labels.en || concept.labels.ca || concept.labels.es || concept.keyword;
        
        // Add main concept
        addNode(concept.wikidata_id, mainLabel, 'selected', concept);
        selectedNodeId = concept.wikidata_id;
        
        // Add parents
        concept.parents.forEach(p => {
            const parentLabel = p.label_en || p.label_ca || p.label_es || p.keyword;
            const parentType = p.in_kb ? 'parent' : 'external';
            addNode(p.wikidata_id, parentLabel, parentType, p);
            addLink(concept.wikidata_id, p.wikidata_id, 'parent');
        });
        
        // Add children
        concept.children.forEach(c => {
            const childLabel = c.label_en || c.label_ca || c.label_es || c.keyword;
            addNode(c.wikidata_id, childLabel, 'child', c);
            addLink(c.wikidata_id, concept.wikidata_id, 'child');
        });
        
        renderGraph();
        showConceptDetails(concept);
    } catch (error) {
        console.error('Error loading concept:', error);
        alert('Error carregant concepte: ' + error.message);
    }
}

// Expand a node
async function expandNode(wikidataId) {
    const node = getNode(wikidataId);
    if (!node || node.type === 'external') {
        alert('Aquest concepte no és al KB');
        return;
    }
    
    try {
        const response = await apiFetch(`${API_URL}/kb/concept/${encodeURIComponent(wikidataId)}`);
        const concept = await response.json();
        
        node.data = concept;
        
        // Add parents
        concept.parents.forEach(p => {
            const parentLabel = p.label_en || p.label_ca || p.label_es || p.keyword;
            const parentType = p.in_kb ? 'parent' : 'external';
            addNode(p.wikidata_id, parentLabel, parentType, p);
            addLink(wikidataId, p.wikidata_id, 'parent');
        });
        
        // Add children
        concept.children.forEach(c => {
            const childLabel = c.label_en || c.label_ca || c.label_es || c.keyword;
            addNode(c.wikidata_id, childLabel, 'child', c);
            addLink(c.wikidata_id, wikidataId, 'child');
        });
        
        renderGraph(false);
    } catch (error) {
        console.error('Error expanding node:', error);
    }
}

// Render graph with D3
function renderGraph(resetPositions = true) {
    const svg = d3.select('#graph-canvas');
    const container = document.querySelector('.graph-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();
    
    updateInfoBar();
    hideContextMenu();
    
    if (graphNodes.size === 0) return;
    
    // Convert to arrays for D3
    const nodesArray = Array.from(graphNodes.values());
    const linksArray = graphLinks.map(l => ({ source: l.source, target: l.target, type: l.type }));
    
    // Container for zoom
    const g = svg.append('g');
    
    // Zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);
    
    // Colors
    const colorScale = {
        'selected': '#667eea',
        'parent': '#28a745',
        'child': '#17a2b8',
        'external': '#ccc'
    };
    
    // Simulation
    if (graphSimulation) graphSimulation.stop();
    
    graphSimulation = d3.forceSimulation(nodesArray)
        .force('link', d3.forceLink(linksArray).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(40));
    
    // Preserve positions if not resetting
    if (!resetPositions) {
        nodesArray.forEach(n => {
            const existing = graphNodes.get(n.id);
            if (existing && existing.x != null) {
                n.x = existing.x;
                n.y = existing.y;
            }
        });
        graphSimulation.alpha(0.3).restart();
    }
    
    // Draw links
    const link = g.append('g')
        .selectAll('line')
        .data(linksArray)
        .enter().append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 2);
    
    // Draw nodes
    const node = g.append('g')
        .selectAll('g')
        .data(nodesArray)
        .enter().append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Node circles
    node.append('circle')
        .attr('r', d => d.id === selectedNodeId ? 25 : 18)
        .attr('fill', d => colorScale[d.type] || '#999')
        .attr('stroke', d => d.id === selectedNodeId ? '#333' : '#fff')
        .attr('stroke-width', d => d.id === selectedNodeId ? 3 : 2)
        .style('cursor', 'pointer');
    
    // Node labels
    node.append('text')
        .text(d => d.label.length > 20 ? d.label.substring(0, 18) + '...' : d.label)
        .attr('text-anchor', 'middle')
        .attr('dy', d => d.id === selectedNodeId ? 40 : 32)
        .attr('font-size', '11px')
        .attr('fill', '#333');
    
    // SINGLE CLICK for context menu (instead of right-click)
    node.on('click', function(event, d) {
        event.stopPropagation();
        
        // If menu is open for this node, close it
        if (contextMenuNodeId === d.id && document.getElementById('graph-context-menu').classList.contains('show')) {
            hideContextMenu();
            return;
        }
        
        // Show context menu
        showContextMenu(event, d.id);
    });
    
    // Double-click to expand
    node.on('dblclick', function(event, d) {
        event.preventDefault();
        event.stopPropagation();
        hideContextMenu();
        if (d.type !== 'external') {
            expandNode(d.id);
        }
    });
    
    // Click on background to close menu
    svg.on('click', () => {
        hideContextMenu();
    });
    
    // Update on tick
    graphSimulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node.attr('transform', d => `translate(${d.x},${d.y})`);
        
        // Save positions
        nodesArray.forEach(n => {
            const stored = graphNodes.get(n.id);
            if (stored) {
                stored.x = n.x;
                stored.y = n.y;
            }
        });
    });
    
    function dragstarted(event, d) {
        if (!event.active) graphSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) graphSimulation.alphaTarget(0);
        d.fx = event.x;
        d.fy = event.y;
    }
}

// Context menu - now triggered by single click
function showContextMenu(event, nodeId) {
    contextMenuNodeId = nodeId;
    const node = getNode(nodeId);
    const menu = document.getElementById('graph-context-menu');
    const expandItem = document.getElementById('context-menu-expand');
    
    // Disable expand for external nodes
    if (node && node.type === 'external') {
        expandItem.classList.add('disabled');
    } else {
        expandItem.classList.remove('disabled');
    }
    
    // Position menu near the click
    menu.style.left = (event.pageX || event.clientX) + 'px';
    menu.style.top = (event.pageY || event.clientY) + 'px';
    menu.classList.add('show');
    
    // Also show details panel
    if (node && node.data) {
        showConceptDetails(node.data);
    }
}

function hideContextMenu() {
    document.getElementById('graph-context-menu').classList.remove('show');
    contextMenuNodeId = null;
}

function contextMenuAction(action) {
    if (!contextMenuNodeId) return;
    
    const node = getNode(contextMenuNodeId);
    if (!node) return;
    
    switch (action) {
        case 'search':
            searchProjectsWithConcept(node.label);
            break;
        case 'expand':
            if (node.type !== 'external') {
                expandNode(contextMenuNodeId);
            } else {
                alert('Aquest concepte no és al KB');
            }
            break;
        case 'center':
            loadConceptToGraph(contextMenuNodeId, true);
            break;
        case 'hide':
            removeNode(contextMenuNodeId);
            if (selectedNodeId === contextMenuNodeId) {
                selectedNodeId = null;
                document.getElementById('graph-details-panel').classList.remove('show');
            }
            renderGraph(false);
            break;
    }
    
    hideContextMenu();
}

// Show concept details panel
function showConceptDetails(concept) {
    const panel = document.getElementById('graph-details-panel');
    const labels = concept.labels || {};
    const aliases = concept.aliases || {};
    
    let aliasesHtml = '';
    for (const [lang, aliasList] of Object.entries(aliases)) {
        if (aliasList && aliasList.length > 0) {
            aliasesHtml += `<div class="aliases-list">
                ${aliasList.slice(0, 5).map(a => `<span class="alias-tag">${a}</span>`).join('')}
                ${aliasList.length > 5 ? `<span class="alias-tag">+${aliasList.length - 5}</span>` : ''}
            </div>`;
        }
    }
    
    const displayLabel = labels.en || labels.ca || labels.es || concept.keyword || concept.wikidata_id;
    
    panel.innerHTML = `
        <h4>${displayLabel}</h4>
        <div class="detail-row">
            <div class="detail-label">Etiquetes</div>
            <div class="detail-value">
                ${labels.ca ? `CA: ${labels.ca}<br>` : ''}
                ${labels.es ? `ES: ${labels.es}<br>` : ''}
                ${labels.en ? `EN: ${labels.en}` : ''}
            </div>
        </div>
        ${aliasesHtml ? `<div class="detail-row"><div class="detail-label">Àlies</div>${aliasesHtml}</div>` : ''}
        ${concept.definition ? `<div class="detail-row"><div class="detail-label">Definició</div><div class="detail-value">${concept.definition}</div></div>` : ''}
        <div class="detail-row">
            <div class="detail-label">Wikidata</div>
            <div class="detail-value"><a href="https://www.wikidata.org/wiki/${concept.wikidata_id}" target="_blank">${concept.wikidata_id}</a></div>
        </div>
        <div class="action-buttons">
            <button onclick="searchProjectsWithConcept('${(displayLabel || '').replace(/'/g, "\\'")}')">🔍 Cercar projectes</button>
        </div>
    `;
    
    panel.classList.add('show');
}

// Search projects with concept
function searchProjectsWithConcept(conceptLabel) {
    // Switch to search tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector('.tab').classList.add('active');
    document.getElementById('search-tab').classList.add('active');
    
    // Set query and search
    document.getElementById('search-input').value = conceptLabel;
    performSearch();
}