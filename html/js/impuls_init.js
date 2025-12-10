// ============================================
// IMPULS Initialization
// ============================================

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Health check on load and periodically
    checkHealth();
    setInterval(checkHealth, 30000);
    
    // Search input - Enter key
    document.getElementById('search-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') performSearch();
    });
    
    // Graph search input - Enter key
    const graphSearchInput = document.getElementById('graph-search-input');
    if (graphSearchInput) {
        graphSearchInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') searchKBConcepts();
        });
    }
    
    // Close context menu on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.graph-context-menu') && !e.target.closest('.node')) {
            hideContextMenu();
        }
    });
    
    // Close context menu on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideContextMenu();
        }
    });
});
