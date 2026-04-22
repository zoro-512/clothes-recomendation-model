document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const queryInput = document.getElementById('query-input');
    const resultsContainer = document.getElementById('results-container');
    const template = document.getElementById('product-card-template');
    const loader = document.getElementById('loader');
    const emptyState = document.getElementById('empty-state');
    const seasonPills = document.querySelectorAll('.season-pill');
    const genderPills = document.querySelectorAll('.gender-pill');

    let currentSeason = 'unknown';
    let currentGender = 'all';

    // Handle Season Pills
    seasonPills.forEach(pill => {
        pill.addEventListener('click', () => {
            seasonPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentSeason = pill.dataset.season;
        });
    });

    // Handle Gender Pills
    genderPills.forEach(pill => {
        pill.addEventListener('click', () => {
            genderPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentGender = pill.dataset.gender;
        });
    });

    // Handle Search Submit
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // UI State
        emptyState.classList.add('hidden');
        resultsContainer.innerHTML = '';
        loader.classList.remove('hidden');

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    user_id: 'USER_001', // Mock user for demo
                    top_k: 5,
                    season: currentSeason,
                    gender: currentGender
                })
            });

            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            renderResults(data.recommendations);
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            resultsContainer.innerHTML = `<div class="empty-state" style="color: #ef4444;"><i class="ph-duotone ph-warning-circle"></i><h3>Something went wrong.</h3><p>Make sure the backend is running properly.</p></div>`;
        } finally {
            loader.classList.add('hidden');
        }
    });

    // Render Logic
    function renderResults(items) {
        if (!items || items.length === 0) {
            resultsContainer.innerHTML = `<div class="empty-state"><i class="ph-duotone ph-magnifying-glass"></i><h3>No results found</h3></div>`;
            return;
        }

        items.forEach((item, index) => {
            const clone = template.content.cloneNode(true);
            
            // Populate Data
            clone.querySelector('.rank-badge span').textContent = index + 1;
            clone.querySelector('.semantic-badge span').textContent = Math.round(item.semantic_score * 100);
            
            clone.querySelector('.product-title').textContent = item.product_name;
            clone.querySelector('.product-group').textContent = item.product_group || 'Apparel';
            clone.querySelector('.product-desc').textContent = item.description || 'No description available for this item...';
            
            if (item.explanation) {
                clone.querySelector('.explanation-text').textContent = item.explanation;
            } else {
                clone.querySelector('.llm-explanation').style.display = 'none';
            }

            // Use real product URL from API response
            const myntraUrl = item.product_url || '#';
            clone.querySelector('.shop-now-btn').href = myntraUrl;

            // Stagger animation delay
            const card = clone.querySelector('.product-card');
            card.style.animationDelay = `${index * 0.15}s`;

            resultsContainer.appendChild(clone);
        });
    }
});
