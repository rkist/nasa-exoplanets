// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';
const POSITIVE_LABEL = 'Candidate';
const NEGATIVE_LABEL = 'Likely False Positive';
let cachedModelStats = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    createStars();
    initializeSliders();
    resetStatsDisplay();
});

// Create animated stars background
function createStars() {
    const starsContainer = document.getElementById('stars');
    for (let i = 0; i < 100; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.width = star.style.height = Math.random() * 3 + 'px';
        star.style.animationDelay = Math.random() * 3 + 's';
        starsContainer.appendChild(star);
    }
}

// Tab switching
function switchTab(event, tabName) {
    event.preventDefault();
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    const tabElement = document.getElementById(tabName);
    if (tabElement) {
        tabElement.classList.add('active');
    }
    event.currentTarget.classList.add('active');
    
    // Load stats when stats tab is opened
    if (tabName === 'stats') {
        loadModelStats();
    }
}

// Initialize all sliders
function initializeSliders() {
    const sliders = document.querySelectorAll('.slider');
    sliders.forEach(slider => {
        if (slider.id === 'threshold-slider') {
            updateThresholdPreview();
        } else {
            updateSlider(slider.id);
        }
    });
}

// Update slider value display
function updateSlider(id) {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + '-value');
    if (!slider || !valueDisplay) return;
    let value = slider.value;
    
    if (id.includes('split')) {
        value += '%';
    }
    
    valueDisplay.textContent = value;
}

// Classify single data point
async function classifyData() {
    const featureFields = [
        { id: 'eff_temp', label: 'Effective Temperature' },
        { id: 'surface_gravity', label: 'Surface Gravity' },
        { id: 'metallicity', label: 'Metallicity' },
        { id: 'radius', label: 'Planet Radius' },
        { id: 'reddening', label: 'Reddening' },
        { id: 'extinction', label: 'Extinction' },
        { id: 'gkcolor', label: 'g-k Color Index' },
        { id: 'grcolor', label: 'g-r Color Index' },
        { id: 'jkcolor', label: 'j-k Color Index' }
    ];

    const data = {};
    const missingFields = [];

    featureFields.forEach(({ id, label }) => {
        const input = document.getElementById(id);
        const rawValue = input.value.trim();

        if (rawValue === '') {
            missingFields.push(label);
            return;
        }

        const numericValue = parseFloat(rawValue);
        if (!Number.isFinite(numericValue)) {
            missingFields.push(label);
            return;
        }

        data[id] = numericValue;
    });

    if (missingFields.length > 0) {
        alert(`Please provide valid numeric values for: ${missingFields.join(', ')}`);
        return;
    }
    
    const resultDiv = document.getElementById('classification-result');
    resultDiv.innerHTML = '<div class="loading"></div> Classifying...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const models = result.models || {};
            const tableHtml = renderModelComparison(models);

            resultDiv.innerHTML = `
                <div class="result-box" style="animation: fadeIn 0.5s ease;">
                    <h3 style="color: #667eea; margin-bottom: 15px;">Model Comparison</h3>
                    <p style="color: #8b95a5;">Each ensemble model scored this sample independently.</p>
                    ${tableHtml}
                    <p style="margin-top: 12px; color: #8b95a5; font-size: 0.9em;">Green rows highlight ${POSITIVE_LABEL} predictions; red rows mark ${NEGATIVE_LABEL} assessments.</p>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="result-box result-negative">
                    <p><strong>Error:</strong> ${result.error}</p>
                </div>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="result-box result-negative">
                <p><strong>Error:</strong> Could not connect to the server. Make sure the backend is running.</p>
            </div>
        `;
    }
}

// Classify JSONL payload
async function classifyJsonl() {
    const textarea = document.getElementById('jsonl-input');
    const resultDiv = document.getElementById('jsonl-result');
    const payload = textarea.value;

    if (!payload.trim()) {
        alert('Please paste at least one JSON line before submitting.');
        return;
    }

    resultDiv.innerHTML = '<div class="loading"></div> Classifying JSONL payload...';

    try {
        const response = await fetch(`${API_BASE_URL}/classify/jsonl`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ jsonl: payload })
        });

        const result = await response.json();

        if (response.ok) {
            const summary = `
                <p><strong>Total Lines:</strong> ${result.total}</p>
                <p><strong>Processed:</strong> ${result.processed}</p>
                <p><strong>Failed:</strong> ${result.failed}</p>
            `;

            const listItems = (result.results || []).map(formatJsonlResult).join('');

            resultDiv.innerHTML = `
                <div class="result-box" style="margin-top: 20px;">
                    ${summary}
                    <ul style="margin-top: 10px; color: #fff;">
                        ${listItems}
                    </ul>
                </div>
            `;
        } else {
            const listItems = (result.results || []).map(formatJsonlResult).join('');
            resultDiv.innerHTML = `
                <div class="result-box result-negative" style="margin-top: 20px;">
                    <p><strong>Error:</strong> ${result.error || 'Unable to classify JSONL payload.'}</p>
                    ${listItems ? `<ul style="margin-top: 10px;">${listItems}</ul>` : ''}
                </div>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="result-box result-negative" style="margin-top: 20px;">
                <p><strong>Error:</strong> Could not connect to the server. Make sure the backend is running.</p>
            </div>
        `;
    }
}

function formatJsonlResult(item) {
    if (!item) {
        return '';
    }

    if (item.status === 'ok') {
        const table = renderModelComparison(item.predictions || {});
        return `<li><strong>Line ${item.line}:</strong><div style="margin-top: 8px;">${table}</div></li>`;
    }

    return `<li><strong>Line ${item.line}:</strong> <span style="color: #ef4444;">${item.error || 'Unknown error'}</span></li>`;
}

function updateThresholdPreview() {
    const slider = document.getElementById('threshold-slider');
    const label = document.getElementById('threshold-slider-value');
    if (!slider || !label) return;

    const value = (parseInt(slider.value, 10) / 100).toFixed(2);
    label.textContent = value;
}

async function applyThreshold() {
    const slider = document.getElementById('threshold-slider');
    const modelSelect = document.getElementById('threshold-model');
    if (!slider || !modelSelect || !modelSelect.value) {
        alert('Please select a model before updating the threshold.');
        return;
    }

    const threshold = parseInt(slider.value, 10) / 100;

    try {
        const response = await fetch(`${API_BASE_URL}/model/threshold`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ threshold, model: modelSelect.value })
        });

        const result = await response.json();

        if (response.ok) {
            loadModelStats();
        } else {
            alert('Unable to update threshold: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: Could not connect to the server to update threshold.');
    }
}

function formatMetric(value) {
    if (typeof value === 'number' && !Number.isNaN(value)) {
        return (value * 100).toFixed(1) + '%';
    }
    return '--';
}

function renderModelComparison(models) {
    const entries = Object.entries(models || {});
    if (entries.length === 0) {
        return '<p style="color: #8b95a5;">No model predictions available.</p>';
    }

    const rows = entries.map(([name, details]) => {
        const isCandidate = details.prediction === POSITIVE_LABEL;
        const predictionBadge = isCandidate
            ? `<span style="color: #10b981; font-weight: 600;">${POSITIVE_LABEL}</span>`
            : `<span style="color: #ef4444; font-weight: 600;">${NEGATIVE_LABEL}</span>`;
        const probability = typeof details.probability === 'number'
            ? `${(details.probability * 100).toFixed(1)}%`
            : '--';
        const confidence = typeof details.confidence === 'number'
            ? `${(details.confidence * 100).toFixed(1)}%`
            : '--';
        const threshold = typeof details.threshold === 'number'
            ? details.threshold.toFixed(2)
            : '--';
        const rowStyle = isCandidate
            ? 'background: rgba(16, 185, 129, 0.08);'
            : 'background: rgba(239, 68, 68, 0.08);';

        return `
            <tr style="${rowStyle}">
                <td style="padding: 6px 8px;">${name}</td>
                <td style="padding: 6px 8px;">${predictionBadge}</td>
                <td style="padding: 6px 8px;">${probability}</td>
                <td style="padding: 6px 8px;">${confidence}</td>
                <td style="padding: 6px 8px;">${threshold}</td>
            </tr>
        `;
    }).join('');

    return `
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="color: #8b95a5; text-align: left;">
                    <th style="padding: 6px 8px;">Model</th>
                    <th style="padding: 6px 8px;">Prediction</th>
                    <th style="padding: 6px 8px;">P(${POSITIVE_LABEL})</th>
                    <th style="padding: 6px 8px;">Confidence</th>
                    <th style="padding: 6px 8px;">Threshold</th>
                </tr>
            </thead>
            <tbody>
                ${rows}
            </tbody>
        </table>
    `;
}

function renderMetricsTable(models) {
    if (!models || models.length === 0) {
        return '<div class="result-box" style="margin-top: 20px;"><p style="color: #8b95a5;">No models trained yet.</p></div>';
    }

    const rows = models.map(model => {
        const accuracy = formatMetric(model.accuracy);
        const precision = formatMetric(model.precision);
        const recall = formatMetric(model.recall);
        const f1 = formatMetric(model.f1_score);
        const threshold = typeof model.threshold === 'number' ? model.threshold.toFixed(2) : '--';
        const updated = model.last_updated || '--';

        return `
            <tr>
                <td style="padding: 6px 8px;">${model.name}</td>
                <td style="padding: 6px 8px;">${accuracy}</td>
                <td style="padding: 6px 8px;">${precision}</td>
                <td style="padding: 6px 8px;">${recall}</td>
                <td style="padding: 6px 8px;">${f1}</td>
                <td style="padding: 6px 8px;">${threshold}</td>
                <td style="padding: 6px 8px;">${updated}</td>
            </tr>
        `;
    }).join('');

    return `
        <div class="result-box" style="margin-top: 20px;">
            <h3 style="color: #667eea; margin-bottom: 10px;">Model Metrics</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="color: #8b95a5; text-align: left;">
                        <th style="padding: 6px 8px;">Model</th>
                        <th style="padding: 6px 8px;">Accuracy</th>
                        <th style="padding: 6px 8px;">Precision</th>
                        <th style="padding: 6px 8px;">Recall</th>
                        <th style="padding: 6px 8px;">F1 Score</th>
                        <th style="padding: 6px 8px;">Threshold</th>
                        <th style="padding: 6px 8px;">Last Updated</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        </div>
    `;
}

function renderModelSummary(models, bestModelName) {
    if (!models || models.length === 0) {
        return '';
    }

    const positiveLabel = typeof POSITIVE_LABEL === 'string' ? POSITIVE_LABEL : 'Positive';
    const positiveDescriptor = positiveLabel.toLowerCase() === 'candidate'
        ? 'True Candidates'
        : `True ${positiveLabel}`;

    const sorted = [...models].sort((a, b) => {
        const aAcc = typeof a.accuracy === 'number' ? a.accuracy : -1;
        const bAcc = typeof b.accuracy === 'number' ? b.accuracy : -1;
        return bAcc - aAcc;
    });

    return sorted.map(model => {
        const accuracy = formatMetric(model.accuracy);
        const precision = formatMetric(model.precision);
        const recall = formatMetric(model.recall);
        const f1 = formatMetric(model.f1_score);
        const threshold = typeof model.threshold === 'number' ? model.threshold.toFixed(2) : '--';
        const lastUpdated = model.last_updated || '--';
        const confusion = model.confusion_matrix || {};
        const tp = confusion.tp ?? '--';
        const fp = confusion.fp ?? '--';
        const fn = confusion.fn ?? '--';
        const isBest = bestModelName && model.name === bestModelName;

        return `
            <article class="model-summary-card${isBest ? ' model-summary-card-best' : ''}">
                <header class="model-summary-header">
                    <div class="model-summary-title">
                        <h4 class="model-summary-name">${model.name}</h4>
                        <span class="model-summary-updated">${lastUpdated}</span>
                    </div>
                    ${isBest ? '<span class="model-summary-badge" title="Highest accuracy in this ensemble">Best</span>' : ''}
                </header>
                <section class="model-summary-body">
                    <div class="model-summary-metric">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">${accuracy}</span>
                    </div>
                    <div class="model-summary-metric">
                        <span class="metric-label">Precision</span>
                        <span class="metric-value">${precision}</span>
                    </div>
                    <div class="model-summary-metric">
                        <span class="metric-label">Recall</span>
                        <span class="metric-value">${recall}</span>
                    </div>
                    <div class="model-summary-metric">
                        <span class="metric-label">F1</span>
                        <span class="metric-value">${f1}</span>
                    </div>
                </section>
                <section class="model-summary-details">
                    <div class="model-summary-detail">
                        <span class="metric-label">Threshold</span>
                        <span class="metric-value">${threshold}</span>
                    </div>
                    <div class="model-summary-detail">
                        <span class="metric-label">${positiveDescriptor}</span>
                        <span class="metric-value">${tp}</span>
                    </div>
                </section>
                <footer class="model-summary-confusion">
                    <span class="confusion-pill confusion-pill-candidate" title="True Candidates">TC ${tp}</span>
                    <span class="confusion-pill confusion-pill-fp" title="False Alarms (Predicted Candidate, actually false)">FA ${fp}</span>
                    <span class="confusion-pill confusion-pill-fn" title="Missed Candidates">MC ${fn}</span>
                </footer>
            </article>
        `;
    }).join('');
}

function populateThresholdControls(stats) {
    const select = document.getElementById('threshold-model');
    const slider = document.getElementById('threshold-slider');
    if (!select || !slider) return;

    const models = stats.models || [];
    if (!models.length) {
        select.innerHTML = '';
        select.disabled = true;
        slider.disabled = true;
        slider.value = 50;
        updateThresholdPreview();
        return;
    }

    select.disabled = false;
    slider.disabled = false;

    const currentValue = select.value;
    select.innerHTML = models.map(model => `<option value="${model.name}">${model.name}</option>`).join('');

    let targetName = currentValue && models.some(model => model.name === currentValue)
        ? currentValue
        : (stats.best_model || models[0].name);

    select.value = targetName;

    const targetModel = models.find(model => model.name === targetName);
    if (targetModel && typeof targetModel.threshold === 'number') {
        slider.value = Math.round(targetModel.threshold * 100);
    } else {
        slider.value = 50;
    }
    updateThresholdPreview();
}

function resetStatsDisplay() {
    document.getElementById('accuracy-stat').textContent = '--';
    document.getElementById('precision-stat').textContent = '--';
    document.getElementById('recall-stat').textContent = '--';
    document.getElementById('f1-stat').textContent = '--';
    document.getElementById('current-model').textContent = 'Not trained yet';
    document.getElementById('current-threshold').textContent = '--';
    document.getElementById('last-updated').textContent = '--';
    document.getElementById('tp-stat').textContent = '--';
    document.getElementById('fp-stat').textContent = '--';
    document.getElementById('fn-stat').textContent = '--';

    const metricsDiv = document.getElementById('model-metrics');
    if (metricsDiv) {
        metricsDiv.innerHTML = '<div class="result-box" style="margin-top: 20px;"><p style="color: #8b95a5;">No models loaded.</p></div>';
    }

    const summaryDiv = document.getElementById('model-summary');
    if (summaryDiv) {
        summaryDiv.innerHTML = '<div class="model-summary-empty">No models loaded.</div>';
    }

    const select = document.getElementById('threshold-model');
    const slider = document.getElementById('threshold-slider');
    if (select) {
        select.innerHTML = '';
        select.disabled = true;
    }
    if (slider) {
        slider.disabled = true;
        slider.value = 50;
        updateThresholdPreview();
    }
}

function syncThresholdFromSelect() {
    if (!cachedModelStats) return;
    const select = document.getElementById('threshold-model');
    const slider = document.getElementById('threshold-slider');
    if (!select || !slider) return;

    const models = cachedModelStats.models || [];
    const targetModel = models.find(model => model.name === select.value);
    if (targetModel && typeof targetModel.threshold === 'number') {
        slider.disabled = false;
        slider.value = Math.round(targetModel.threshold * 100);
    } else {
        slider.value = 50;
    }
    updateThresholdPreview();
}

// Handle CSV file upload
async function handleFileUpload(input) {
    if (input.files.length === 0) return;
    
    const file = input.files[0];
    const statusDiv = document.getElementById('upload-status');
    
    const formData = new FormData();
    formData.append('file', file);
    
    statusDiv.innerHTML = `
        <div class="result-box" style="margin-top: 20px;">
            <p><strong>File:</strong> ${file.name}</p>
            <p><strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB</p>
            <div class="loading"></div> Processing...
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE_URL}/classify/batch`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const modelsSummary = result.models || {};
            const summaryRows = Object.entries(modelsSummary).map(([name, data]) => {
                const threshold = typeof data.threshold === 'number' ? data.threshold.toFixed(2) : '--';
                return `
                    <tr>
                        <td>${name}</td>
                        <td>${data.candidates}</td>
                        <td>${data.likely_false_positives}</td>
                        <td>${threshold}</td>
                    </tr>
                `;
            }).join('');

            const summaryTable = `
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="color: #8b95a5; text-align: left;">
                            <th>Model</th>
                            <th>Candidates</th>
                            <th>Likely False Positives</th>
                            <th>Threshold</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${summaryRows}
                    </tbody>
                </table>
            `;

            statusDiv.innerHTML = `
                <div class="result-box" style="margin-top: 20px;">
                    <p><strong>File:</strong> ${file.name}</p>
                    <p><strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB</p>
                    <div class="progress-bar" style="margin-top: 10px;">
                        <div class="progress-fill" style="width: 100%;"></div>
                    </div>
                    <p style="margin-top: 10px; color: #10b981;">✓ Processing complete!</p>
                    <p style="margin-top: 5px;"><strong>Total Processed:</strong> ${result.total}</p>
                    ${summaryTable}
                    <button class="btn" style="margin-top: 15px;" onclick="downloadResults('${result.result_file}')">Download Results</button>
                </div>
            `;
        } else {
            statusDiv.innerHTML = `
                <div class="result-box result-negative" style="margin-top: 20px;">
                    <p><strong>Error:</strong> ${result.error}</p>
                </div>
            `;
        }
    } catch (error) {
        statusDiv.innerHTML = `
            <div class="result-box result-negative" style="margin-top: 20px;">
                <p><strong>Error:</strong> Could not connect to the server. Make sure the backend is running.</p>
            </div>
        `;
    }
}

// Download batch results
function downloadResults(filename) {
    window.open(`${API_BASE_URL}/download/${filename}`, '_blank');
}

// Load model statistics
async function loadModelStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const stats = await response.json();

        if (response.ok) {
            cachedModelStats = stats;
            const bestMetrics = stats.best_model_metrics || {};

            document.getElementById('accuracy-stat').textContent = formatMetric(bestMetrics.accuracy);
            document.getElementById('precision-stat').textContent = formatMetric(bestMetrics.precision);
            document.getElementById('recall-stat').textContent = formatMetric(bestMetrics.recall);
            document.getElementById('f1-stat').textContent = formatMetric(bestMetrics.f1_score);

            document.getElementById('current-model').textContent = stats.best_model ? stats.best_model : 'Not trained yet';
            document.getElementById('current-threshold').textContent = typeof bestMetrics.threshold === 'number' ? bestMetrics.threshold.toFixed(2) : '--';
            document.getElementById('last-updated').textContent = bestMetrics.last_updated || '--';

            const confusion = bestMetrics.confusion_matrix || {};
            document.getElementById('tp-stat').textContent = confusion.tp ?? '--';
            document.getElementById('fp-stat').textContent = confusion.fp ?? '--';
            document.getElementById('fn-stat').textContent = confusion.fn ?? '--';

            populateThresholdControls(stats);

            const metricsDiv = document.getElementById('model-metrics');
            if (metricsDiv) {
                metricsDiv.innerHTML = renderMetricsTable(stats.models);
            }

            const summaryDiv = document.getElementById('model-summary');
            if (summaryDiv) {
                const summaryMarkup = renderModelSummary(stats.models, stats.best_model);
                summaryDiv.innerHTML = summaryMarkup || '<div class="model-summary-empty">No models loaded.</div>';
            }
        } else {
            cachedModelStats = null;
            resetStatsDisplay();
        }
    } catch (error) {
        cachedModelStats = null;
        console.error('Error loading stats:', error);
        resetStatsDisplay();
    }
}
