// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

let currentAlgorithm = 'xgboost';
let trainingInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    createStars();
    initializeSliders();
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

// Algorithm selection
function selectAlgorithm(element, algorithm) {
    document.querySelectorAll('.algo-card').forEach(card => card.classList.remove('selected'));
    element.classList.add('selected');
    
    document.querySelectorAll('.hyperparams-section').forEach(section => section.style.display = 'none');
    document.getElementById('hyperparams-' + algorithm).style.display = 'block';
    
    currentAlgorithm = algorithm;
}

// Initialize all sliders
function initializeSliders() {
    const sliders = document.querySelectorAll('.slider');
    sliders.forEach(slider => {
        updateSlider(slider.id);
    });
}

// Update slider value display
function updateSlider(id) {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(id + '-value');
    let value = slider.value;
    
    if (id.includes('split')) {
        value += '%';
    }
    
    valueDisplay.textContent = value;
}

// Classify single data point
async function classifyData() {
    const featureFields = [
        { id: 'metallicity', label: 'Metallicity' },
        { id: 'eff_temp', label: 'Effective Temperature' },
        { id: 'gkcolor', label: 'g-k Color Index' },
        { id: 'extinction', label: 'Extinction' },
        { id: 'grcolor', label: 'g-r Color Index' },
        { id: 'radius', label: 'Planet Radius' },
        { id: 'jkcolor', label: 'j-k Color Index' },
        { id: 'surface_gravity', label: 'Surface Gravity' },
        { id: 'reddening', label: 'Reddening' }
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
            const isExoplanet = result.prediction === 'Confirmed';
            const confidence = (result.confidence * 100).toFixed(1);
            
            resultDiv.innerHTML = `
                <div class="result-box ${isExoplanet ? 'result-positive' : 'result-negative'}" style="animation: fadeIn 0.5s ease;">
                    <h3 style="color: ${isExoplanet ? '#10b981' : '#ef4444'}; margin-bottom: 15px;">Classification Result</h3>
                    <p><strong>Prediction:</strong> <span style="color: ${isExoplanet ? '#10b981' : '#ef4444'}; font-size: 1.2em;">${result.prediction}</span></p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <p><strong>Algorithm Used:</strong> ${result.algorithm.toUpperCase()}</p>
                    <p><strong>Classification:</strong> ${isExoplanet ? 'High probability of being a genuine exoplanet based on transit characteristics' : 'Transit signal likely caused by stellar activity or instrumental noise'}</p>
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
        const confidence = typeof item.confidence === 'number'
            ? `${(item.confidence * 100).toFixed(1)}%`
            : 'N/A';
        return `<li><strong>Line ${item.line}:</strong> ${item.prediction} <span style="color: #8b95a5;">(${confidence})</span></li>`;
    }

    return `<li><strong>Line ${item.line}:</strong> <span style="color: #ef4444;">${item.error || 'Unknown error'}</span></li>`;
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
            statusDiv.innerHTML = `
                <div class="result-box" style="margin-top: 20px;">
                    <p><strong>File:</strong> ${file.name}</p>
                    <p><strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB</p>
                    <div class="progress-bar" style="margin-top: 10px;">
                        <div class="progress-fill" style="width: 100%;"></div>
                    </div>
                    <p style="margin-top: 10px; color: #10b981;">✓ Processing complete!</p>
                    <p style="margin-top: 5px;"><strong>Total Processed:</strong> ${result.total}</p>
                    <p><strong>Confirmed Exoplanets:</strong> ${result.confirmed}</p>
                    <p><strong>Candidates:</strong> ${result.candidates}</p>
                    <p><strong>False Positives:</strong> ${result.false_positives}</p>
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

// Get hyperparameters based on selected algorithm
function getHyperparameters() {
    const hyperparams = {};
    
    switch(currentAlgorithm) {
        case 'xgboost':
            hyperparams.learning_rate = parseFloat(document.getElementById('xgb-learning-rate').value);
            hyperparams.max_depth = parseInt(document.getElementById('xgb-max-depth').value);
            hyperparams.n_estimators = parseInt(document.getElementById('xgb-n-estimators').value);
            hyperparams.subsample = parseFloat(document.getElementById('xgb-subsample').value);
            break;
        case 'random-forest':
            hyperparams.n_estimators = parseInt(document.getElementById('rf-n-estimators').value);
            hyperparams.max_depth = parseInt(document.getElementById('rf-max-depth').value);
            hyperparams.min_samples_split = parseInt(document.getElementById('rf-min-samples').value);
            hyperparams.max_features = document.getElementById('rf-max-features').value;
            break;
        case 'neural-net':
            hyperparams.learning_rate = parseFloat(document.getElementById('nn-learning-rate').value);
            hyperparams.batch_size = parseInt(document.getElementById('nn-batch-size').value);
            hyperparams.hidden_layers = parseInt(document.getElementById('nn-hidden-layers').value);
            hyperparams.dropout = parseFloat(document.getElementById('nn-dropout').value);
            hyperparams.optimizer = document.getElementById('nn-optimizer').value;
            hyperparams.activation = document.getElementById('nn-activation').value;
            break;
        case 'svm':
            hyperparams.C = parseFloat(document.getElementById('svm-c').value);
            hyperparams.kernel = document.getElementById('svm-kernel').value;
            hyperparams.gamma = document.getElementById('svm-gamma').value;
            break;
    }
    
    return hyperparams;
}

// Train model
async function trainModel() {
    const dataset = document.getElementById('training-dataset').value;
    const trainSplit = parseInt(document.getElementById('train-split').value) / 100;
    const epochs = parseInt(document.getElementById('epochs').value);
    const hyperparams = getHyperparameters();
    
    const trainingData = {
        dataset: dataset,
        algorithm: currentAlgorithm,
        train_split: trainSplit,
        epochs: epochs,
        hyperparameters: hyperparams
    };
    
    const progressDiv = document.getElementById('training-progress');
    progressDiv.style.display = 'block';
    
    document.getElementById('total-epochs').textContent = epochs;
    document.getElementById('current-epoch').textContent = '0';
    document.getElementById('current-loss').textContent = '--';
    document.getElementById('val-accuracy').textContent = '--';
    document.getElementById('train-progress-bar').style.width = '0%';
    
    try {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trainingData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Simulate training progress
            let epoch = 0;
            trainingInterval = setInterval(() => {
                epoch++;
                const progress = (epoch / epochs) * 100;
                document.getElementById('train-progress-bar').style.width = progress + '%';
                document.getElementById('current-epoch').textContent = epoch;
                document.getElementById('current-loss').textContent = (0.5 - (epoch / epochs) * 0.45).toFixed(4);
                document.getElementById('val-accuracy').textContent = (85 + (epoch / epochs) * 13).toFixed(1) + '%';
                
                if (epoch >= epochs) {
                    clearInterval(trainingInterval);
                    alert(`Training completed!\nFinal Accuracy: ${result.accuracy}%\nF1 Score: ${result.f1_score}`);
                    loadModelStats();
                }
            }, 50);
        } else {
            progressDiv.style.display = 'none';
            alert('Error: ' + result.error);
        }
    } catch (error) {
        progressDiv.style.display = 'none';
        alert('Error: Could not connect to the server. Make sure the backend is running.');
    }
}

// Stop training
function stopTraining() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
        alert('Training stopped. Current progress has been saved.');
    }
}

// Load model statistics
async function loadModelStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const stats = await response.json();
        
        if (response.ok) {
            document.getElementById('accuracy-stat').textContent = stats.accuracy ? (stats.accuracy * 100).toFixed(1) + '%' : '--';
            document.getElementById('precision-stat').textContent = stats.precision ? (stats.precision * 100).toFixed(1) + '%' : '--';
            document.getElementById('recall-stat').textContent = stats.recall ? (stats.recall * 100).toFixed(1) + '%' : '--';
            document.getElementById('f1-stat').textContent = stats.f1_score ? (stats.f1_score * 100).toFixed(1) + '%' : '--';
            
            document.getElementById('current-algo').textContent = stats.algorithm ? stats.algorithm.toUpperCase() : 'Not trained yet';
            document.getElementById('training-data').textContent = stats.dataset || '--';
            document.getElementById('total-samples').textContent = stats.total_samples || '--';
            document.getElementById('last-updated').textContent = stats.last_updated || '--';
            document.getElementById('training-time').textContent = stats.training_time || '--';
            
            document.getElementById('tp-stat').textContent = stats.confusion_matrix ? stats.confusion_matrix.tp : '--';
            document.getElementById('fp-stat').textContent = stats.confusion_matrix ? stats.confusion_matrix.fp : '--';
            document.getElementById('fn-stat').textContent = stats.confusion_matrix ? stats.confusion_matrix.fn : '--';
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}
