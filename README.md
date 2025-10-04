# ExoDetect AI - NASA Exoplanet Detection Platform

An advanced machine learning platform for detecting and classifying exoplanets from NASA mission data (Kepler, K2, and TESS).

## 🌟 Features

- **Multiple ML Algorithms**: XGBoost, Random Forest, Neural Networks, and SVM
- **Real-time Classification**: Classify individual exoplanet candidates
- **Batch Processing**: Upload CSV files for bulk classification
- **Custom Training**: Train models with custom hyperparameters
- **Performance Metrics**: View detailed model statistics and accuracy
- **Beautiful UI**: Modern, space-themed interface

## 📁 Project Structure

```
exoplanet-detection/
├── frontend/
│   ├── index.html         # Main HTML file
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── main.js        # Frontend JavaScript
├── backend/
│   ├── app.py             # Flask server
│   ├── models.py          # ML model implementations
│   ├── data_processing.py # Data preprocessing
│   ├── requirements.txt   # Python dependencies
│   ├── uploads/           # Uploaded CSV files (created automatically)
│   ├── results/           # Classification results (created automatically)
│   └── saved_models/      # Trained models (created automatically)
├── README.md
└── QUICKSTART.md
```

## 🚀 Setup Instructions

### Backend Setup

1. **Install Python dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Run the Flask server**:
```bash
python app.py
```

The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Serve the frontend** using any web server:

**Option 1: Python HTTP Server**
```bash
cd frontend
python -m http.server 8000
```

**Option 2: Node.js HTTP Server**
```bash
cd frontend
npx http-server -p 8000
```

**Option 3: VS Code Live Server**
- Install "Live Server" extension
- Right-click on `frontend/index.html` → "Open with Live Server"

2. **Access the application**:
Open your browser and go to `http://localhost:8000`

## 📊 Usage

### 1. Training a Model

1. Navigate to the **Train Model** tab
2. Select a dataset (Kepler, K2, TESS, or Combined)
3. Choose an algorithm (XGBoost, Random Forest, Neural Network, or SVM)
4. Adjust hyperparameters as needed
5. Set training/validation split and epochs
6. Click **Start Training**

### 2. Classifying Data

#### Single Classification
1. Go to the **Classify Data** tab
2. Enter planetary parameters:
   - Orbital Period (days)
   - Transit Duration (hours)
   - Planetary Radius (Earth radii)
   - Equilibrium Temperature (K)
   - Stellar Radius (Solar radii)
   - Stellar Mass (Solar masses)
3. Click **Classify Exoplanet**

#### Batch Classification
1. Prepare a CSV file with the required columns
2. Upload the CSV file
3. View results and download the classification report

### 3. View Statistics

- Navigate to the **Model Statistics** tab
- View accuracy, precision, recall, and F1 score
- Check confusion matrix
- See model information and training details

## 📝 CSV Format

Your CSV file should contain the following columns:
```csv
orbital_period,transit_duration,planetary_radius,eq_temp,stellar_radius,stellar_mass
3.52,2.7,1.2,288,1.0,1.0
10.5,4.1,2.3,450,1.2,1.1
...
```

## 🔬 Algorithms

### XGBoost
- **Best for**: High accuracy, fast training
- **Hyperparameters**: Learning rate, max depth, n_estimators, subsample

### Random Forest
- **Best for**: Interpretability, robustness
- **Hyperparameters**: N_estimators, max depth, min samples split

### Neural Network
- **Best for**: Complex patterns, large datasets
- **Hyperparameters**: Learning rate, batch size, hidden layers, dropout, optimizer

### SVM
- **Best for**: Small to medium datasets
- **Hyperparameters**: C, kernel, gamma

## 🌐 API Endpoints

### Classification
- `POST /api/classify` - Classify single sample
- `POST /api/classify/batch` - Classify batch from CSV
- `GET /api/download/<filename>` - Download results

### Training
- `POST /api/train` - Train new model
- `GET /api/stats` - Get model statistics
- `GET /api/health` - Health check

## 📚 Data Sources

The platform supports data from:
- **Kepler Mission**: NASA's Kepler Objects of Interest (KOI)
- **K2 Mission**: Extended Kepler mission data
- **TESS Mission**: Transiting Exoplanet Survey Satellite TOI
- **Custom Data**: Upload your own CSV files

## 🧰 CLI Data Pipeline (NASA Archive)

These scripts fetch NASA Exoplanet Archive data and prepare training frames. Reference: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html

1) Create environment and install deps (repo root):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Generate KOI labels and manifest:
```bash
python download_koi_labels.py
python generate_manifest_koi.py  # writes data/labels/koi_manifest.csv
```

3) Bulk download Kepler time-series summaries (quarter 14 example):
```bash
python bulk_download_keplertimeseries.py \
  --manifest data/labels/koi_manifest.csv \
  --labels "CONFIRMED,FALSE POSITIVE,CANDIDATE" \
  --quarters 14 \
  --limit 1000 \
  --workers 12 \
  --out data/kepler
```

4) Parse to pandas-friendly frames (Parquet/CSV):
```bash
python parse_to_pandas.py \
  --ipac-dir data/kepler \
  --labels data/labels/koi_manifest.csv \
  --out data/frames \
  --workers 12 \
  --batch-size 1500
```

Outputs: `data/frames/kepler_summary_with_labels.parquet` (tracked), CSV is ignored in git.

## 🤖 FT-Transformer Training (Tabular)

Supervised FT-Transformer-style trainer on the merged Parquet. Inspired by: https://gist.github.com/fabriciocarraro/66b878a798630502d8684d7ce4349236

Quick start (subset):
```bash
python train_tabular_transformer.py \
  --data data/frames/kepler_summary_with_labels.parquet \
  --epochs 5 --batch_size 512 --embed_dim 64 --heads 4 --layers 3 \
  --lr 1e-3 --sample_frac 0.2 --out models
```

Full run (larger model):
```bash
python train_tabular_transformer.py \
  --data data/frames/kepler_summary_with_labels.parquet \
  --epochs 30 --batch_size 128 --embed_dim 128 --heads 8 --layers 4 \
  --dropout 0.1 --lr 3e-4 --sample_frac 1.0 --out models
```

Parameters:
- `--data`: Parquet with features + `label`
- `--out`: Output dir for `tabular_transformer.pt` and `feature_config.json`
- `--epochs`: Training epochs
- `--batch_size`: Batch size (tune for memory)
- `--embed_dim`: Embedding size (divisible by `--heads`)
- `--heads`: Attention heads per layer
- `--layers`: Transformer encoder layers
- `--dropout`: Dropout probability
- `--lr`: AdamW learning rate
- `--sample_frac`: Fraction of data to use (1.0 = all)

## 🛠️ Technologies

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- Modern responsive design
- Animated star field background

### Backend
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Random Forest, SVM, preprocessing
- **XGBoost**: Gradient boosting
- **TensorFlow/Keras**: Neural networks

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🌟 Acknowledgments

- NASA Exoplanet Archive for providing open-source datasets
- The Kepler, K2, and TESS mission teams
- All contributors to the open-source ML libraries used in this project

---

**Made with ❤️ for space exploration and exoplanet discovery**