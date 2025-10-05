# ExoDetect AI - NASA Exoplanet Detection Platform

An advanced machine learning platform for detecting and classifying exoplanets from NASA mission data (Kepler, K2, and TESS).

## Prediction

Researchers, scientists, and the general public can leverage our frontend page that connects to our Machine Learning models to make predictions for new data.

### Feature Selection for Exoplanet Detection

The selection of features for our exoplanet detection model is based on the physical principles of the **transit method** and the need to distinguish genuine planetary transits from false positives. Our feature set combines both **planetary characteristics** and **stellar properties** to achieve robust classification.

#### 🌍 Planetary Features

**Orbital Period** (days)
- **Why**: The time it takes for a planet to complete one orbit around its star
- **Importance**: Different orbital periods indicate different types of planets (hot Jupiters vs. Earth-like planets)
- **Detection**: Determines the frequency of transit events
- **Range**: From hours (ultra-hot Jupiters) to years (outer planets)

**Transit Duration** (hours)
- **Why**: The length of time a planet blocks its star's light during transit
- **Importance**: Related to planetary size, orbital velocity, and stellar radius
- **Detection**: Helps validate that the transit is caused by a planetary body
- **Physics**: Longer durations may indicate larger planets or slower orbital velocities

**Planetary Radius** (Earth radii)
- **Why**: The size of the planet relative to Earth
- **Importance**: Directly affects the transit depth (amount of light blocked)
- **Detection**: Larger planets create deeper, more detectable transits
- **Classification**: Helps categorize planet types (super-Earth, Neptune-like, Jupiter-like)

**Equilibrium Temperature** (Kelvin)
- **Why**: The theoretical temperature of the planet based on stellar radiation
- **Importance**: Indicates potential habitability and atmospheric composition
- **Detection**: Correlates with orbital distance and stellar properties
- **Context**: Hot planets close to their stars vs. cold planets far away

#### ⭐ Stellar Features

**Stellar Radius** (Solar radii)
- **Why**: The size of the host star relative to our Sun
- **Importance**: Critical for calculating planetary radius from transit depth
- **Physics**: Transit depth = (R_planet / R_star)²
- **Validation**: Helps identify false positives from stellar variability

**Stellar Mass** (Solar masses)
- **Why**: The mass of the host star relative to our Sun
- **Importance**: Affects orbital dynamics and system stability
- **Detection**: Massive stars produce different signals than small stars
- **Context**: Planet formation and evolution depend on stellar mass

**Metallicity** [Fe/H]
- **Why**: The abundance of elements heavier than hydrogen and helium in the star
- **Importance**: **HIGHEST FEATURE IMPORTANCE** (~13.5% in Random Forest model)
- **Science**: Metal-rich stars are more likely to host planets
- **Detection**: Strong indicator for distinguishing real planets from stellar mimics
- **Research**: Supported by the core accretion theory of planet formation

**Effective Temperature** (Kelvin)
- **Why**: The surface temperature of the host star
- **Importance**: Second-highest feature importance (~13.3%)
- **Detection**: Affects the contrast and detectability of transits
- **Classification**: Different spectral types (hot vs. cool stars) produce different signals
- **Physics**: Determines the habitable zone location

**Color Indices** (gkcolor, grcolor, jkcolor)
- **Why**: Differences in stellar brightness measured at different wavelengths
- **Importance**: Combined importance of ~32.5% across all color features
- **Detection**: Helps identify the stellar spectral type
- **False Positives**: Distinguishes planets from eclipsing binaries and blended stars
- **Multi-wavelength**: Different colors reveal different physical properties

**Extinction** (magnitudes)
- **Why**: Amount of light absorbed by interstellar dust
- **Importance**: ~11% feature importance
- **Detection**: High extinction can create artificial signals
- **Correction**: Essential for accurate photometric measurements
- **Context**: Helps identify observational biases

**Surface Gravity** log(g)
- **Why**: Gravitational acceleration at the stellar surface
- **Importance**: Related to stellar evolution and system geometry
- **Physics**: log(g) = log(M) - 2log(R) + constant
- **Validation**: Helps confirm stellar parameters are consistent with observations

**Reddening** E(B-V)
- **Why**: Measure of color change due to interstellar dust
- **Importance**: Related to extinction, affects apparent stellar colors
- **Correction**: Necessary for accurate stellar characterization
- **Detection**: Helps identify systematic errors in measurements

The model learns the physical relationships between features:
- Planetary radius must be consistent with transit depth and stellar radius
- Orbital period must be consistent with equilibrium temperature
- Stellar colors must be consistent with effective temperature and metallicity

Our Random Forest model reveals that **stellar characteristics dominate**:
- Top 2 features are stellar (metallicity and effective temperature)
- Color indices account for ~32% of total importance
- This validates that context matters as much as the transit signal itself

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

Here’s the **README section** you can drop into your project to document how to train your machine learning models — clear, production-grade, and matching exactly your command and outputs 👇

---

## 🚀 Training Machine Learning Models

The `ml/train_models.py` script trains and evaluates multiple classifiers (RandomForest, XGBoost, LightGBM, and CatBoost) on the Kepler exoplanet dataset.
It handles automatic scaling, class filtering, threshold optimization, and saves both the model and a detailed `.info` report for each run.

### 🔧 Command Example

```bash
python ml/train_models.py \
  --data data/frames/kepler_summary_with_labels.jsonl \
  --target label \
  --classes 'CONFIRMED','FALSE POSITIVE' \
  --test_size 0.2 \
  --out ml_models \
  --feature eff_temp surface_gravity metallicity radius reddening extinction gkcolor grcolor jkcolor \
  --pos_label 'FALSE POSITIVE'
```

### 🧩 Arguments

| Argument      | Required      | Description                                                                             |
| ------------- | ------------- | --------------------------------------------------------------------------------------- |
| `--data`      | ✅             | Path to dataset (`.csv`, `.jsonl`, or `.parquet`).                                      |
| `--target`    | ✅             | Target column name (e.g., `label`).                                                     |
| `--classes`   | ✅             | Comma-separated list of two class names to keep (e.g., `'CONFIRMED','FALSE POSITIVE'`). |
| `--pos_label` | ✅             | Which class to treat as the *positive* label for recall/F1 metrics.                     |
| `--feature`   | optional      | List of feature columns to use. Defaults to all columns except the target.              |
| `--out`       | optional      | Directory to save trained models and reports (default: `models_out`).                   |
| `--test_size` | optional      | Proportion of data used for testing (default: `0.2`).                                   |
| `--scale`     | optional flag | Apply `StandardScaler` normalization to features before training.                       |

---

### 🧠 What the Script Does

1. **Loads the dataset** (CSV, Parquet, or JSONL).
2. **Filters** the classes according to `--classes`.
3. **Encodes** the positive label (`--pos_label`) as `1`, others as `0`.
4. **Splits** the dataset into training and testing subsets.
5. **Trains** four models:

   * RandomForest
   * XGBoost
   * LightGBM
   * CatBoost
6. **Evaluates** each model:

   * Accuracy, Recall, Precision, and F1-score
   * Confusion matrix
   * Automatic threshold optimization for both positive and negative classes
7. **Saves**:

   * `*_model.pkl` → trained model
   * `*_scaler.pkl` → optional scaler (if `--scale` used)
   * `*.info` → markdown report with metrics, confusion matrices, and threshold tables

---

### 📄 Example Output

```
📂 Loading dataset from data/frames/kepler_summary_with_labels.jsonl
✅ Dataset loaded: 50649 rows, 59 columns
⚠️ Using subset of classes: ['CONFIRMED', 'FALSE POSITIVE']

📊 Class distribution after filtering:
label
FALSE POSITIVE    0.68
CONFIRMED         0.32

✅ Using 'FALSE POSITIVE' as positive class (1), all others as 0

Training RandomForest...
📊 RandomForest Evaluation:
Accuracy: 0.9998 | Recall: 0.9996
Confusion Matrix:
 [[2678    0]
 [   2 5681]]
💾 Model saved to ml_models/RandomForest_model.pkl
Report saved → ml_models/RandomForest.info
...
🏁 Final Model Results:
          Model  Accuracy    Recall
1       XGBoost  0.999880  0.999824
0  RandomForest  0.999761  0.999648
2      LightGBM  0.999641  0.999472
3      CatBoost  0.999641  0.999472
```

---

### 🧾 Output Files (per model)

Each model produces:

```
ml_models/
├── RandomForest_model.pkl
├── RandomForest_scaler.pkl
├── RandomForest.info
├── XGBoost_model.pkl
├── XGBoost.info
├── LightGBM_model.pkl
├── LightGBM.info
├── CatBoost_model.pkl
├── CatBoost.info
```

The `.info` files contain full evaluation results, including the optimal thresholds for both positive and negative classes and top threshold performance tables.

---

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
