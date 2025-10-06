# ExoDetect AI – NASA Exoplanet Detection Platform

ExoDetect AI is an exoplanet discovery pipeline that turns NASA’s Kepler photometry into actionable candidate rankings.

We reimagined the classification stack around an FT-Transformer trained with self-supervised objectives on millions of stellar light-curve summaries, learning feature embeddings before any labeled supervision. This representation pretraining captures subtle photometric correlations—color indices, extinction, metallicity—that classical tabular models often miss, enabling the transformer to flag rare but promising planetary signatures with higher recall.

When the self-supervised backbone is fine-tuned on curated confirmation labels and deployed alongside our legacy tree-based ensemble, it consistently surfaces additional high-confidence candidates while maintaining a low false-positive rate.

By packaging the model inside an accessible web API and UI, we equip mission scientists and citizen researchers to score new stars instantly, prioritize follow-up observations, and accelerate the validation cycle for small, low-signal planets that traditional pipelines overlook.

## Highlights
- Five production-ready classifiers loaded by default: RandomForest, XGBoost, LightGBM, CatBoost, and the new RankingTransformer FT-Transformer (`models/ranking/v0.3-selected_cols`).
- Consistent preprocessing pipeline that normalizes nine stellar/photometric features before inference.
- Batch (CSV/JSONL) and single-record prediction flows, plus on-demand model statistics and threshold tuning from the UI.
- Docker Compose stack that serves the API on `http://localhost:3000/api` and the static frontend on `http://localhost:8080`.

## Feature Inputs
All models expect the same nine features. Values can be supplied via the UI, JSON, or CSV uploads; missing entries are imputed with training-set medians.

| Feature | Description |
| --- | --- |
| `eff_temp` | Stellar effective temperature (K) |
| `surface_gravity` | log(g) in cgs units |
| `metallicity` | [Fe/H] abundance ratio |
| `radius` | Stellar radius (solar units) |
| `reddening` | Color excess E(B−V) |
| `extinction` | Extinction magnitude A(V) |
| `gkcolor` | g−K color index |
| `grcolor` | g−r color index |
| `jkcolor` | J−K color index |

## Repository Layout
```
.
├── backend/
│   └── api/
│       ├── app.py                 # Flask entrypoint
│       ├── data_processing.py     # Shared preprocessing pipeline
│       ├── model_manager.py       # Loads sklearn and transformer models
│       ├── ml/                    # Training utilities (optional)
│       └── saved_models/          # Pre-trained artifacts (mounted in Docker)
├── frontend/
│   ├── Dockerfile
│   ├── entrypoint.sh              # Writes runtime config.js
│   └── js/main.js                 # SPA logic for predictions/statistics
├── docker-compose.yml             # Frontend + backend stack
├── models/                        # Research/training outputs
└── requirements.txt               # Top-level Python tooling
```

## Run With Docker
1. Install Docker Desktop (Compose v2+).
2. From the repository root, build the images:
   ```bash
   docker compose build
   ```
3. Start the stack:
   ```bash
   docker compose up -d
   ```
4. Open `http://localhost:8080` for the UI. The API is available at `http://localhost:3000/api`.
5. Inspect logs if needed:
   ```bash
   docker compose logs backend
   docker compose logs frontend
   ```
6. Shut down with `docker compose down` (add `--volumes` to remove persisted uploads/results).

Set `API_BASE_URL` before `docker compose up` to point the frontend at a non-default API endpoint.

## Run Without Docker (Development)
```bash
# Backend
cd backend/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app.py run --port 3000

# Frontend (static)
cd frontend
python -m http.server 8080
```
Ensure environment variables mirror production defaults (`FLASK_ENV=production`, etc.) when testing.

## Using the Platform
- **Single prediction**: Fill the nine feature sliders/inputs and submit. The response lists each model’s label (`Candidate` or `Likely False Positive`), probability, and threshold.
- **JSONL / JSON batch**: Paste JSON Lines or arrays into the batch form. The backend returns per-line predictions or validation errors.
- **CSV upload**: Upload a CSV containing the nine feature columns. The `/api/classify/batch` endpoint computes counts per model and provides downloadable results.
- **Model statistics**: The stats tab queries `/api/model/stats`, exposing accuracy, precision, recall, F1, thresholds, and best-model selection.
- **Threshold tuning**: Adjust thresholds via the UI; the backend persists them in-memory for the current session.

## Model Artifacts
`backend/api/saved_models/` ships with:
- `RandomForest_model.pkl` (+ `RandomForest.info`)
- `XGBoost_model.pkl`
- `LightGBM_model.pkl`
- `CatBoost_model.pkl`
- `RankingTransformer_model.pt` + `RankingTransformer_config.json`

The transformer wrapper (`model_manager.py`) normalizes inputs using the stored statistics before running inference with torch.

## Training (Optional)
The repository keeps historical training scripts under `backend/api/ml/` and research notebooks under `models/` and `notebook*.ipynb`. These are not required to run the stack but document how the shipped artifacts were produced.

## Support
For issues or enhancement ideas, open an issue or submit a pull request with reproducible steps. Contributions that improve data ingestion, model coverage, or UI diagnostics are welcome.
