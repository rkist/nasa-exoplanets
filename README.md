# NASA Exoplanets Data Toolkit

Utilities to download NASA Exoplanet Archive data, generate manifests, bulk-download Kepler time series, and parse outputs into pandas-friendly Parquet/CSV for local modeling.

Reference: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html

## Setup

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Credentials (optional; most queries are public):
- Create `.env` with:
  - `NASA_EXO_EMAIL=...`
  - `NASA_EXO_TOKEN=...`

## Quick sample (confirmed planets)

Fetch a small recent sample from TAP and print a summary; saves JSON to `data/latest_ps_sample.json`.

```bash
python fetch_exoplanets.py
```

## KOI labels (confirmed / candidates / false positives)

Download DR25 KOI labels to CSVs:

```bash
python download_koi_labels.py
```

Generate a unified manifest with `kepid`, KOI fields, and `label`:

```bash
python generate_manifest_koi.py
# outputs: data/labels/koi_manifest.csv
```

## Bulk download Kepler time series (summaries)

Parallel downloader that reads the KOI manifest, filters by labels, and fetches IPAC summary files per `kepid` and quarter.

Examples:
- Confirmed + false positives for quarter 14 (first 100 IDs):
```bash
python bulk_download_keplertimeseries.py \
  --manifest data/labels/koi_manifest.csv \
  --labels "CONFIRMED,FALSE POSITIVE" \
  --quarters 14 \
  --limit 100 \
  --workers 8 \
  --out data/kepler
```
- All confirmed across quarters 0–17, first 2000 IDs:
```bash
python bulk_download_keplertimeseries.py \
  --manifest data/labels/koi_manifest.csv \
  --labels CONFIRMED \
  --quarters 0-17 \
  --limit 2000 \
  --workers 12
```

Notes:
- Valid labels in the manifest: `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`.
- Valid Kepler quarters: `0–17`.

## Parse to pandas-friendly frames

Convert downloaded IPAC summaries to DataFrames and write Parquet/CSV incrementally (parallel, batched). Also saves a merged view with KOI labels.

```bash
python parse_to_pandas.py \
  --ipac-dir data/kepler \
  --labels data/labels/koi_manifest.csv \
  --out data/frames \
  --workers 12 \
  --batch-size 1500
```

Outputs (in `data/frames/`):
- `kepler_summary_with_labels.parquet` (tracked)
- `kepler_summary_with_labels.csv` (generated locally; ignored in git)
- `koi_manifest.parquet`
- Optionally `kepler_timeseries_summary.*` if you run earlier single-pass parser path


## Train a model (FT-Transformer)

Supervised FT-Transformer-style trainer for the merged Parquet. Auto-detects numeric/categorical features and encodes the `label` column (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`). Inspired by an FT-Transformer tabular approach [link](https://gist.github.com/fabriciocarraro/66b878a798630502d8684d7ce4349236).

Quick start (uses a subset via `--sample_frac` for speed):

```bash
python train_tabular_transformer.py \
  --data data/frames/kepler_summary_with_labels.parquet \
  --epochs 5 --batch_size 512 --embed_dim 64 --heads 4 --layers 3 \
  --lr 1e-3 --sample_frac 0.2 --out models
```

Artifacts:
- `models/tabular_transformer.pt`
- `models/feature_config.json` (feature lists, categorical mappings, label classes)

Notes:
- GPU is used if available (PyTorch). CPU also works.
- Class imbalance is significant; consider tuning `--sample_frac`, model depth, and learning rate. You can also resample the training set upstream if needed.
- CSV outputs are large; Parquet is the tracked artifact. Use Git LFS if you need to version CSVs.

