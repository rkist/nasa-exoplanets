# NASA Exoplanets Sample Fetch

This project fetches a small sample of confirmed exoplanets using the Exoplanet Archive TAP service and prints a quick summary.

Reference: [Exoplanet Archive API User Guide](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)

## Setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure credentials (optional; public queries typically work unauthenticated):

- Copy `.env.example` to `.env` and set `NASA_EXO_EMAIL` and `NASA_EXO_TOKEN`.

## Run

```bash
python fetch_exoplanets.py
```

The script will print a column list, a few sample rows, and save the raw JSON to `data/latest_ps_sample.json`.
