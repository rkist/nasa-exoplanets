import argparse
from pathlib import Path
from typing import Iterable, List

import requests

NSTD_API_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def fetch_kepler_timeseries_summary(kepid: int, quarter: int) -> str:
	params = {
		"table": "keplertimeseries",
		"kepid": str(kepid),
		"quarter": str(quarter),
		"format": "ipac",
	}
	resp = requests.get(NSTD_API_URL, params=params, timeout=120)
	resp.raise_for_status()
	return resp.text


def save_ipac(text: str, path: Path) -> None:
	path.write_text(text, encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description="Download Kepler time series summaries for kepids")
	parser.add_argument("--kepids", type=int, nargs="+", help="List of KIC IDs to download")
	parser.add_argument("--quarter", type=int, default=14, help="Kepler quarter (e.g., 14)")
	parser.add_argument("--out", type=str, default="data/kepler", help="Output directory")
	args = parser.parse_args()

	out_dir = Path(args.out)
	ensure_dir(out_dir)

	for kepid in args.kepids:
		text = fetch_kepler_timeseries_summary(kepid, args.quarter)
		out_path = out_dir / f"kic{kepid}_q{args.quarter}.ipac"
		save_ipac(text, out_path)
		print(f"Saved: {out_path}")


if __name__ == "__main__":
	main()
