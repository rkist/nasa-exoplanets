import os
from pathlib import Path
from typing import Tuple

import requests
from dotenv import load_dotenv

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def fetch_csv(adql: str) -> str:
	params = {
		"request": "doQuery",
		"lang": "ADQL",
		"format": "csv",
		"query": adql,
	}
	resp = requests.get(TAP_SYNC_URL, params=params, timeout=120)
	resp.raise_for_status()
	return resp.text


def build_adql(disposition: str) -> str:
	# q1_q17_dr25_koi contains KOIs with dispositions; we select key columns as labels
	# See docs: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
	return (
		"SELECT kepoi_name, kepid, kepler_name, koi_disposition, koi_pdisposition, "
		"koi_period, koi_prad, koi_depth, koi_time0bk "
		"FROM q1_q17_dr25_koi "
		f"WHERE koi_disposition = '{disposition}'"
	)


def main() -> None:
	load_dotenv()
	out_dir = Path(__file__).parent / "data" / "labels"
	ensure_dir(out_dir)

	for disposition, fname in [
		("CONFIRMED", "koi_dr25_confirmed.csv"),
		("FALSE POSITIVE", "koi_dr25_false_positives.csv"),
	]:
		adql = build_adql(disposition)
		csv_text = fetch_csv(adql)
		(out_dir / fname).write_text(csv_text, encoding="utf-8")
		print(f"Saved: {out_dir / fname} ({len(csv_text.splitlines()) - 1} rows)")


if __name__ == "__main__":
	main()
