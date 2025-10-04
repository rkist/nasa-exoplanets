import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
from tqdm import tqdm

NSTD_API_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path, include_labels: List[str]) -> List[int]:
	kepids: List[int] = []
	with path.open("r", encoding="utf-8") as f:
		r = csv.DictReader(f)
		for row in r:
			label = (row.get("label") or "").strip()
			if include_labels and label not in include_labels:
				continue
			try:
				k = int(row["kepid"]) if row["kepid"] else None
			except Exception:
				k = None
			if k:
				kepids.append(k)
	return list(dict.fromkeys(kepids))  # dedupe, preserve order


def parse_quarters(spec: str) -> List[int]:
	parts = []
	for token in spec.split(","):
		token = token.strip()
		if not token:
			continue
		if "-" in token:
			lo, hi = token.split("-", 1)
			parts.extend(range(int(lo), int(hi) + 1))
		else:
			parts.append(int(token))
	return sorted(set(parts))


def fetch_timeseries(kepid: int, quarter: int, session: requests.Session, timeout: int = 60) -> Tuple[int, int, str]:
	params = {
		"table": "keplertimeseries",
		"kepid": str(kepid),
		"quarter": str(quarter),
		"format": "ipac",
	}
	r = session.get(NSTD_API_URL, params=params, timeout=timeout)
	r.raise_for_status()
	return kepid, quarter, r.text


def save_ipac_text(text: str, out_dir: Path, kepid: int, quarter: int) -> Path:
	ensure_dir(out_dir)
	out_path = out_dir / f"kic{kepid}_q{quarter}.ipac"
	out_path.write_text(text, encoding="utf-8")
	return out_path


def main():
	parser = argparse.ArgumentParser(description="Bulk download Kepler time series summaries from KOI manifest")
	parser.add_argument("--manifest", type=str, default="data/labels/koi_manifest.csv")
	parser.add_argument("--labels", type=str, default="CONFIRMED,FALSE POSITIVE", help="Comma-separated labels to include")
	parser.add_argument("--quarters", type=str, default="14", help="Comma- or dash-separated list/range, e.g. 0-17 or 10,11,12")
	parser.add_argument("--limit", type=int, default=100, help="Max number of kepids to download (0 for all)")
	parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
	parser.add_argument("--out", type=str, default="data/kepler", help="Output directory")
	args = parser.parse_args()

	labels = [s.strip() for s in args.labels.split(",") if s.strip()]
	kepids = read_manifest(Path(args.manifest), include_labels=labels)
	if args.limit and args.limit > 0:
		kepids = kepids[: args.limit]
	quarters = parse_quarters(args.quarters)
	out_dir = Path(args.out)

	session = requests.Session()
	futures = []
	with ThreadPoolExecutor(max_workers=args.workers) as ex:
		for k in kepids:
			for q in quarters:
				out_path = out_dir / f"kic{k}_q{q}.ipac"
				if out_path.exists():
					continue
				futures.append(ex.submit(fetch_timeseries, k, q, session))

		progress = tqdm(total=len(futures), desc="Downloading")
		for fut in as_completed(futures):
			try:
				k, q, text = fut.result()
				save_ipac_text(text, out_dir, k, q)
			except Exception:
				pass
			finally:
				progress.update(1)
		progress.close()

	print("Done.")


if __name__ == "__main__":
	main()
