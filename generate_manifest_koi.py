from pathlib import Path
import csv
import requests

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def fetch_koi_manifest():
	adql = (
		"SELECT kepid, kepoi_name, kepler_name, koi_disposition "
		"FROM q1_q17_dr25_koi"
	)
	params = {
		"request": "doQuery",
		"lang": "ADQL",
		"format": "json",
		"query": adql,
	}
	r = requests.get(TAP_SYNC_URL, params=params, timeout=300)
	r.raise_for_status()
	return r.json()


def write_manifest(rows, out_path: Path) -> None:
	ensure_dir(out_path.parent)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["kepid", "kepoi_name", "kepler_name", "label"])  # label from koi_disposition
		for r in rows:
			label = r.get("koi_disposition", "").strip()
			w.writerow([r.get("kepid"), r.get("kepoi_name"), r.get("kepler_name"), label])


def main():
	rows = fetch_koi_manifest()
	out = Path("data/labels/koi_manifest.csv")
	write_manifest(rows, out)
	print(f"Saved manifest: {out} ({len(rows)} rows)")


if __name__ == "__main__":
	main()
