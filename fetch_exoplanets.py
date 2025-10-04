import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def build_adql_query() -> str:
	# Pull a small, recent sample of confirmed planets with common fields
	# See: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html (TAP is recommended)
	return (
		"SELECT TOP 20 pl_name, hostname, disc_year, discoverymethod, "
		"pl_orbper, pl_rade, pl_bmasse, pl_orbeccen, st_teff, st_mass, st_rad, sy_dist "
		"FROM ps "
		"WHERE disc_year >= 2020 "
		"ORDER BY disc_year DESC"
	)


def perform_tap_query(query: str, email: Optional[str], token: Optional[str]) -> Dict[str, Any]:
	params = {
		"request": "doQuery",
		"lang": "ADQL",
		"format": "json",
		"query": query,
	}

	auth = (email, token) if email and token else None

	resp = requests.get(TAP_SYNC_URL, params=params, auth=auth, timeout=60)
	resp.raise_for_status()
	return resp.json()


def ensure_data_dir() -> Path:
	data_dir = Path(__file__).parent / "data"
	data_dir.mkdir(parents=True, exist_ok=True)
	return data_dir


def save_json(data: Any, path: Path) -> None:
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_rows(rows: List[Dict[str, Any]]) -> None:
	if not rows:
		print("No rows returned.")
		return

	# Print available columns
	columns: List[str] = list(rows[0].keys())
	print(f"Columns ({len(columns)}): {', '.join(columns)}")

	# Show first few examples
	print("\nSample rows:")
	for row in rows[:5]:
		name = row.get("pl_name")
		year = row.get("disc_year")
		method = row.get("discoverymethod")
		host = row.get("hostname")
		period = row.get("pl_orbper")
		radius = row.get("pl_rade")
		mass_e = row.get("pl_bmasse")
		print(
			f"- {name} (host: {host}) — year: {year}, method: {method}, "
			f"orbital period: {period}, radius(Earth): {radius}, mass(Earth): {mass_e}"
		)


if __name__ == "__main__":
	load_dotenv()

	email = os.getenv("NASA_EXO_EMAIL")
	token = os.getenv("NASA_EXO_TOKEN")

	query = build_adql_query()
	print("Running TAP query...\n")

	try:
		data = perform_tap_query(query, email=email, token=token)
	except requests.HTTPError as e:
		print("HTTP error while querying TAP:", e)
		raise

	# TAP JSON format returns a dict with 'metadata' and 'data' or rows directly, depending on service
	rows: List[Dict[str, Any]]
	if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
		# Some TAP servers return {"metadata": [...], "data": [[...],[...]]}; NASA returns column-name objects in 'rows'
		if data.get("columns") and data.get("rows"):
			rows = data["rows"]  # type: ignore[assignment]
		else:
			# Attempt to map using metadata if present
			metadata = data.get("metadata") or data.get("columns")
			raw_rows = data.get("data")
			if metadata and raw_rows:
				col_names = [c.get("name") for c in metadata]
				rows = [dict(zip(col_names, r)) for r in raw_rows]
			else:
				# Fallback: try 'rows' directly
				rows = data.get("rows", [])  # type: ignore[assignment]
	elif isinstance(data, dict) and data.get("rows"):
		rows = data["rows"]  # type: ignore[assignment]
	elif isinstance(data, list):
		rows = data  # type: ignore[assignment]
	else:
		rows = []

	print(f"Returned rows: {len(rows)}\n")
	summarize_rows(rows)

	# Save raw payload for inspection
	out_dir = ensure_data_dir()
	out_path = out_dir / "latest_ps_sample.json"
	save_json(data, out_path)
	print(f"\nSaved raw JSON to: {out_path}")
