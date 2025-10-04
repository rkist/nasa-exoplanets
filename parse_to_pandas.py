import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def parse_ipac_row(path: Path) -> Optional[Dict]:
	# Read IPAC (single-row) using astropy
	try:
		tab = Table.read(str(path), format="ipac")
		df = tab.to_pandas()
		if df.empty:
			return None
		row = df.iloc[0].to_dict()
		m = re.search(r"kic(\d+)_q(\d+)\.ipac$", path.name)
		if m:
			row["kepid"] = int(m.group(1))
			row["quarter"] = int(m.group(2))
		else:
			row["kepid"] = None
			row["quarter"] = None
		return row
	except Exception:
		return None


def load_label_map(labels_manifest: Path) -> Dict[int, str]:
	label_by_kepid: Dict[int, str] = {}
	if labels_manifest.exists():
		labels_df = pd.read_csv(labels_manifest)
		for _, r in labels_df.iterrows():
			k = r.get("kepid")
			if pd.notna(k):
				label_by_kepid[int(k)] = str(r.get("label", ""))
	return label_by_kepid


def main() -> None:
	parser = argparse.ArgumentParser(description="Parse IPAC Kepler summaries to pandas-friendly Parquet/CSV (parallel, batched)")
	parser.add_argument("--ipac-dir", type=str, default="data/kepler")
	parser.add_argument("--labels", type=str, default="data/labels/koi_manifest.csv")
	parser.add_argument("--out", type=str, default="data/frames")
	parser.add_argument("--workers", type=int, default=12)
	parser.add_argument("--batch-size", type=int, default=1000)
	args = parser.parse_args()

	ipac_dir = Path(args.ipac_dir)
	out_dir = Path(args.out)
	ensure_dir(out_dir)

	csv_out = out_dir / "kepler_summary_with_labels.csv"
	parquet_out = out_dir / "kepler_summary_with_labels.parquet"
	labels_manifest = Path(args.labels)

	print("Building file list ...")
	paths: List[Path] = sorted(ipac_dir.glob("*.ipac"))
	print(f"Found {len(paths)} IPAC files")

	label_by_kepid = load_label_map(labels_manifest)

	# Prepare writers
	csv_header_written = False
	parquet_writer: Optional[pq.ParquetWriter] = None

	def flush_batch(batch_rows: List[Dict]):
		nonlocal csv_header_written, parquet_writer
		if not batch_rows:
			return
		df = pd.DataFrame(batch_rows)
		# CSV append
		mode = "a"
		header = not csv_header_written
		df.to_csv(csv_out, mode=mode, index=False, header=header)
		csv_header_written = True
		# Parquet append via ParquetWriter for efficiency
		table = pa.Table.from_pandas(df, preserve_index=False)
		if parquet_writer is None:
			parquet_writer = pq.ParquetWriter(parquet_out, table.schema)
		parquet_writer.write_table(table)

	print("Parsing and writing (parallel) ...")
	batch: List[Dict] = []
	with ThreadPoolExecutor(max_workers=args.workers) as ex:
		futs = {ex.submit(parse_ipac_row, p): p for p in paths}
		completed = 0
		for fut in as_completed(futs):
			row = fut.result()
			if row is not None:
				k = row.get("kepid")
				if isinstance(k, int):
					row["label"] = label_by_kepid.get(k)
				batch.append(row)
			if len(batch) >= args.batch_size:
				flush_batch(batch)
				batch.clear()
			completed += 1
			if completed % (args.batch_size // 5 or 1) == 0:
				print(f"Processed {completed}/{len(paths)} files ...")

	# Flush remaining
	flush_batch(batch)
	if parquet_writer is not None:
		parquet_writer.close()

	print(f"Done. Wrote: {csv_out} and {parquet_out}")


if __name__ == "__main__":
	main()
