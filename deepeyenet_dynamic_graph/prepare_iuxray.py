from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Download and inspect the IU-XRay Kaggle dataset.")
    parser.add_argument("--output-dir", default="outputs/iuxray_prepare")
    parser.add_argument("--dataset", default="raddar/chest-xrays-indiana-university")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install kagglehub first: pip install kagglehub") from exc

    dataset_path = Path(kagglehub.dataset_download(args.dataset))
    reports_path = dataset_path / "indiana_reports.csv"
    images_dir = dataset_path / "images" / "images_normalized"
    if not images_dir.exists():
        images_dir = dataset_path / "images_normalized"
    if not reports_path.exists():
        raise FileNotFoundError(f"Could not find {reports_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find IU-XRay images directory under {dataset_path}")

    reports_df = pd.read_csv(reports_path).copy()
    for col in ["findings", "impression", "indication", "comparison"]:
        if col not in reports_df.columns:
            reports_df[col] = ""
        reports_df[col] = reports_df[col].fillna("")

    pattern = re.compile(r"(\d+)_IM-\d+-\d+\.dcm\.png")
    uid_to_images: dict[int, list[str]] = defaultdict(list)
    for image_file in images_dir.iterdir():
        match = pattern.match(image_file.name)
        if match:
            uid_to_images[int(match.group(1))].append(image_file.name)

    reports_df = reports_df[(reports_df["findings"].str.len() > 0) | (reports_df["impression"].str.len() > 0)]
    reports_df["image_files"] = reports_df["uid"].apply(lambda uid: sorted(uid_to_images.get(int(uid), [])))
    reports_df["num_images"] = reports_df["image_files"].apply(len)
    reports_df = reports_df[reports_df["num_images"] > 0]
    reports_df["image_files_str"] = reports_df["image_files"].apply("|".join)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_df.to_csv(output_dir / "cleaned_reports.csv", index=False)
    with (cache_dir / "uid_to_images.pkl").open("wb") as f:
        pickle.dump(dict(uid_to_images), f)

    print(f"Dataset path: {dataset_path}")
    print(f"Reports: {len(reports_df):,}")
    print(f"Image UID mappings: {len(uid_to_images):,}")
    print(f"Cleaned CSV: {output_dir / 'cleaned_reports.csv'}")
    print("Use this for training:")
    print(f"python -m deepeyenet_dynamic_graph.train --dataset iuxray --data-root {dataset_path}")


if __name__ == "__main__":
    main()
