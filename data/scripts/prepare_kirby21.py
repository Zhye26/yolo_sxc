"""Kirby21 data preparation: collect NIfTI files and split into train/val/test."""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _discover_nifti(source: Path) -> List[Path]:
    nii = list(source.rglob("*.nii"))
    nii_gz = list(source.rglob("*.nii.gz"))
    return sorted(nii + nii_gz)


def _extract_subject_id(path: Path) -> str:
    """Map session number to real subject ID (sessions 1-21 = scan1, 22-42 = scan2)."""
    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    m = re.search(r"KKI2009[-_]?(\d+)", stem, re.IGNORECASE)
    if m:
        session = int(m.group(1))
        subj = session if session <= 21 else session - 21
        return f"KKI2009-{subj:02d}"
    m = re.match(r"(\d+)", stem)
    if m:
        return m.group(1)
    if path.parent.name and path.parent.name not in (".", ""):
        return path.parent.name
    return stem


def group_by_subject(paths: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        sid = _extract_subject_id(p)
        groups[sid].append(p)
    return dict(groups)


def split_subjects(
    subjects: Sequence[str],
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    ids = list(subjects)
    random.seed(seed)
    random.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def prepare(source_dir: str, output_dir: str, seed: int = 42) -> None:
    src = Path(source_dir)
    dst = Path(output_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    files = _discover_nifti(src)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {src}")

    groups = group_by_subject(files)
    print(f"Found {len(files)} NIfTI files from {len(groups)} subjects")

    splits = split_subjects(list(groups.keys()), seed=seed)
    for split_name, subject_ids in splits.items():
        split_dir = dst / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for sid in subject_ids:
            for f in groups[sid]:
                dest = split_dir / f.name
                shutil.copy2(f, dest)
                print(f"  [{split_name}] {f.name}")
        print(f"{split_name}: {len(subject_ids)} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Kirby21 dataset")
    parser.add_argument("--source", required=True, help="Raw NIfTI source directory")
    parser.add_argument("--output", default="./data/kirby21", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
