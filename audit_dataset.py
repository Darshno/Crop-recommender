"""Run the Phase 2 AgriSense data integrity and leakage audit.

Usage from the project directory:
    python audit_dataset.py
    python audit_dataset.py --data Crop_recommendation.csv --output-dir reports/phase2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.dataset_audit import run_audit


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project_dir / "Crop_recommendation.csv")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "reports" / "phase2")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.data}")
    report = run_audit(args.data, args.output_dir, args.test_size, args.random_state)
    integrity = report["integrity"]
    split = report["train_test_leakage"]
    coverage = report["real_world_coverage_assessment"]

    print("Phase 2 dataset audit complete")
    print(f"Dataset: {report['dataset']['row_count']:,} rows, {report['dataset']['class_count']} classes")
    print(f"Exact duplicate full rows (beyond first): {integrity['exact_duplicate_full_rows_beyond_first']}")
    print(f"Duplicate feature vectors (beyond first): {integrity['exact_duplicate_feature_rows_beyond_first']}")
    print(f"Conflicting identical-feature groups: {integrity['exact_conflicting_feature_group_count']}")
    print(f"Train/test exact feature overlap: {split['exact_feature_vector_overlap_count']}")
    print(f"Train/test exact full-row overlap: {split['exact_full_row_overlap_count']}")
    print(f"Contiguous crop-label blocks: {report['row_order']['label_run_count']}")
    print(f"Real-farm assessment: {coverage['status']}")
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
