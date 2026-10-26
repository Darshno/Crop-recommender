"""Evaluate a frozen AgriSense model on a labelled independent dataset.

Example:
  python evaluate_external.py --data external_field_data.csv --manifest external_manifest.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from evaluation.phase3 import evaluate_external_dataset, write_input_contract

def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Independent CSV with model features and a 'label' column")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON provenance and units manifest")
    parser.add_argument("--model", type=Path, default=project / "crop_model.pkl")
    parser.add_argument("--training-data", type=Path, default=project / "Crop_recommendation.csv")
    parser.add_argument("--output-dir", type=Path, default=project / "reports" / "phase3" / "external_evaluation")
    args = parser.parse_args()
    write_input_contract(project / "reports" / "phase3" / "input_contract.json")
    result = evaluate_external_dataset(args.data, args.manifest, args.model, args.output_dir, args.training_data)
    print("External evaluation completed without fitting or tuning the model.")
    print(f"Rows evaluated: {result['rows']}")
    for name, value in result["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
    print(f"Reports: {result['output_dir']}")

if __name__ == "__main__":
    main()
