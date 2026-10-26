"""Chronological evaluation of the regional evidence ranking layer.

The regional CSV has no soil features, so the base comparator is the frozen
model's class prior (uniform here), not a fabricated soil prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from regional_recommender import build_historical_evidence, load_regional_data, rank_with_regional_evidence


def _recall(ranking: list[str], target: str, k: int = 3) -> float:
    return float(target in ranking[:k])


def _ndcg(ranking: list[str], target: str, k: int = 3) -> float:
    try:
        rank = ranking[:k].index(target)
    except ValueError:
        return 0.0
    return 1.0 / np.log2(rank + 2)


def evaluate(data: pd.DataFrame, labels: list[str], weight: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    base_probs = np.ones(len(labels), dtype=float) / len(labels)
    for year in sorted(data["Crop_Year"].astype(int).unique()):
        held_out = data[data["Crop_Year"] == year]
        for (state, season), group in held_out.groupby(["State", "Season"], sort=True):
            evidence = build_historical_evidence(data, state, season, int(year))
            result = rank_with_regional_evidence(
                base_probs, labels, evidence, state=state, season=season,
                regional_weight=weight, min_observations=1,
            )
            base_rank = [x["crop_key"] for x in result["base_ranking"]]
            hybrid_rank = [x["crop_key"] for x in result["hybrid_ranking"]]
            for target in group["Crop"].map(lambda x: " ".join(str(x).strip().casefold().split())):
                rows.append({
                    "year": int(year), "state": state, "season": season,
                    "target_crop": target, "support": int(len(evidence)),
                    "fallback_used": bool(result["fallback_used"]),
                    "base_recall_at_3": _recall(base_rank, target),
                    "hybrid_recall_at_3": _recall(hybrid_rank, target),
                    "base_ndcg_at_3": _ndcg(base_rank, target),
                    "hybrid_ndcg_at_3": _ndcg(hybrid_rank, target),
                })
    details = pd.DataFrame(rows)
    summary = {
        "rows_evaluated": int(len(details)),
        "years": sorted(details["year"].unique().tolist()) if not details.empty else [],
        "base_recall_at_3": float(details["base_recall_at_3"].mean()) if not details.empty else 0.0,
        "hybrid_recall_at_3": float(details["hybrid_recall_at_3"].mean()) if not details.empty else 0.0,
        "base_ndcg_at_3": float(details["base_ndcg_at_3"].mean()) if not details.empty else 0.0,
        "hybrid_ndcg_at_3": float(details["hybrid_ndcg_at_3"].mean()) if not details.empty else 0.0,
        "evidence_coverage": float((~details["fallback_used"]).mean()) if not details.empty else 0.0,
        "interpretation": "The base comparator is uniform because regional rows lack the soil features required by the frozen model.",
    }
    return details, summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "regional" / "crop_yield.csv")
    parser.add_argument("--model", type=Path, default=root / "crop_model.pkl")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "phase4")
    parser.add_argument("--regional-weight", type=float, default=0.20)
    args = parser.parse_args()
    regional = load_regional_data(args.data)
    bundle = joblib.load(args.model)
    labels = [str(label) for label in bundle["encoder"].classes_]
    details, summary = evaluate(regional.data, labels, args.regional_weight)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "chronological_results.csv", index=False)
    if details.empty:
        per_state = pd.DataFrame(columns=[
            "state", "rows_evaluated", "base_recall_at_3",
            "hybrid_recall_at_3", "base_ndcg_at_3", "hybrid_ndcg_at_3",
            "evidence_coverage",
        ])
    else:
        per_state = (
            details.groupby("state", sort=True)
            .agg(
                rows_evaluated=("target_crop", "size"),
                base_recall_at_3=("base_recall_at_3", "mean"),
                hybrid_recall_at_3=("hybrid_recall_at_3", "mean"),
                base_ndcg_at_3=("base_ndcg_at_3", "mean"),
                hybrid_ndcg_at_3=("hybrid_ndcg_at_3", "mean"),
                evidence_coverage=("fallback_used", lambda values: float((~values).mean())),
            )
            .reset_index()
        )
    per_state.to_csv(args.output_dir / "per_state_results.csv", index=False)
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "data_quality.json").write_text(
        json.dumps(regional.report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
