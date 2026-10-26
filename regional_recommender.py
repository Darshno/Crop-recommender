"""Leakage-safe historical evidence for regional crop recommendations.

This module intentionally does not merge the regional data with the soil model
features.  Yield units are reported as dataset-dependent because the source
CSV does not document them in this repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Crop", "Crop_Year", "Season", "State", "Area", "Production",
    "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield",
}
NUMERIC_COLUMNS = [
    "Crop_Year", "Area", "Production", "Annual_Rainfall",
    "Fertilizer", "Pesticide", "Yield",
]


def normalize_crop_name(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.replace(r"\s+", " ", regex=True)


@dataclass(frozen=True)
class RegionalDataset:
    data: pd.DataFrame
    report: dict[str, Any]


def load_regional_data(path: str | Path) -> RegionalDataset:
    """Load and validate the regional CSV without silently repairing bad rows."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Regional dataset not found: {source}")
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    report: dict[str, Any] = {
        "source": str(source),
        "input_rows": int(len(frame)),
        "missing_columns": missing,
        "invalid_rows_dropped": 0,
        "issues": [],
        "yield_units": "dataset-dependent; source documentation was not provided",
    }
    if missing:
        raise ValueError("Regional dataset is missing required columns: " + ", ".join(missing))

    frame = frame.copy()
    for column in ("Crop", "State", "Season"):
        frame[column] = _clean_text(frame[column])
    before = len(frame)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric_invalid = ~np.isfinite(frame[NUMERIC_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    empty_text = (frame["Crop"] == "") | (frame["State"] == "") | (frame["Season"] == "")
    invalid = numeric_invalid | empty_text
    if invalid.any():
        report["invalid_rows_dropped"] = int(invalid.sum())
        report["issues"].append(
            f"Dropped {int(invalid.sum())} rows with missing/invalid text or non-finite numeric values."
        )
        frame = frame.loc[~invalid].copy()
    frame["Crop_key"] = frame["Crop"].map(normalize_crop_name)
    frame["State_key"] = frame["State"].map(normalize_crop_name)
    frame["Season_key"] = frame["Season"].map(normalize_crop_name)
    report["output_rows"] = int(len(frame))
    report["numeric_columns_checked"] = list(NUMERIC_COLUMNS)
    report["empty_or_invalid_rows_before_drop"] = int(before - len(frame))
    return RegionalDataset(frame.reset_index(drop=True), report)


def build_historical_evidence(
    data: pd.DataFrame,
    state: str,
    season: str,
    cutoff_year: int,
) -> pd.DataFrame:
    """Aggregate only records strictly before ``cutoff_year``."""
    if not isinstance(cutoff_year, (int, np.integer)):
        raise TypeError("cutoff_year must be an integer")
    required = {"Crop", "Crop_Year", "Season", "State", "Yield"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Evidence data is missing columns: " + ", ".join(sorted(missing)))
    state_key, season_key = normalize_crop_name(state), normalize_crop_name(season)
    work = data.copy()
    for col in ("Crop", "State", "Season"):
        work[f"{col}_key"] = work[col].map(normalize_crop_name)
    work["Crop_Year"] = pd.to_numeric(work["Crop_Year"], errors="coerce")
    work["Yield"] = pd.to_numeric(work["Yield"], errors="coerce")
    mask = (
        (work["State_key"] == state_key)
        & (work["Season_key"] == season_key)
        & (work["Crop_Year"] < cutoff_year)
        & np.isfinite(work["Crop_Year"])
        & np.isfinite(work["Yield"])
    )
    filtered = work.loc[mask]
    if filtered.empty:
        return pd.DataFrame(columns=[
            "Crop", "crop_key", "State", "Season", "median_yield",
            "observations", "last_year", "yield_units",
        ])
    result = (
        filtered.groupby(["Crop_key", "Crop", "State", "Season"], sort=True, as_index=False)
        .agg(median_yield=("Yield", "median"), observations=("Yield", "size"),
             last_year=("Crop_Year", "max"))
        .rename(columns={"Crop_key": "crop_key"})
    )
    result["yield_units"] = "dataset-dependent"
    return result.sort_values(["median_yield", "observations", "crop_key"],
                              ascending=[False, False, True], kind="mergesort").reset_index(drop=True)


def _base_rows(labels: Iterable[str], probabilities: Iterable[float]) -> pd.DataFrame:
    labels = list(labels)
    probabilities = np.asarray(list(probabilities), dtype=float)
    if len(labels) != len(probabilities):
        raise ValueError("base_labels and base_probabilities must have the same length")
    if len(labels) == 0 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Base probabilities must be finite, non-negative, and non-empty")
    return pd.DataFrame({
        "crop": labels,
        "crop_key": [normalize_crop_name(label) for label in labels],
        "base_score": probabilities,
    })


def rank_with_regional_evidence(
    base_probabilities: Iterable[float],
    base_labels: Iterable[str],
    evidence: pd.DataFrame,
    *,
    state: str,
    season: str,
    regional_weight: float = 0.20,
    min_observations: int = 2,
) -> dict[str, Any]:
    """Return transparent base, evidence, and conservative hybrid rankings."""
    if not 0 <= regional_weight <= 1:
        raise ValueError("regional_weight must be between 0 and 1")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    base = _base_rows(base_labels, base_probabilities)
    usable = evidence.copy()
    if usable.empty or not {"crop_key", "median_yield", "observations"}.issubset(usable):
        usable = pd.DataFrame(columns=["crop_key", "median_yield", "observations"])
    usable = usable[usable["observations"] >= min_observations].copy()
    usable["median_yield"] = pd.to_numeric(usable["median_yield"], errors="coerce")
    usable = usable[np.isfinite(usable["median_yield"])]
    if usable.empty:
        base["regional_score"] = 0.0
        base["hybrid_score"] = base["base_score"]
        return {
            "state": state, "season": season, "fallback_used": True,
            "reason": "No sufficiently supported historical evidence for this state/season.",
            "base_ranking": base.sort_values("base_score", ascending=False, kind="mergesort").to_dict("records"),
            "hybrid_ranking": base.sort_values("base_score", ascending=False, kind="mergesort").to_dict("records"),
            "regional_only_candidates": [],
        }
    low, high = usable["median_yield"].min(), usable["median_yield"].max()
    if high == low:
        usable["regional_score"] = 1.0
    else:
        usable["regional_score"] = (usable["median_yield"] - low) / (high - low)
    merged = base.merge(usable[["crop_key", "regional_score"]], on="crop_key", how="left")
    merged["regional_score"] = merged["regional_score"].fillna(0.0)
    base_total = merged["base_score"].sum()
    base_norm = merged["base_score"] / base_total if base_total else merged["base_score"]
    merged["hybrid_score"] = (1 - regional_weight) * base_norm + regional_weight * merged["regional_score"]
    regional_keys = set(usable["crop_key"])
    base_keys = set(base["crop_key"])
    regional_only = usable[~usable["crop_key"].isin(base_keys)].sort_values(
        ["regional_score", "crop_key"], ascending=[False, True], kind="mergesort"
    )
    def ranking(column: str) -> list[dict[str, Any]]:
        return merged.sort_values([column, "crop_key"], ascending=[False, True], kind="mergesort").to_dict("records")
    return {
        "state": state, "season": season, "fallback_used": False,
        "reason": "Historical evidence is shown separately; hybrid weight is configurable.",
        "base_ranking": ranking("base_score"), "hybrid_ranking": ranking("hybrid_score"),
        "regional_only_candidates": regional_only.to_dict("records"),
        "regional_coverage": len(regional_keys & base_keys),
    }
