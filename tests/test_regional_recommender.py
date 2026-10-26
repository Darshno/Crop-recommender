import numpy as np
import pandas as pd

from regional_recommender import (
    build_historical_evidence,
    load_regional_data,
    rank_with_regional_evidence,
)


def rows():
    return pd.DataFrame([
        [" Rice ", 2018, " Kharif  ", " Karnataka ", 1, 10, 1, 1, 1, 2],
        ["Rice", 2020, "Kharif", "Karnataka", 1, 12, 1, 1, 1, 4],
        ["Maize", 2019, "Kharif", "Karnataka", 1, 20, 1, 1, 1, 3],
        ["Sugarcane", 2019, "Kharif", "Karnataka", 1, 30, 1, 1, 1, 8],
        ["Rice", 2019, "Rabi", "Karnataka", 1, 2, 1, 1, 1, 99],
    ], columns=["Crop", "Crop_Year", "Season", "State", "Area", "Production",
                "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"])


def test_loader_normalizes_text_and_reports_invalid_values(tmp_path):
    frame = rows()
    frame.loc[len(frame)] = ["bad", 2019, "Kharif", "Karnataka", 1, np.inf, 1, 1, 1, 2]
    path = tmp_path / "regional.csv"
    frame.to_csv(path, index=False)
    loaded = load_regional_data(path)
    assert loaded.data.loc[0, "State"] == "Karnataka"
    assert loaded.data.loc[0, "Season"] == "Kharif"
    assert loaded.report["invalid_rows_dropped"] == 1


def test_evidence_filters_state_season_and_strict_year_cutoff():
    evidence = build_historical_evidence(rows(), "karnataka", "kharif", 2020)
    assert set(evidence["crop_key"]) == {"rice", "maize", "sugarcane"}
    assert evidence.loc[evidence["crop_key"] == "rice", "median_yield"].iloc[0] == 2
    assert build_historical_evidence(rows(), "Karnataka", "Rabi", 2020).iloc[0]["crop_key"] == "rice"


def test_unknown_and_regional_only_crops_are_not_claimed_by_base_model():
    evidence = build_historical_evidence(rows(), "Karnataka", "Kharif", 2021)
    result = rank_with_regional_evidence(
        [0.8, 0.2], ["rice", "maize"], evidence,
        state="Karnataka", season="Kharif", min_observations=1,
    )
    assert [row["crop_key"] for row in result["regional_only_candidates"]] == ["sugarcane"]
    assert {row["crop_key"] for row in result["hybrid_ranking"]} == {"rice", "maize"}


def test_sparse_evidence_falls_back_to_deterministic_base_ranking():
    evidence = build_historical_evidence(rows(), "Unknown", "Kharif", 2021)
    result = rank_with_regional_evidence(
        [0.5, 0.5], ["maize", "rice"], evidence,
        state="Unknown", season="Kharif",
    )
    assert result["fallback_used"] is True
    assert [row["crop_key"] for row in result["hybrid_ranking"]] == ["maize", "rice"]


def test_invalid_numeric_values_are_dropped_not_used(tmp_path):
    frame = rows()
    frame["Yield"] = frame["Yield"].astype("object")
    frame.loc[0, "Yield"] = "not-a-number"
    path = tmp_path / "regional.csv"
    frame.to_csv(path, index=False)
    loaded = load_regional_data(path)
    assert len(loaded.data) == len(frame) - 1
