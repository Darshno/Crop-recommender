from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def test_existing_experiment_model_contract_matches_evaluation_schema():
    baseline = joblib.load(ROOT / "crop_model.pkl")
    experimental = joblib.load(
        ROOT / "experiments" / "independent_data_v1" / "crop_model_experimental_v1.pkl"
    )
    eligible = pd.read_csv(
        ROOT / "experiments" / "independent_data_v1" / "eligible_evaluation.csv"
    )
    assert baseline["features"] == FEATURES
    assert experimental["features"] == FEATURES
    independent_labels = set(pd.read_csv(ROOT / "independent_data.csv")["label"].str.strip().str.casefold())
    assert independent_labels <= set(experimental["encoder"].classes_)
    assert list(eligible[FEATURES].columns) == FEATURES
    assert set(eligible["label"]) <= set(baseline["encoder"].classes_)
