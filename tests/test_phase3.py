import json
import pytest

from evaluation.phase3 import validate_provenance_manifest, input_contract, count_exact_feature_overlap
from evaluation.validation import FEATURES, TARGET
import pandas as pd


def valid_manifest():
    return {
        "source_name": "Independent field study",
        "source_reference": "Study protocol v1",
        "collection_method": "Soil lab + field records",
        "geography": "Example district",
        "observation_period": "2025-06 to 2025-11",
        "label_definition": "Recorded planted crop; outcome not assumed",
        "independent_of_training_data": True,
        "units": {
            "N": "documented unit", "P": "documented unit", "K": "documented unit",
            "temperature": "C", "humidity": "%", "ph": "dimensionless", "rainfall": "mm per season"
        }
    }


def test_complete_manifest_passes():
    assert validate_provenance_manifest(valid_manifest()) == []


def test_manifest_requires_independence():
    manifest = valid_manifest()
    manifest["independent_of_training_data"] = False
    assert any("must be true" in e for e in validate_provenance_manifest(manifest))


def test_manifest_requires_units_for_all_features():
    manifest = valid_manifest()
    del manifest["units"]["rainfall"]
    assert any("rainfall" in e for e in validate_provenance_manifest(manifest))


def test_input_contract_flags_unresolved_semantics():
    contract = input_contract()
    assert "UNRESOLVED" in contract["status"]
    assert "divided by 3" in contract["known_app_behavior"]["rainfall"]


def test_external_feature_overlap_detects_reordered_subset():
    row = {feature: float(i + 1) for i, feature in enumerate(FEATURES)}
    train = pd.DataFrame([row, {**row, "N": 99.0}])
    external = pd.DataFrame([{**row, TARGET: "rice"}])
    assert count_exact_feature_overlap(train, external) == 1


def test_external_feature_overlap_allows_disjoint_features():
    train = pd.DataFrame([{feature: float(i + 1) for i, feature in enumerate(FEATURES)}])
    external = pd.DataFrame([{feature: float(i + 20) for i, feature in enumerate(FEATURES)}])
    assert count_exact_feature_overlap(train, external) == 0
