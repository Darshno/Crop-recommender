import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

from crop_model import recommend, recommend_with_input_semantics
from evaluation.validation import FEATURES, build_training_ranges


def test_recommend_preserves_app_compatible_bundle(tmp_path):
    encoder = LabelEncoder().fit(["maize", "rice", "wheat"])
    x_train = pd.DataFrame([
        [30, 20, 20, 20, 50, 6.0, 100],
        [70, 40, 35, 25, 75, 6.5, 150],
        [40, 30, 60, 30, 65, 7.0, 200],
    ], columns=FEATURES)
    y_train = encoder.transform(["maize", "rice", "wheat"])
    model = DummyClassifier(strategy="prior").fit(x_train, y_train)
    bundle = {
        "model": model,
        "encoder": encoder,
        "features": FEATURES,
        "training_ranges": build_training_ranges(x_train),
    }
    path = tmp_path / "bundle.pkl"
    joblib.dump(bundle, path)

    result = recommend(40, 30, 60, 30, 65, 7.0, 200, k=2, model_path=path)
    assert len(result) == 2
    assert all(isinstance(crop, str) for crop, _ in result)
    assert all(0 <= score <= 100 for _, score in result)


def test_recommend_rejects_invalid_input(tmp_path):
    encoder = LabelEncoder().fit(["maize", "rice"])
    x_train = pd.DataFrame([
        [30, 20, 20, 20, 50, 6.0, 100],
        [70, 40, 35, 25, 75, 6.5, 150],
    ], columns=FEATURES)
    y_train = encoder.transform(["maize", "rice"])
    model = DummyClassifier(strategy="prior").fit(x_train, y_train)
    bundle = {
        "model": model,
        "encoder": encoder,
        "features": FEATURES,
        "training_ranges": build_training_ranges(x_train),
    }
    path = tmp_path / "bundle.pkl"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="Invalid recommendation inputs"):
        recommend(-20, 30, 60, 30, 65, 7.0, 200, model_path=path)


def test_k_must_be_positive(tmp_path):
    encoder = LabelEncoder().fit(["maize", "rice"])
    x_train = pd.DataFrame([
        [30, 20, 20, 20, 50, 6.0, 100],
        [70, 40, 35, 25, 75, 6.5, 150],
    ], columns=FEATURES)
    y_train = encoder.transform(["maize", "rice"])
    model = DummyClassifier(strategy="prior").fit(x_train, y_train)
    path = tmp_path / "bundle.pkl"
    joblib.dump({"model": model, "encoder": encoder, "features": FEATURES,
                 "training_ranges": build_training_ranges(x_train)}, path)
    with pytest.raises(ValueError, match="k must be"):
        recommend(40, 30, 30, 22, 60, 6.0, 120, k=0, model_path=path)


def test_recommend_with_input_semantics_preserves_existing_prediction_contract(tmp_path):
    encoder = LabelEncoder().fit(["maize", "rice"])
    x_train = pd.DataFrame([
        [30, 20, 20, 20, 50, 6.0, 100],
        [70, 40, 35, 25, 75, 6.5, 150],
    ], columns=FEATURES)
    y_train = encoder.transform(["maize", "rice"])
    model = DummyClassifier(strategy="prior").fit(x_train, y_train)
    path = tmp_path / "bundle.pkl"
    joblib.dump({
        "model": model,
        "encoder": encoder,
        "features": FEATURES,
        "training_ranges": build_training_ranges(x_train),
    }, path)

    values = dict(zip(FEATURES, [40, 30, 30, 22, 60, 6.0, 120]))
    result, semantics = recommend_with_input_semantics(
        values,
        model_path=path,
        field_metadata={"N": {"origin": "manual", "unit_status": "unknown"}},
    )
    assert len(result) == 2
    assert semantics.fields[0].feature == "N"
    assert semantics.primary_state.value == "SEMANTICS_UNRESOLVED"
