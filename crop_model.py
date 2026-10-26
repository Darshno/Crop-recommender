"""Train, evaluate, and package the AgriSense crop recommendation model.

Model selection is performed using cross-validation on the training split only.
The held-out test split is used only for the final, one-time evaluation.

Run from the project root:
    python crop_model.py
    python crop_model.py --data Crop_recommendation.csv --output-dir reports/model_benchmarks

The saved crop_model.pkl preserves the keys used by the existing Streamlit apps:
    bundle['model'], bundle['encoder'], bundle['features']
It also contains training ranges and metadata for validation and auditability.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluation.metrics import classification_metrics, calibration_bin_table
from evaluation.validation import (
    FEATURES,
    TARGET,
    build_training_ranges,
    validate_dataset,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
CALIBRATION_SPLITS = 3


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nPass its path with --data.")
    frame = pd.read_csv(path)
    report = validate_dataset(frame)
    if not report["valid"]:
        raise ValueError(
            "Dataset validation failed:\n- " + "\n- ".join(report["errors"])
        )
    frame = frame.loc[:, FEATURES + [TARGET]].copy()
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    return frame, report


def make_candidate_models(include_xgboost: bool = True) -> dict[str, Any]:
    """Build fresh, CPU-friendly estimators; XGBoost is optional at runtime."""
    candidates: dict[str, Any] = {
        "DummyPrior": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2500, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            candidates["XGBoost"] = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            )
        except ImportError:
            warnings.warn(
                "XGBoost is not installed; skipping that candidate. Install xgboost to include it.",
                RuntimeWarning,
            )
    return candidates


def top_k_accuracy_scorer(k: int = 3) -> Callable[[Any, pd.DataFrame, np.ndarray], float]:
    """Create an sklearn scorer compatible with arbitrary estimator class order."""
    def scorer(estimator: Any, x: pd.DataFrame, y_true: np.ndarray) -> float:
        probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
        classes = np.asarray(estimator.classes_)
        k_eff = min(k, probabilities.shape[1])
        top_indices = np.argsort(probabilities, axis=1)[:, -k_eff:]
        top_labels = classes[top_indices]
        return float(np.mean(np.any(top_labels == np.asarray(y_true).reshape(-1, 1), axis=1)))
    return scorer


def run_cross_validation(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: dict[str, Any],
) -> tuple[str, pd.DataFrame]:
    """Compare candidates using training-only CV and select by mean macro-F1."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "top3_accuracy": top_k_accuracy_scorer(3),
    }
    rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        print(f"[CV] {name} ...", flush=True)
        started = time.perf_counter()
        results = cross_validate(
            candidate,
            x_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            scores = results[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(np.mean(scores))
            row[f"cv_{metric}_std"] = float(np.std(scores, ddof=1))
        row["cv_fit_seconds_mean"] = float(np.mean(results["fit_time"]))
        row["cv_score_seconds_mean"] = float(np.mean(results["score_time"]))
        row["cv_total_seconds"] = float(time.perf_counter() - started)
        rows.append(row)
        print(
            f"      macro-F1={row['cv_macro_f1_mean']:.4f} ± {row['cv_macro_f1_std']:.4f}; "
            f"top-3={row['cv_top3_accuracy_mean']:.4f}",
            flush=True,
        )
    table = pd.DataFrame(rows).sort_values(
        by=["cv_macro_f1_mean", "cv_balanced_accuracy_mean", "cv_accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    table.insert(0, "cv_rank", np.arange(1, len(table) + 1))
    # Selection happens here, without inspecting the held-out test data.
    selected_name = str(table.iloc[0]["model"])
    return selected_name, table


def _feature_importances(model: Any, x_train: pd.DataFrame, y_train: np.ndarray) -> pd.Series:
    """Return a consistent feature-importance vector for trees or linear pipelines."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, Pipeline):
        last_step = model.steps[-1][1]
        if hasattr(last_step, "coef_"):
            values = np.mean(np.abs(np.asarray(last_step.coef_, dtype=float)), axis=0)
        elif hasattr(last_step, "feature_importances_"):
            values = np.asarray(last_step.feature_importances_, dtype=float)
        else:
            values = np.zeros(x_train.shape[1], dtype=float)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        values = np.zeros(x_train.shape[1], dtype=float)
    if len(values) != len(x_train.columns):
        return pd.Series(np.zeros(x_train.shape[1]), index=x_train.columns, name="importance")
    return pd.Series(values, index=x_train.columns, name="importance").sort_values(ascending=False)


def _generate_plots(
    output_dir: Path,
    labels: list[str],
    y_test: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    importance: pd.Series,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, predictions, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, interpolation="nearest", aspect="auto", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Held-out Test Confusion Matrix",
        xlabel="Predicted crop",
        ylabel="True crop",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
    )
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    imp = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index.astype(str), imp.values)
    ax.set(title="Feature Importance (selected uncalibrated model)", xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    table = calibration_bin_table(y_test, probabilities, n_bins=10)
    table.to_csv(output_dir / "calibration_bins.csv", index=False)
    populated = table[table["count"] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if not populated.empty:
        ax.plot(
            populated["mean_confidence"],
            populated["accuracy"],
            marker="o",
            label="Model",
        )
    ax.set(
        title="Top-label Reliability Diagram (held-out test)",
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "reliability_diagram.png", dpi=160)
    plt.close(fig)


def _run_jitter_sensitivity(
    model: Any,
    x_test: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Small per-feature jitter experiment; diagnostic only, not a safety guarantee."""
    rows = []
    baseline_prob = np.asarray(model.predict_proba(x_test), dtype=float)
    baseline_pred = np.asarray(model.classes_)[baseline_prob.argmax(axis=1)]
    rng = np.random.default_rng(RANDOM_STATE)
    for feature in FEATURES:
        perturbed = x_test.copy()
        q25, q75 = x_test[feature].quantile([0.25, 0.75])
        scale = max(float(q75 - q25) * 0.05, 1e-8)
        perturbed[feature] = perturbed[feature].to_numpy() + rng.normal(0.0, scale, size=len(perturbed))
        p2 = np.asarray(model.predict_proba(perturbed), dtype=float)
        pred2 = np.asarray(model.classes_)[p2.argmax(axis=1)]
        tv_distance = 0.5 * np.abs(baseline_prob - p2).sum(axis=1)
        rows.append({
            "feature_perturbed": feature,
            "jitter_sd_5pct_iqr": scale,
            "top1_prediction_change_rate": float(np.mean(baseline_pred != pred2)),
            "mean_probability_total_variation": float(np.mean(tv_distance)),
            "sample_count": int(len(x_test)),
            "interpretation": "Sensitivity diagnostic only; perturbation is synthetic, not a real farming scenario.",
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "robustness_jitter_sensitivity.csv", index=False)
    return result


def train_and_evaluate(data_path: Path, project_dir: Path, report_dir: Path) -> dict[str, Any]:
    frame, data_report = load_and_validate_data(data_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "data_validation.json", data_report)
    print(f"Dataset: {len(frame):,} rows, {frame[TARGET].nunique()} crops", flush=True)

    x = frame[FEATURES].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[TARGET].to_numpy())
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = make_candidate_models(include_xgboost=True)
    selected_name, cv_table = run_cross_validation(x_train, y_train, candidates)
    cv_table.to_csv(report_dir / "cross_validation_comparison.csv", index=False)
    print(f"Selected using training-only CV macro-F1: {selected_name}", flush=True)

    base_model = clone(candidates[selected_name])
    base_model.fit(x_train, y_train)

    # Calibrator learns from CV predictions inside the training split. Test data is never used here.
    calibration_cv = StratifiedKFold(
        n_splits=CALIBRATION_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(candidates[selected_name]),
        method="sigmoid",
        cv=calibration_cv,
        ensemble=False,
        n_jobs=-1,
    )
    calibrated_model.fit(x_train, y_train)

    class_ids = np.arange(len(encoder.classes_))
    raw_probabilities = np.asarray(base_model.predict_proba(x_test), dtype=float)
    calibrated_probabilities = np.asarray(calibrated_model.predict_proba(x_test), dtype=float)
    raw_predictions = class_ids[raw_probabilities.argmax(axis=1)]
    calibrated_predictions = class_ids[calibrated_probabilities.argmax(axis=1)]

    raw_metrics = classification_metrics(y_test, raw_probabilities, class_ids, top_k=3)
    calibrated_metrics = classification_metrics(y_test, calibrated_probabilities, class_ids, top_k=3)
    final_results = {
        "evaluation_protocol": {
            "random_state": RANDOM_STATE,
            "test_size_fraction": TEST_SIZE,
            "training_rows": int(len(x_train)),
            "held_out_test_rows": int(len(x_test)),
            "number_of_classes": int(len(encoder.classes_)),
            "selection_rule": "Highest mean 5-fold CV macro-F1 on training split; tie-break by balanced accuracy, then accuracy.",
            "test_split_used_for_selection": False,
            "calibration": "Sigmoid/Platt calibration fitted via 3-fold CV within training split only.",
            "probability_warning": "Calibrated scores are model probabilities, not verified agronomic suitability or yield probabilities.",
        },
        "selected_model": selected_name,
        "uncalibrated_test_metrics": raw_metrics,
        "calibrated_test_metrics": calibrated_metrics,
        "classes": [str(c) for c in encoder.classes_],
    }
    write_json(report_dir / "final_test_metrics.json", final_results)

    report_dict = classification_report(
        y_test,
        calibrated_predictions,
        labels=class_ids,
        target_names=[str(c) for c in encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(report_dir / "classification_report.csv", index=True)

    importances = _feature_importances(base_model, x_train, y_train)
    importances.rename_axis("feature").reset_index().to_csv(report_dir / "feature_importance.csv", index=False)
    _generate_plots(
        report_dir,
        [str(c) for c in encoder.classes_],
        y_test,
        calibrated_probabilities,
        calibrated_predictions,
        importances,
    )
    jitter = _run_jitter_sensitivity(calibrated_model, x_test, report_dir)

    bundle = {
        # Existing Streamlit apps require these three keys. Keep them stable.
        "model": calibrated_model,
        "encoder": encoder,
        "features": list(FEATURES),
        # Additional metadata supports input validation and reproducibility.
        "base_model": base_model,
        "training_ranges": build_training_ranges(x_train),
        "metadata": {
            "project": "AgriSense AI",
            "selected_model": selected_name,
            "selection_metric": "mean_cv_macro_f1",
            "random_state": RANDOM_STATE,
            "test_size_fraction": TEST_SIZE,
            "calibration_method": "sigmoid",
            "calibration_cv_splits": CALIBRATION_SPLITS,
            "features": list(FEATURES),
            "classes": [str(c) for c in encoder.classes_],
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "dataset_filename": data_path.name,
            "sklearn_version": sklearn.__version__,
        },
    }
    model_path = project_dir / "crop_model.pkl"
    joblib.dump(bundle, model_path, compress=3)

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "model_artifact": str(model_path.name),
        "model_artifact_bytes": int(model_path.stat().st_size),
    }
    try:
        import xgboost
        environment["xgboost_version"] = xgboost.__version__
    except ImportError:
        environment["xgboost_version"] = None
    write_json(report_dir / "environment.json", environment)

    print("\nFinal held-out test metrics (uncalibrated -> calibrated):", flush=True)
    for key in ("accuracy", "top_3_accuracy", "macro_f1", "balanced_accuracy", "log_loss", "multiclass_brier_score", "top_label_ece_10_bins"):
        print(
            f"  {key}: {raw_metrics[key]:.4f} -> {calibrated_metrics[key]:.4f}",
            flush=True,
        )
    print(f"\nSaved model bundle: {model_path}", flush=True)
    print(f"Saved evaluation artifacts: {report_dir}", flush=True)
    print("Note: the jitter report is a sensitivity diagnostic, not a real-world robustness guarantee.", flush=True)
    return final_results


def recommend(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    k: int = 3,
    model_path: str | Path = "crop_model.pkl",
) -> list[tuple[str, float]]:
    """Return top-k crop labels and calibrated model scores as percentages.

    The percentages must not be interpreted as probability that a crop will succeed.
    Input validation is intentionally performed before inference.
    """
    bundle = joblib.load(model_path)
    values = dict(zip(FEATURES, [N, P, K, temperature, humidity, ph, rainfall]))
    result, _ = recommend_with_input_semantics(
        values,
        k=k,
        model_path=model_path,
    )
    return result


def recommend_with_input_semantics(
    values: dict[str, Any],
    *,
    k: int = 3,
    model_path: str | Path = "crop_model.pkl",
    field_metadata: dict[str, Any] | None = None,
    rainfall: Any = None,
) -> tuple[list[tuple[str, float]], Any]:
    """Recommend while returning the input semantics context used for inference."""
    from evaluation.input_semantics import build_prediction_input

    bundle = joblib.load(model_path)
    semantics = build_prediction_input(
        values,
        field_metadata=field_metadata,
        training_ranges=bundle.get("training_ranges"),
        rainfall=rainfall,
    )
    invalid = [item.message for item in semantics.findings if item.kind.value == "INVALID_INPUT"]
    if invalid:
        raise ValueError("Invalid recommendation inputs: " + "; ".join(invalid))
    if k < 1:
        raise ValueError("k must be at least 1")
    row = pd.DataFrame([[semantics.values[f] for f in bundle["features"]]], columns=bundle["features"])
    probabilities = np.asarray(bundle["model"].predict_proba(row)[0], dtype=float)
    classes = np.asarray(bundle["model"].classes_)
    top = np.argsort(probabilities)[::-1][: min(k, len(probabilities))]
    return (
        [(str(bundle["encoder"].classes_[int(classes[i])]), round(float(probabilities[i]) * 100, 1)) for i in top],
        semantics,
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate AgriSense AI crop recommendation model.")
    parser.add_argument("--data", type=Path, default=script_dir / "Crop_recommendation.csv", help="Path to crop CSV dataset")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "reports" / "model_benchmarks", help="Directory for evaluation reports")
    args = parser.parse_args()
    train_and_evaluate(args.data.resolve(), script_dir, args.output_dir.resolve())


if __name__ == "__main__":
    main()
