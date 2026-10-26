"""Structured provenance and validation context for prediction inputs.

This module records semantics; it does not infer units, OCR confidence, or
compatibility. Numeric validation remains delegated to evaluation.validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping
import math

from evaluation.validation import FEATURES, validate_input_values


class PrimaryState(str, Enum):
    INVALID = "INVALID"
    SEMANTICS_UNRESOLVED = "SEMANTICS_UNRESOLVED"
    EXTRAPOLATED = "EXTRAPOLATED"
    VALIDATED = "VALIDATED"


class FindingKind(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    OUTSIDE_TRAINING_RANGE = "OUTSIDE_TRAINING_RANGE"
    UNRESOLVED_SEMANTICS = "UNRESOLVED_SEMANTICS"
    OCR_UNCERTAINTY = "OCR_UNCERTAINTY"
    MISSING_SOURCE_METADATA = "MISSING_SOURCE_METADATA"


@dataclass(frozen=True)
class ValidationFinding:
    kind: FindingKind
    message: str
    feature: str | None = None


@dataclass(frozen=True)
class InputField:
    feature: str
    numeric_value: float | None
    origin: str
    original_value: Any = None
    transformations: tuple[str, ...] = ()
    unit: str | None = None
    unit_status: str | None = None
    measurement_method_status: str | None = None
    ocr_confidence: float | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RainfallProvenance:
    source: str | None
    original_value: float | None
    original_unit: str | None
    start_date: str | None
    end_date: str | None
    interval: str | None
    aggregation_method: str | None
    transformed_value: float | None
    transformations: tuple[str, ...] = ()
    training_feature_compatibility: bool | None = None
    actual_start_date: str | None = None
    actual_end_date: str | None = None
    observation_count: int | None = None
    missing_precipitation_count: int | None = None
    missing_data_treatment: str | None = None
    clamping_history: tuple[str, ...] = ()
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionInputSemantics:
    fields: tuple[InputField, ...]
    values: dict[str, float]
    findings: tuple[ValidationFinding, ...]
    primary_state: PrimaryState
    rainfall: RainfallProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_state"] = self.primary_state.value
        payload["findings"] = [
            {"kind": finding.kind.value, "message": finding.message, "feature": finding.feature}
            for finding in self.findings
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionInputSemantics":
        fields = tuple(InputField(**item) for item in payload["fields"])
        findings = tuple(
            ValidationFinding(
                kind=FindingKind(item["kind"]),
                message=item["message"],
                feature=item.get("feature"),
            )
            for item in payload["findings"]
        )
        rainfall_payload = payload.get("rainfall")
        rainfall = RainfallProvenance(**rainfall_payload) if rainfall_payload else None
        return cls(
            fields=fields,
            values={str(k): float(v) for k, v in payload["values"].items()},
            findings=findings,
            primary_state=PrimaryState(payload["primary_state"]),
            rainfall=rainfall,
        )


def _finite_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _metadata_for(metadata: Mapping[str, Any] | None, feature: str) -> Mapping[str, Any]:
    if not metadata:
        return {}
    value = metadata.get(feature, {})
    return value if isinstance(value, Mapping) else {}


def _primary_state(findings: list[ValidationFinding]) -> PrimaryState:
    kinds = {finding.kind for finding in findings}
    if FindingKind.INVALID_INPUT in kinds:
        return PrimaryState.INVALID
    if {
        FindingKind.UNRESOLVED_SEMANTICS,
        FindingKind.OCR_UNCERTAINTY,
        FindingKind.MISSING_SOURCE_METADATA,
    } & kinds:
        return PrimaryState.SEMANTICS_UNRESOLVED
    if FindingKind.OUTSIDE_TRAINING_RANGE in kinds:
        return PrimaryState.EXTRAPOLATED
    return PrimaryState.VALIDATED


def build_prediction_input(
    values: Mapping[str, Any],
    *,
    field_metadata: Mapping[str, Any] | None = None,
    training_ranges: Mapping[str, Mapping[str, float]] | None = None,
    rainfall: RainfallProvenance | None = None,
) -> PredictionInputSemantics:
    """Build ordered input metadata and findings without inferring semantics."""
    checked = validate_input_values(values, training_ranges)
    fields: list[InputField] = []
    findings: list[ValidationFinding] = []

    for feature in FEATURES:
        metadata = _metadata_for(field_metadata, feature)
        numeric_value = checked["values"].get(feature)
        fields.append(
            InputField(
                feature=feature,
                numeric_value=numeric_value,
                origin=str(metadata.get("origin", "manual")),
                original_value=metadata.get("original_value", values.get(feature)),
                transformations=tuple(metadata.get("transformations", ())),
                unit=metadata.get("unit"),
                unit_status=metadata.get("unit_status"),
                measurement_method_status=metadata.get("measurement_method_status"),
                ocr_confidence=metadata.get("ocr_confidence"),
                source_metadata=dict(metadata.get("source_metadata", {})),
            )
        )

    for message in checked["errors"]:
        feature = next((name for name in FEATURES if name in message), None)
        findings.append(ValidationFinding(FindingKind.INVALID_INPUT, message, feature))
    for message in checked["warnings"]:
        feature = next((name for name in FEATURES if name in message), None)
        findings.append(ValidationFinding(FindingKind.OUTSIDE_TRAINING_RANGE, message, feature))

    for field in fields:
        metadata = _metadata_for(field_metadata, field.feature)
        if field.unit_status in {"unknown", "unverified", "incompatible"}:
            findings.append(ValidationFinding(
                FindingKind.UNRESOLVED_SEMANTICS,
                f"{field.feature} unit status is {field.unit_status}.",
                field.feature,
            ))
        if field.measurement_method_status in {"unknown", "unverified", "incompatible"}:
            findings.append(ValidationFinding(
                FindingKind.UNRESOLVED_SEMANTICS,
                f"{field.feature} measurement method status is {field.measurement_method_status}.",
                field.feature,
            ))
        if field.origin == "ocr" and metadata.get("ocr_uncertain") is True:
            findings.append(ValidationFinding(
                FindingKind.OCR_UNCERTAINTY,
                f"{field.feature} OCR extraction was marked uncertain.",
                field.feature,
            ))
        if metadata.get("source_metadata_required") is True and not field.source_metadata:
            findings.append(ValidationFinding(
                FindingKind.MISSING_SOURCE_METADATA,
                f"{field.feature} source metadata is required but missing.",
                field.feature,
            ))

    if rainfall is not None:
        if rainfall.training_feature_compatibility is not True:
            findings.append(ValidationFinding(
                FindingKind.UNRESOLVED_SEMANTICS,
                "Rainfall compatibility with the training feature has not been established.",
                "rainfall",
            ))
        if not rainfall.source or not rainfall.interval or not rainfall.aggregation_method:
            findings.append(ValidationFinding(
                FindingKind.MISSING_SOURCE_METADATA,
                "Rainfall source, interval, and aggregation metadata are incomplete.",
                "rainfall",
            ))

    return PredictionInputSemantics(
        fields=tuple(fields),
        values=dict(checked["values"]),
        findings=tuple(findings),
        primary_state=_primary_state(findings),
        rainfall=rainfall,
    )


def summarize_rainfall_response(
    daily: Any,
    *,
    source: str | None,
    requested_start_date: str | None,
    requested_end_date: str | None,
    divisor: float,
    original_unit: str | None = None,
    training_feature_compatibility: bool | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> RainfallProvenance:
    """Reproduce the existing sum/divisor arithmetic and record its evidence.

    ``daily`` must provide ``time`` and ``precipitation_sum`` columns. Pandas'
    existing sum behavior is retained: missing precipitation values are
    skipped, and the divisor is independent of returned-row count.
    """
    if divisor == 0:
        raise ValueError("Rainfall divisor must not be zero.")
    times = daily["time"]
    precipitation = daily["precipitation_sum"]
    total = float(precipitation.sum())
    actual_dates = [str(value) for value in times.dropna().tolist()]
    actual_start = min(actual_dates) if actual_dates else None
    actual_end = max(actual_dates) if actual_dates else None
    missing_count = int(precipitation.isna().sum())
    transformed = total / divisor
    return RainfallProvenance(
        source=source,
        original_value=total,
        original_unit=original_unit,
        start_date=requested_start_date,
        end_date=requested_end_date,
        interval=f"requested {requested_start_date} to {requested_end_date}",
        aggregation_method=f"sum daily precipitation then divide by {divisor:g}",
        transformed_value=transformed,
        transformations=(f"divide_by_{divisor:g}",),
        training_feature_compatibility=training_feature_compatibility,
        actual_start_date=actual_start,
        actual_end_date=actual_end,
        observation_count=int(len(daily)),
        missing_precipitation_count=missing_count,
        missing_data_treatment="missing precipitation values skipped by pandas sum",
        source_metadata=dict(source_metadata or {}),
    )


def record_rainfall_clamp(
    rainfall: RainfallProvenance,
    *,
    final_value: float,
    lower: float,
    upper: float,
) -> RainfallProvenance:
    """Record an existing caller-side clamp without changing its value."""
    if final_value != rainfall.transformed_value:
        history = rainfall.clamping_history + (
            f"clamped to [{lower:g}, {upper:g}]",
        )
    else:
        history = rainfall.clamping_history
    return replace(
        rainfall,
        transformed_value=float(final_value),
        clamping_history=history,
    )
