from __future__ import annotations

import math
from itertools import combinations
from typing import Any


class CalculationError(ValueError):
    pass


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise CalculationError(f"{name} must be finite and non-negative")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise CalculationError(f"{name} must be finite")
    return value


def _named_values(values: dict[str, float], *, nonnegative: bool = True) -> dict[str, float]:
    if not values:
        raise CalculationError("at least one named value is required")
    validator = _finite_nonnegative if nonnegative else _finite
    return {str(name): validator(str(name), value) for name, value in values.items()}


def aggregate_named_values(values: dict[str, float], unit: str = "kgCO2e") -> dict[str, Any]:
    governed = _named_values(values)
    return {
        "tool": "aggregate_named_values",
        "formula": "total = sum(non_overlapping_named_values)",
        "values": governed,
        "total": sum(governed.values()),
        "item_count": len(governed),
        "unit": unit,
    }


def compare_named_values(
    values: dict[str, float],
    unit: str = "kgCO2e",
    baseline_name: str | None = None,
) -> dict[str, Any]:
    governed = _named_values(values)
    if baseline_name is not None and baseline_name not in governed:
        raise CalculationError(f"unknown baseline_name: {baseline_name}")
    ranked = sorted(governed.items(), key=lambda item: (-item[1], item[0]))
    maximum_name, maximum_value = ranked[0]
    minimum_name, minimum_value = ranked[-1]
    pairwise = []
    for left, right in combinations(sorted(governed), 2):
        difference = governed[left] - governed[right]
        pairwise.append(
            {
                "left": left,
                "right": right,
                "difference": difference,
                "absolute_difference": abs(difference),
                "percentage_relative_to_right": (
                    difference / governed[right] * 100.0 if governed[right] else None
                ),
            }
        )
    baseline_comparison = []
    if baseline_name is not None:
        baseline = governed[baseline_name]
        baseline_comparison = [
            {
                "name": name,
                "difference_from_baseline": value - baseline,
                "percentage_change_from_baseline": (
                    (value - baseline) / baseline * 100.0 if baseline else None
                ),
            }
            for name, value in sorted(governed.items())
        ]
    return {
        "tool": "compare_named_values",
        "formula": "difference = left - right; percentage = difference / reference * 100",
        "values": governed,
        "ranked": [
            {"rank": index, "name": name, "value": value}
            for index, (name, value) in enumerate(ranked, start=1)
        ],
        "maximum": {"name": maximum_name, "value": maximum_value},
        "minimum": {"name": minimum_name, "value": minimum_value},
        "range": maximum_value - minimum_value,
        "pairwise_differences": pairwise,
        "baseline_name": baseline_name,
        "baseline_comparison": baseline_comparison,
        "unit": unit,
    }


def calculate_shares(values: dict[str, float], unit: str = "kgCO2e") -> dict[str, Any]:
    governed = _named_values(values)
    total = sum(governed.values())
    if total <= 0:
        raise CalculationError("share denominator must be greater than zero")
    return {
        "tool": "calculate_shares",
        "formula": "share = named_value / sum(named_values)",
        "values": governed,
        "total": total,
        "shares": [
            {
                "name": name,
                "value": value,
                "share_fraction": value / total,
                "percentage": value / total * 100.0,
            }
            for name, value in sorted(governed.items())
        ],
        "unit": unit,
    }


def rank_named_values(
    values: dict[str, float],
    unit: str = "kgCO2e",
    descending: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    governed = _named_values(values)
    if limit is not None and limit < 1:
        raise CalculationError("limit must be at least one")
    ordered = sorted(
        governed.items(),
        key=(lambda item: (-item[1], item[0])) if descending else (lambda item: (item[1], item[0])),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return {
        "tool": "rank_named_values",
        "formula": "stable sort by value and then name",
        "descending": descending,
        "ranked": [
            {"rank": index, "name": name, "value": value}
            for index, (name, value) in enumerate(ordered, start=1)
        ],
        "unit": unit,
    }


def one_at_a_time_sensitivity(
    baseline_value: float,
    baseline_parameter: float,
    parameter_values: list[float],
    *,
    parameter_name: str,
    result_unit: str = "kgCO2e",
) -> dict[str, Any]:
    baseline_value = _finite_nonnegative("baseline_value", baseline_value)
    baseline_parameter = _finite_nonnegative("baseline_parameter", baseline_parameter)
    if baseline_parameter == 0:
        raise CalculationError("baseline_parameter must be greater than zero")
    if not parameter_values:
        raise CalculationError("parameter_values must not be empty")
    governed_values = sorted({_finite_nonnegative("parameter_value", value) for value in parameter_values})
    points = []
    for parameter_value in governed_values:
        result = baseline_value * parameter_value / baseline_parameter
        delta = result - baseline_value
        points.append(
            {
                "parameter_value": parameter_value,
                "result": result,
                "delta": delta,
                "percentage_change": delta / baseline_value * 100.0 if baseline_value else 0.0,
            }
        )
    return {
        "tool": "one_at_a_time_sensitivity",
        "formula": "result_i = baseline_result * parameter_i / baseline_parameter",
        "relationship": "direct_proportional",
        "parameter_name": parameter_name,
        "baseline_parameter": baseline_parameter,
        "baseline_value": baseline_value,
        "points": points,
        "minimum": min(points, key=lambda item: item["result"]),
        "maximum": max(points, key=lambda item: item["result"]),
        "result_range": max(item["result"] for item in points) - min(item["result"] for item in points),
        "elasticity": 1.0,
        "unit": result_unit,
    }


def carbon_intensity(
    emission_kgco2e: float,
    denominator: float,
    denominator_unit: str,
) -> dict[str, Any]:
    emission = _finite_nonnegative("emission_kgco2e", emission_kgco2e)
    denominator = _finite_nonnegative("denominator", denominator)
    if denominator == 0:
        raise CalculationError("denominator must be greater than zero")
    return {
        "tool": "carbon_intensity",
        "formula": "intensity = emission_kgco2e / denominator",
        "emission_kgco2e": emission,
        "denominator": denominator,
        "denominator_unit": denominator_unit,
        "result": emission / denominator,
        "unit": f"kgCO2e/{denominator_unit}",
    }


def material_emission(
    work_quantity: float,
    consumption_rate: float,
    emission_factor: float,
    result_unit: str = "kgCO2e",
) -> dict[str, Any]:
    work_quantity = _finite_nonnegative("work_quantity", work_quantity)
    consumption_rate = _finite_nonnegative("consumption_rate", consumption_rate)
    emission_factor = _finite_nonnegative("emission_factor", emission_factor)
    result = work_quantity * consumption_rate * emission_factor
    return {
        "tool": "material_emission",
        "formula": "E = Q_work × q_resource × EF",
        "inputs": {
            "work_quantity": work_quantity,
            "consumption_rate": consumption_rate,
            "emission_factor": emission_factor,
        },
        "result": result,
        "unit": result_unit,
    }


def transport_emission(
    mass: float,
    mass_unit: str,
    distance_km: float,
    factor_kgco2e_per_tkm: float,
) -> dict[str, Any]:
    mass = _finite_nonnegative("mass", mass)
    distance_km = _finite_nonnegative("distance_km", distance_km)
    factor = _finite_nonnegative("factor_kgco2e_per_tkm", factor_kgco2e_per_tkm)
    normalised_unit = mass_unit.lower().replace(" ", "")
    if normalised_unit in {"kg", "kilogram", "kilograms"}:
        mass_tonnes = mass / 1000.0
    elif normalised_unit in {"t", "tonne", "tonnes", "ton"}:
        mass_tonnes = mass
    else:
        raise CalculationError(f"unsupported mass unit: {mass_unit}")
    result = mass_tonnes * distance_km * factor
    return {
        "tool": "transport_emission",
        "formula": "E_A4 = mass_t × distance_km × EF_tkm",
        "inputs": {
            "mass": mass,
            "mass_unit": mass_unit,
            "mass_tonnes": mass_tonnes,
            "distance_km": distance_km,
            "factor_kgco2e_per_tkm": factor,
        },
        "result": result,
        "unit": "kgCO2e",
        "stage": "A4_TransportationStage",
    }


def equipment_emission(
    work_volume: float,
    productivity_per_hour: float,
    energy_per_hour: float,
    energy_factor: float,
) -> dict[str, Any]:
    work_volume = _finite_nonnegative("work_volume", work_volume)
    productivity = _finite_nonnegative("productivity_per_hour", productivity_per_hour)
    energy_per_hour = _finite_nonnegative("energy_per_hour", energy_per_hour)
    energy_factor = _finite_nonnegative("energy_factor", energy_factor)
    if productivity == 0:
        raise CalculationError("productivity_per_hour must be greater than zero")
    machine_hours = work_volume / productivity
    result = machine_hours * energy_per_hour * energy_factor
    return {
        "tool": "equipment_emission",
        "formula": "E_A5 = (Q_work / productivity) × energy_rate × EF_energy",
        "inputs": {
            "work_volume": work_volume,
            "productivity_per_hour": productivity,
            "machine_hours": machine_hours,
            "energy_per_hour": energy_per_hour,
            "energy_factor": energy_factor,
        },
        "result": result,
        "unit": "kgCO2e",
        "stage": "A5_ConstructionStage",
    }


def factor_reduction(baseline_emission: float, reduction_fraction: float) -> dict[str, Any]:
    baseline = _finite_nonnegative("baseline_emission", baseline_emission)
    fraction = float(reduction_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise CalculationError("reduction_fraction must be within [0, 1]")
    alternative = baseline * (1 - fraction)
    delta = alternative - baseline
    return {
        "tool": "factor_reduction",
        "formula": "E_alt = E_baseline × (1 - reduction_fraction)",
        "baseline": baseline,
        "alternative": alternative,
        "delta": delta,
        "reduction": -delta,
        "percentage_change": -100 * fraction,
        "unit": "kgCO2e",
    }


def material_substitution(
    baseline_emission: float,
    substitution_fraction: float,
    baseline_factor: float,
    alternative_factor: float,
) -> dict[str, Any]:
    baseline = _finite_nonnegative("baseline_emission", baseline_emission)
    fraction = float(substitution_fraction)
    base_factor = _finite_nonnegative("baseline_factor", baseline_factor)
    alt_factor = _finite_nonnegative("alternative_factor", alternative_factor)
    if not 0 <= fraction <= 1:
        raise CalculationError("substitution_fraction must be within [0, 1]")
    if base_factor == 0:
        raise CalculationError("baseline_factor must be greater than zero")
    multiplier = (1 - fraction) + fraction * (alt_factor / base_factor)
    alternative = baseline * multiplier
    delta = alternative - baseline
    return {
        "tool": "material_substitution",
        "formula": "E_alt = E_baseline × [(1-f) + f × EF_alt/EF_base]",
        "baseline": baseline,
        "alternative": alternative,
        "delta": delta,
        "reduction": -delta,
        "percentage_change": (delta / baseline * 100) if baseline else 0.0,
        "unit": "kgCO2e",
        "inputs": {
            "substitution_fraction": fraction,
            "baseline_factor": base_factor,
            "alternative_factor": alt_factor,
        },
    }


def amortise_reusable_resource(baseline_emission: float, reuse_cycles: float) -> dict[str, Any]:
    baseline = _finite_nonnegative("baseline_emission", baseline_emission)
    cycles = _finite_nonnegative("reuse_cycles", reuse_cycles)
    if cycles < 1:
        raise CalculationError("reuse_cycles must be at least 1")
    alternative = baseline / cycles
    delta = alternative - baseline
    return {
        "tool": "amortise_reusable_resource",
        "formula": "E_allocated = E_initial / reuse_cycles",
        "baseline": baseline,
        "alternative": alternative,
        "delta": delta,
        "reduction": -delta,
        "percentage_change": (delta / baseline * 100) if baseline else 0.0,
        "unit": "kgCO2e",
        "inputs": {"reuse_cycles": cycles},
    }


def apply_named_deltas(
    baseline_value: float,
    deltas: dict[str, float],
    unit: str = "kgCO2e",
) -> dict[str, Any]:
    """Compose independently calculated deltas onto one governed baseline."""

    baseline = _finite_nonnegative("baseline_value", baseline_value)
    governed_deltas = _named_values(deltas, nonnegative=False)
    net_delta = sum(governed_deltas.values())
    alternative = baseline + net_delta
    if alternative < 0:
        raise CalculationError("composed deltas must not produce a negative alternative")
    return {
        "tool": "apply_named_deltas",
        "formula": "alternative = baseline + sum(named_deltas)",
        "baseline": baseline,
        "deltas": governed_deltas,
        "net_delta": net_delta,
        "alternative": alternative,
        "reduction": -net_delta,
        "reduction_fraction": (-net_delta / baseline) if baseline else 0.0,
        "percentage_change": (net_delta / baseline * 100.0) if baseline else 0.0,
        "unit": unit,
    }


def calculate_ratio(
    numerator: float,
    denominator: float,
    *,
    numerator_name: str = "numerator",
    denominator_name: str = "denominator",
) -> dict[str, Any]:
    numerator_value = _finite_nonnegative("numerator", numerator)
    denominator_value = _finite_nonnegative("denominator", denominator)
    if denominator_value == 0:
        raise CalculationError("ratio denominator must be greater than zero")
    return {
        "tool": "calculate_ratio",
        "formula": "ratio = numerator / denominator",
        "numerator": {"name": numerator_name, "value": numerator_value},
        "denominator": {"name": denominator_name, "value": denominator_value},
        "ratio": numerator_value / denominator_value,
        "unit": "dimensionless",
    }


def evaluate_threshold(
    value: float,
    threshold: float,
    comparison: str = "greater_than",
) -> dict[str, Any]:
    governed_value = _finite("value", value)
    governed_threshold = _finite("threshold", threshold)
    predicates = {
        "greater_than": governed_value > governed_threshold,
        "greater_than_or_equal": governed_value >= governed_threshold,
        "less_than": governed_value < governed_threshold,
        "less_than_or_equal": governed_value <= governed_threshold,
    }
    if comparison not in predicates:
        raise CalculationError(f"unsupported threshold comparison: {comparison}")
    return {
        "tool": "evaluate_threshold",
        "formula": f"value {comparison} threshold",
        "value": governed_value,
        "threshold": governed_threshold,
        "comparison": comparison,
        "passed": predicates[comparison],
        "margin": governed_value - governed_threshold,
        "unit": "dimensionless",
    }


def project_foundation_geometry(
    gross_envelope_m3: float,
    plan_utilisation_factor: float,
    backfill_ratio: float,
    loose_volume_factor: float,
) -> dict[str, Any]:
    gross = _finite_nonnegative("gross_envelope_m3", gross_envelope_m3)
    utilisation = _finite_nonnegative("plan_utilisation_factor", plan_utilisation_factor)
    backfill_fraction = _finite_nonnegative("backfill_ratio", backfill_ratio)
    loose_factor = _finite_nonnegative("loose_volume_factor", loose_volume_factor)
    if utilisation > 1:
        raise CalculationError("plan_utilisation_factor must not exceed 1")
    if backfill_fraction > 1:
        raise CalculationError("backfill_ratio must not exceed 1")
    excavation = gross * utilisation
    backfill = excavation * backfill_fraction
    offsite_in_situ = excavation - backfill
    return {
        "tool": "project_foundation_geometry",
        "formula": (
            "excavation = gross_envelope * utilisation; backfill = excavation * "
            "backfill_ratio; loose_offsite = (excavation - backfill) * loose_factor"
        ),
        "inputs": {
            "gross_envelope_m3": gross,
            "plan_utilisation_factor": utilisation,
            "backfill_ratio": backfill_fraction,
            "loose_volume_factor": loose_factor,
        },
        "excavation_m3": excavation,
        "backfill_m3": backfill,
        "offsite_disposal_in_situ_m3": offsite_in_situ,
        "offsite_transport_loose_m3": offsite_in_situ * loose_factor,
        "unit": "m3",
    }


def compare_to_reference(
    values: dict[str, float],
    reference_value: float,
    reference_name: str = "governed_reference",
    tolerance: float = 0.0,
) -> dict[str, Any]:
    governed = _named_values(values)
    reference = _finite_nonnegative("reference_value", reference_value)
    governed_tolerance = _finite_nonnegative("tolerance", tolerance)
    comparisons = []
    for name, value in sorted(governed.items()):
        difference = value - reference
        comparisons.append(
            {
                "name": name,
                "value": value,
                "difference": difference,
                "matches_reference": abs(difference) <= governed_tolerance,
                "conflicts_with_reference": abs(difference) > governed_tolerance,
            }
        )
    return {
        "tool": "compare_to_reference",
        "formula": "conflict = abs(value - reference) > tolerance",
        "reference": {"name": reference_name, "value": reference},
        "tolerance": governed_tolerance,
        "comparisons": comparisons,
        "unit": "dimensionless",
    }
