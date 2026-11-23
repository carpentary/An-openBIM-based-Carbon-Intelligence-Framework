from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import QueryMode


@dataclass(frozen=True)
class CalculationCall:
    tool_name: str
    arguments: dict[str, Any]
    rationale: str


class CalculationPlanningError(ValueError):
    def __init__(self, code: str, message: str, missing_fields: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.missing_fields = missing_fields or []


def query_requires_calculator(query: str, mode: QueryMode) -> bool:
    if mode == QueryMode.STRATEGIC_ANALYSIS:
        return True
    if mode == QueryMode.KNOWLEDGE_CORRECTION:
        return False
    lowered = query.lower()
    if any(term in lowered for term in ("ontology term", "ontology术语", "ontology 术语", "subontology", "本体术语")):
        return False
    direct_verb = bool(re.match(r"^\s*(?:please\s+)?(?:calculate|compute)\b", lowered))
    explicit_material_formula = (
        direct_verb
        and any(term in lowered for term in ("material emission", "material carbon"))
    ) or all(term in lowered for term in ("work quantity", "consumption rate", "emission factor"))
    explicit_equipment_formula = (
        direct_verb
        and any(term in lowered for term in ("equipment emission", "equipment carbon"))
    ) or (
        "work volume" in lowered
        and "productivity" in lowered
        and any(term in lowered for term in ("energy per hour", "energy rate"))
        and "energy factor" in lowered
    )
    if explicit_material_formula or explicit_equipment_formula:
        return True
    if any(term in lowered for term in ("carbon", "emission", "碳", "排放")) and any(
        term in lowered for term in ("total", "合计", "总量", "总碳", "总排放", "总carbon")
    ):
        return True
    markers = (
        "total carbon",
        "total emission",
        "carbon totals",
        "stage totals",
        "scope totals",
        "system totals",
        "totals separately",
        "total length",
        "total volume",
        "gross station envelope volume",
        "in total",
        "combined count",
        "aggregate",
        "sum of",
        "compare",
        "comparison",
        "difference",
        "share",
        "percentage",
        "proportion",
        "ratio",
        "rank",
        "highest",
        "largest",
        "contributes the most",
        "lowest",
        "smallest",
        "carbon intensity",
        "per m2",
        "per m3",
        "总碳",
        "总排放",
        "合计",
        "聚合",
        "比较",
        "差值",
        "占比",
        "百分比",
        "比例",
        "排序",
        "最高",
        "最大",
        "最低",
        "最小",
        "碳强度",
        "每平方米",
        "每立方米",
    )
    for marker in markers:
        if re.fullmatch(r"[a-z0-9][a-z0-9 ._-]*", marker):
            if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", lowered):
                return True
        elif marker in lowered:
            return True
    return False


def _labelled_number(query: str, labels: tuple[str, ...]) -> float | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?:{label_pattern})\s*(?:=|is|of|:)?\s*([-+]?\d+(?:\.\d+)?)",
        query,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _explicit_formula_call(query: str) -> CalculationCall | None:
    lowered = query.lower()
    direct_verb = bool(re.match(r"^\s*(?:please\s+)?(?:calculate|compute)\b", lowered))
    equipment_request = direct_verb and any(
        marker in lowered
        for marker in (
            "equipment emission",
            "equipment carbon",
            "equipment calculation",
        )
    ) or (
        "productivity" in lowered
        and any(marker in lowered for marker in ("energy per hour", "energy rate", "energy factor"))
        and any(marker in lowered for marker in ("work volume", "work quantity"))
    )
    if equipment_request:
        values = {
            "work_volume": _labelled_number(query, ("work volume", "work quantity")),
            "productivity_per_hour": _labelled_number(
                query,
                ("productivity per hour", "hourly productivity", "productivity"),
            ),
            "energy_per_hour": _labelled_number(
                query,
                ("energy per hour", "hourly energy", "energy rate"),
            ),
            "energy_factor": _labelled_number(
                query,
                ("energy emission factor", "energy factor"),
            ),
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise CalculationPlanningError(
                "MISSING_EQUIPMENT_EMISSION_PARAMETERS",
                "A5 equipment calculation requires work volume, hourly productivity, hourly energy use and an energy emission factor.",
                missing,
            )
        return CalculationCall(
            "equipment_emission",
            {name: float(value) for name, value in values.items()},
            "Apply the explicit CPD equipment activity and EFD energy factor using the governed A5 formula.",
        )

    material_request = direct_verb and any(
        marker in lowered
        for marker in (
            "material emission",
            "material carbon",
            "material calculation",
        )
    ) or ("work quantity" in lowered and "consumption rate" in lowered)
    if material_request:
        values = {
            "work_quantity": _labelled_number(query, ("work quantity",)),
            "consumption_rate": _labelled_number(
                query,
                ("resource consumption rate", "material consumption rate", "consumption rate"),
            ),
            "emission_factor": _labelled_number(
                query,
                ("material emission factor", "emission factor"),
            ),
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise CalculationPlanningError(
                "MISSING_MATERIAL_EMISSION_PARAMETERS",
                "A1-A3 material calculation requires work quantity, resource consumption rate and an emission factor.",
                missing,
            )
        return CalculationCall(
            "material_emission",
            {
                **{name: float(value) for name, value in values.items()},
                "result_unit": "kgCO2e",
            },
            "Apply the explicit CPD material consumption rate and EFD factor using the governed A1-A3 formula.",
        )
    return None


def _stage_values(overview: dict[str, Any]) -> dict[str, float]:
    return {
        str(item["stage"]): float(item["value"])
        for item in overview.get("totals", [])
        if item.get("value") is not None
    }


def _scope_values(overview: dict[str, Any]) -> dict[str, float]:
    scopes = overview.get("scope_stage_totals_kgco2e", {})
    return {
        str(scope): sum(float(value) for value in stages.values())
        for scope, stages in scopes.items()
    }


def _hotspot_values(context: dict[str, Any]) -> dict[str, float]:
    return {
        str(item.get("element") or item.get("element_uri")): float(item["value"])
        for item in context.get("hotspots", [])
        if item.get("value") is not None
    }


def _work_package_values(context: dict[str, Any]) -> dict[str, float]:
    return {
        str(item.get("template") or item.get("template_uri")): float(item["work_package_count"])
        for item in context.get("work_packages", [])
        if item.get("work_package_count") is not None
    }


def _baseline_emission(context: dict[str, Any]) -> float:
    breakdown = context.get("resource_breakdown", {})
    if breakdown.get("total_emission") is not None:
        value = float(breakdown["total_emission"])
        if value > 0:
            return value
    overview = context.get("project_overview", {})
    if overview.get("combined_total_kgco2e") is not None:
        return float(overview["combined_total_kgco2e"])
    return sum(_stage_values(overview).values())


def _scenario_call(context: dict[str, Any], overlay: dict[str, Any]) -> CalculationCall:
    changes = overlay.get("changes", [])
    if not changes:
        raise CalculationPlanningError("MISSING_SCENARIO_CHANGE", "No executable scenario change was supplied.")
    change = changes[0]
    parameters = change.get("parameters", {})
    change_type = change.get("change_type")
    baseline = _baseline_emission(context)
    if change_type == "factor_reduction":
        return CalculationCall(
            "factor_reduction",
            {"baseline_emission": baseline, "reduction_fraction": parameters["reduction_fraction"]},
            "Apply the explicit factor-reduction fraction to the governed resource or project baseline.",
        )
    if change_type == "material_substitution":
        return CalculationCall(
            "material_substitution",
            {
                "baseline_emission": baseline,
                "substitution_fraction": parameters["substitution_fraction"],
                "baseline_factor": parameters["baseline_factor"],
                "alternative_factor": parameters["alternative_factor"],
            },
            "Apply the explicit substitution fraction and compatible factor pair.",
        )
    if change_type == "transport_override":
        return CalculationCall(
            "transport_emission",
            {
                "mass": parameters["mass"],
                "mass_unit": parameters["mass_unit"],
                "distance_km": parameters["distance_km"],
                "factor_kgco2e_per_tkm": parameters["factor_kgco2e_per_tkm"],
            },
            "Calculate A4 using the explicit mass, distance and tonne-kilometre factor.",
        )
    if change_type == "reuse_cycle_override":
        return CalculationCall(
            "amortise_reusable_resource",
            {
                "baseline_emission": float(parameters.get("baseline_emission", baseline)),
                "reuse_cycles": parameters["reuse_cycles"],
            },
            "Allocate the governed reusable-resource baseline over the explicit cycle count.",
        )
    if change_type == "sensitivity_analysis":
        overview = context.get("project_overview", {})
        target = change.get("target")
        if target == "foundation_pit_combined_carbon":
            foundation = overview.get("foundation_pit", {})
            base = foundation.get("sensitivity", {}).get("base", {})
            baseline_value = float(
                base.get("combined_kgco2e")
                or sum(float(value) for value in foundation.get("stage_totals_kgco2e", {}).values())
            )
        elif target == "a4_carbon":
            baseline_value = float(_stage_values(overview).get("A4_TransportationStage", 0.0))
        else:
            baseline_value = float(overview.get("combined_total_kgco2e") or sum(_stage_values(overview).values()))
        if baseline_value <= 0:
            raise CalculationPlanningError(
                "SENSITIVITY_BASELINE_MISSING",
                "A positive governed baseline is required for sensitivity analysis.",
                ["baseline_value"],
            )
        return CalculationCall(
            "one_at_a_time_sensitivity",
            {
                "baseline_value": baseline_value,
                "baseline_parameter": parameters["baseline_parameter"],
                "parameter_values": parameters["parameter_values"],
                "parameter_name": parameters["parameter_name"],
                "result_unit": parameters.get("result_unit", "kgCO2e"),
            },
            "Run a governed direct-proportional one-at-a-time parameter sweep.",
        )
    raise CalculationPlanningError("UNSUPPORTED_CALCULATION", f"Unsupported change type: {change_type}")


def plan_calculation_call(
    query: str,
    mode: QueryMode,
    context: dict[str, Any],
    overlay: dict[str, Any] | None = None,
) -> CalculationCall | None:
    if mode == QueryMode.STRATEGIC_ANALYSIS:
        return _scenario_call(context, overlay or {})

    direct_call = _explicit_formula_call(query)
    if direct_call is not None:
        return direct_call

    lowered = query.lower()
    overview = context.get("project_overview", {})
    if any(term in lowered for term in ("carbon intensity", "per m2", "per m3", "碳强度", "每平方米", "每立方米")):
        if any(term in lowered for term in ("foundation", "earthwork", "excavation", "基坑", "土方", "开挖")):
            foundation = overview.get("foundation_pit", {})
            numerator = sum(float(value) for value in foundation.get("stage_totals_kgco2e", {}).values())
            denominator = float(foundation.get("geometry", {}).get("excavation_m3") or 0.0)
            denominator_unit = "m3"
        else:
            numerator = float(overview.get("combined_total_kgco2e") or sum(_stage_values(overview).values()))
            denominator = float(overview.get("public_geometry", {}).get("building_area_m2") or 0.0)
            denominator_unit = "m2"
        if numerator <= 0 or denominator <= 0:
            raise CalculationPlanningError(
                "MISSING_INTENSITY_DENOMINATOR",
                "Carbon intensity requires a positive governed numerator and denominator.",
                ["emission_kgco2e", "denominator"],
            )
        return CalculationCall(
            "carbon_intensity",
            {
                "emission_kgco2e": numerator,
                "denominator": denominator,
                "denominator_unit": denominator_unit,
            },
            "Normalize governed carbon by the declared project area or foundation excavation volume.",
        )

    scope_markers = (
        "main structure",
        "enclosure",
        "foundation pit",
        "engineering scope",
        "主体",
        "围护",
        "基坑",
        "工程范围",
    )
    stage_markers = ("lifecycle stage", "stage", "a1", "a4", "a5", "生命周期", "阶段")
    scope_stage_values = overview.get("scope_stage_totals_kgco2e", {})
    scope_marker_groups = (
        ("main structure", "主体"),
        ("enclosure", "围护"),
        ("foundation pit", "基坑"),
    )
    named_scope_count = sum(
        any(marker in lowered for marker in marker_group)
        for marker_group in scope_marker_groups
    )
    a4_across_scopes = "a4" in lowered and (
        named_scope_count > 1
        or any(marker in lowered for marker in ("engineering scope", "工程范围", "each scope"))
    )
    if a4_across_scopes:
        values = {
            str(scope): float(stages.get("A4_TransportationStage", 0.0))
            for scope, stages in scope_stage_values.items()
        }
        rationale = "Use scope-resolved A4 values from the governed reference-case partition."
    elif context.get("a4_transport_legs") and any(
        term in lowered for term in ("transport leg", "material transport", "a4 leg", "运输leg", "运输项", "材料运输")
    ):
        values = {
            f"{row.get('scope')}::{row.get('material_code')}": float(row["emission_kgco2e"])
            for row in context["a4_transport_legs"].get("rows", [])
            if row.get("emission_kgco2e") is not None
        }
        rationale = "Use material-level A4 transport legs from the governed reference artifact."
    elif context.get("hotspots") and any(term in lowered for term in ("element", "hotspot", "top", "构件", "热点", "前十")):
        values = _hotspot_values(context)
        rationale = "Use element-level carbon aggregates from the indexed CKG."
    elif context.get("work_packages") and any(term in lowered for term in ("work package", "template", "工作包", "模板")):
        values = _work_package_values(context)
        rationale = "Use work-package instantiation counts from the indexed CKG."
    elif any(marker in lowered for marker in scope_markers):
        all_scope_values = _scope_values(overview)
        named_scopes = [
            name
            for name, markers in (
                ("main_structure", ("main structure", "主体结构")),
                ("enclosure_structure", ("enclosure", "围护")),
                ("foundation_pit", ("foundation pit", "foundation-pit", "基坑")),
            )
            if any(marker in lowered for marker in markers)
        ]
        multi_scope = len(named_scopes) > 1 or any(
            term in lowered for term in ("engineering scope", "three engineering", "each engineering", "工程范围", "三个工程", "各工程")
        )
        if len(named_scopes) == 1 and not multi_scope:
            scope = named_scopes[0]
            values = dict(overview.get("scope_stage_totals_kgco2e", {}).get(scope, {}))
            rationale = f"Use the non-overlapping lifecycle-stage values for {scope}."
        else:
            values = all_scope_values
            rationale = "Use non-overlapping main, enclosure and foundation-pit totals."
    elif any(marker in lowered for marker in stage_markers):
        values = _stage_values(overview)
        rationale = "Use non-overlapping A1-A3, A4 and A5 project totals."
    else:
        values = _stage_values(overview)
        rationale = "Use non-overlapping project stage totals."
    if not values:
        raise CalculationPlanningError(
            "ANALYSIS_VALUES_MISSING",
            "No governed named values were available for the requested analysis.",
            ["named values"],
        )

    if any(term in lowered for term in ("share", "percentage", "proportion", "ratio", "占比", "百分比", "比例")):
        return CalculationCall("calculate_shares", {"values": values, "unit": "kgCO2e"}, rationale)
    if any(term in lowered for term in ("compare", "comparison", "difference", "highest", "largest", "lowest", "smallest", "比较", "差值", "最高", "最大", "最低", "最小")):
        return CalculationCall("compare_named_values", {"values": values, "unit": "kgCO2e"}, rationale)
    if any(term in lowered for term in ("rank", "ranking", "top", "排序", "前十")):
        return CalculationCall(
            "rank_named_values",
            {"values": values, "unit": "kgCO2e", "descending": True, "limit": min(10, len(values))},
            rationale,
        )
    return CalculationCall("aggregate_named_values", {"values": values, "unit": "kgCO2e"}, rationale)
