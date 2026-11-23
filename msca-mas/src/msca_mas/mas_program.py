from __future__ import annotations

import re
from typing import Any

from .analysis import CalculationCall, CalculationPlanningError, plan_calculation_call
from .entity_resolution import matching_transport_defaults, transport_material_labels
from .models import QueryMode
from .semantic_models import QueryContract, QueryOperator


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


def _step(
    step_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    rationale: str,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "tool_name": tool_name,
        "arguments": arguments,
        "dependencies": list(dependencies or []),
        "expected_unit": str(arguments.get("result_unit") or arguments.get("unit") or "") or None,
        "rationale": rationale,
    }


def _match_number(pattern: str, query: str, label: str) -> float:
    match = re.search(pattern, query, re.I)
    if not match:
        raise CalculationPlanningError(
            "CALCULATION_PROGRAM_PARAMETER_MISSING",
            f"The compound calculation program requires an explicit {label}.",
            [label],
        )
    return float(match.group(1))


def _target_percentage(query: str, target: str, label: str) -> float:
    patterns = (
        rf"\b(?:reduce|reduces|reduced|lower|lowers|lowered|decrease|decreases|decreased)\b"
        rf"[^.;]{{0,48}}?\b{target}\b[^.;]{{0,48}}?\bby\s*({_NUMBER})\s*%",
        rf"({_NUMBER})\s*%\s*\b{target}\b[^.;]{{0,48}}?"
        rf"\b(?:reduction|improvement|decrease|lowering)\b",
        rf"\b{target}\b[^.;]{{0,48}}?({_NUMBER})\s*%\s*"
        rf"\b(?:reduction|improvement|decrease)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.I)
        if match:
            return float(match.group(1)) / 100.0
    raise CalculationPlanningError(
        "CALCULATION_PROGRAM_PARAMETER_MISSING",
        f"The compound calculation program requires an explicit {label}.",
        [label],
    )


def _decision_threshold(query: str) -> float:
    patterns = (
        rf"\b(?:exceeds?|threshold(?:\s+of)?|target(?:\s+of)?)\s*({_NUMBER})\s*%",
        rf"({_NUMBER})\s*%\s*(?:decision\s+)?threshold\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.I)
        if match:
            return float(match.group(1)) / 100.0
    raise CalculationPlanningError(
        "CALCULATION_PROGRAM_PARAMETER_MISSING",
        "The compound calculation program requires an explicit reduction threshold.",
        ["reduction threshold"],
    )


def _transport_alternative_inputs(query: str) -> tuple[float, str, list[float], float]:
    mass_match = re.search(rf"({_NUMBER})\s*(kg|t|tonne|tonnes|ton)(?![a-z])", query, re.I)
    if not mass_match:
        raise CalculationPlanningError(
            "CALCULATION_PROGRAM_PARAMETER_MISSING",
            "A transport-alternative program requires an explicit mass.",
            ["mass", "mass_unit"],
        )
    list_match = re.search(
        rf"((?:{_NUMBER}\s*(?:,|/|\band\b)\s*)+{_NUMBER})\s*km\b",
        query,
        re.I,
    )
    if list_match:
        distances = [float(item) for item in re.findall(_NUMBER, list_match.group(1), re.I)]
    else:
        distances = [float(item) for item in re.findall(rf"({_NUMBER})\s*km\b", query, re.I)]
    factor_match = re.search(
        rf"({_NUMBER})\s*kg\s*co2e?\s*(?:per|/)\s*(?:t|tonne)[- *]?km",
        query,
        re.I,
    ) or re.search(rf"\bfactor\s*(?:=|is|of|:)?\s*({_NUMBER})", query, re.I)
    if len(distances) < 2 or not factor_match:
        raise CalculationPlanningError(
            "CALCULATION_PROGRAM_PARAMETER_MISSING",
            "Transport alternatives require at least two distances and a tonne-kilometre factor.",
            ["distances", "factor"],
        )
    return (
        float(mass_match.group(1)),
        mass_match.group(2),
        list(dict.fromkeys(distances)),
        float(factor_match.group(1)),
    )


def _stage_values(overview: dict[str, Any]) -> dict[str, float]:
    return {
        str(item["stage"]): float(item["value"])
        for item in overview.get("totals", [])
        if item.get("value") is not None
    }


def _scope_values(overview: dict[str, Any]) -> dict[str, float]:
    return {
        str(scope): sum(float(value) for value in stages.values())
        for scope, stages in overview.get("scope_stage_totals_kgco2e", {}).items()
    }


def _resource_breakdown(context: dict[str, Any], target: str) -> dict[str, Any]:
    target_lower = target.lower()
    breakdowns = context.get("resource_breakdowns", {})
    for key, value in breakdowns.items():
        if target_lower in str(key).lower():
            return value
    single = context.get("resource_breakdown", {})
    descriptor = " ".join(
        str(single.get(key, "")) for key in ("term", "resource_term", "label")
    ).lower()
    if not breakdowns and (target_lower in descriptor or single.get("total_emission") is not None):
        return single
    return {}


def _resource_baseline(context: dict[str, Any], target: str) -> float:
    breakdown = _resource_breakdown(context, target)
    if breakdown.get("total_emission") is not None:
        # A zero result must not fall back to the project total.
        return float(breakdown["total_emission"])
    raise CalculationPlanningError(
        "RESOURCE_BASELINE_MISSING",
        f"No governed {target} resource baseline was retrieved.",
        [f"{target}_baseline"],
    )


def _project_total(overview: dict[str, Any]) -> float:
    value = overview.get("combined_total_kgco2e")
    return float(value) if value is not None else sum(_stage_values(overview).values())


def _ir_values(contract: QueryContract, context: dict[str, Any]) -> tuple[dict[str, float], str]:
    overview = context.get("project_overview", {})
    operations = set(contract.operation_ids)
    scope_stages = overview.get("scope_stage_totals_kgco2e", {})
    scopes = list(contract.target_scopes)
    stages = contract.target_stages
    # Keep the requested output grouping, regardless of how the evidence was retrieved.
    if len(stages) > 1 and len(scopes) == 1:
        scope = scopes[0]
        if scope == "project":
            selected = {
                stage: value
                for stage, value in _stage_values(overview).items()
                if stage in stages
            }
        else:
            selected = {
                str(stage): float(value)
                for stage, value in scope_stages.get(scope, {}).items()
                if stage in stages
            }
        return selected, f"Use the contracted lifecycle-stage axis for {scope}."
    if "retrieve_elements" in operations and context.get("hotspots"):
        return (
            {
                str(item.get("element") or item.get("element_uri")): float(item["value"])
                for item in context["hotspots"]
                if item.get("value") is not None
            },
            "Use the ontology-routed element result set returned by the Reader.",
        )
    if "retrieve_work_packages" in operations and context.get("work_packages"):
        return (
            {
                str(item.get("template") or item.get("template_uri")): float(item["work_package_count"])
                for item in context["work_packages"]
                if item.get("work_package_count") is not None
            },
            "Use the governed work-package counts returned by the Reader.",
        )
    # Prefer typed targets to scope mentions.
    if scopes:
        if scopes == ["project"]:
            selected = {
                stage: value
                for stage, value in _stage_values(overview).items()
                if not stages or stage in stages
            }
            return selected, "Use the reconciled project lifecycle-stage totals."
        if "project" in scopes:
            scopes = [scope for scope in scopes if scope != "project"]
        if len(scopes) == 1:
            selected = {
                str(stage): float(value)
                for stage, value in scope_stages.get(scopes[0], {}).items()
                if not stages or stage in stages
            }
            return selected, f"Use lifecycle-stage values for the contracted {scopes[0]} scope."
        if stages == ["A4_TransportationStage"]:
            return (
                {
                    scope: float(scope_stages.get(scope, {}).get("A4_TransportationStage", 0.0))
                    for scope in scopes
                },
                "Use the A4 partition for every contracted engineering scope.",
            )
        return (
            {
                scope: sum(float(value) for value in scope_stages.get(scope, {}).values())
                for scope in scopes
            },
            "Use non-overlapping A1-A5 totals for every contracted engineering scope.",
        )
    return _stage_values(overview), "Use the reconciled non-overlapping project lifecycle-stage totals."


def _contractual_ir_call(
    query: str,
    contract: QueryContract,
    context: dict[str, Any],
) -> CalculationCall:
    if contract.primary_operator == QueryOperator.INTENSITY:
        call = plan_calculation_call(query, QueryMode.INFORMATION_RETRIEVAL, context, {})
        if call is None:
            raise CalculationPlanningError("NO_CALCULATION_SELECTED", "No intensity calculation was selected.")
        return call
    values, rationale = _ir_values(contract, context)
    if not values:
        raise CalculationPlanningError(
            "ANALYSIS_VALUES_MISSING",
            "No governed values satisfy the contracted scope and stage filters.",
            ["named values"],
        )
    operator = contract.primary_operator
    if operator not in {
        QueryOperator.AGGREGATE,
        QueryOperator.COMPARE,
        QueryOperator.RANK,
        QueryOperator.SHARE,
        QueryOperator.INTENSITY,
    }:
        lowered = query.casefold()
        if any(
            term in lowered
            for term in ("compare", "reconcile", "side by side", "against", "versus", " vs ", "difference")
        ):
            operator = QueryOperator.COMPARE
        elif any(
            term in lowered
            for term in ("highest", "lowest", "largest", "smallest", "rank", "ranking", "top ", "extreme")
        ):
            operator = QueryOperator.RANK
        else:
            operator = QueryOperator.AGGREGATE
        rationale = f"{rationale} Operator inferred from the query's output-action wording ({operator.value})."
    if operator == QueryOperator.SHARE:
        return CalculationCall("calculate_shares", {"values": values, "unit": "kgCO2e"}, rationale)
    if operator == QueryOperator.COMPARE:
        return CalculationCall("compare_named_values", {"values": values, "unit": "kgCO2e"}, rationale)
    if operator == QueryOperator.RANK:
        return CalculationCall(
            "rank_named_values",
            {"values": values, "unit": "kgCO2e", "descending": True, "limit": min(10, len(values))},
            rationale,
        )
    return CalculationCall("aggregate_named_values", {"values": values, "unit": "kgCO2e"}, rationale)


def _simple_call(
    query: str,
    mode: QueryMode,
    contract: QueryContract,
    context: dict[str, Any],
    overlay: dict[str, Any],
) -> CalculationCall:
    if mode == QueryMode.INFORMATION_RETRIEVAL:
        # Use dedicated tools for explicit formulas; infer other operations from the requested output.
        from .analysis import _explicit_formula_call

        direct = _explicit_formula_call(query)
        if direct is not None:
            return direct
        return _contractual_ir_call(query, contract, context)

    call = plan_calculation_call(query, mode, context, overlay)
    if call is None:
        raise CalculationPlanningError(
            "NO_CALCULATION_SELECTED",
            "The query reached Calculator but no governed function was selected.",
        )
    changes = overlay.get("changes", [])
    if mode == QueryMode.STRATEGIC_ANALYSIS and changes:
        change = changes[0]
        target = str(change.get("target", "")).lower()
        if call.tool_name == "factor_reduction" and target != "all_resources":
            for resource in ("concrete", "steel", "formwork"):
                if resource in target:
                    call = CalculationCall(
                        call.tool_name,
                        {**call.arguments, "baseline_emission": _resource_baseline(context, resource)},
                        "Bind the factor reduction to the exact resource baseline selected by the query contract.",
                    )
                    break
        elif call.tool_name == "material_substitution":
            call = CalculationCall(
                call.tool_name,
                {**call.arguments, "baseline_emission": _resource_baseline(context, "concrete")},
                "Bind material substitution to the governed concrete resource baseline.",
            )
    return call


def _program_specification(template: str) -> dict[str, list[str]]:
    specifications = {
        "dual_reduction_threshold": [
            "factor_reduction(concrete,0.10)",
            "factor_reduction(A4,0.15)",
            "aggregate",
            "threshold(0.08)",
        ],
        "substitution_rank_project_delta": [
            "material_substitution",
            "project_delta",
            "rank_options",
        ],
        "sensitivity_argmin_post_retrieval": [
            "sensitivity",
            "argmin",
            "retrieve_geometry_at_selected_point",
        ],
        "reuse_reduction_composition": [
            "reuse_allocation",
            "factor_reduction",
            "compose_deltas",
            "compare_options",
        ],
        "multi_transport_rank_post_retrieval": [
            "three_transport_calls",
            "rank_options",
            "retrieve_default_after_calculation",
        ],
    }
    if template not in specifications:
        raise CalculationPlanningError(
            "UNSUPPORTED_COMPOUND_PROGRAM",
            f"No governed dependency compiler exists for compound template {template}.",
        )
    return {"steps": specifications[template]}



_REDUCTION_TARGET_RESOURCES = ("concrete", "steel", "formwork")
_REDUCTION_TARGET_STAGES = ("a4", "a1-a3", "a5")


def _parse_reduction_intents(query: str) -> list[tuple[str, float]]:
    """Each distinct 'reduce <target> by N%' clause is one reduction operator."""

    intents: list[tuple[str, float]] = []
    for match in re.finditer(
        rf"\b(?:reduce|reduces|reduced|lower|lowers|lowered|cut|cuts)\s+"
        rf"(?P<target>[a-z0-9\-]+(?:\s+[a-z0-9\-]+){{0,3}}?)\s+by\s+({_NUMBER})\s*%",
        query,
        re.I,
    ):
        target = match.group("target").strip().lower()
        fraction = float(match.group(2)) / 100.0
        normalised = next(
            (resource for resource in _REDUCTION_TARGET_RESOURCES if resource in target),
            None,
        )
        if normalised is None:
            normalised = next(
                (stage for stage in _REDUCTION_TARGET_STAGES if stage in target),
                None,
            )
        if normalised is not None and (normalised, fraction) not in intents:
            intents.append((normalised, fraction))
    return intents


def _parse_substitution_intent(query: str) -> tuple[float, float, float] | None:
    fraction_match = re.search(rf"({_NUMBER})\s*%\s*(?:ggbs|slag|binder)", query, re.I)
    factor_match = re.search(rf"factors?\s*({_NUMBER})\s*(?:and|,)\s*({_NUMBER})", query, re.I)
    if not fraction_match or not factor_match:
        return None
    return (
        float(fraction_match.group(1)) / 100.0,
        float(factor_match.group(1)),
        float(factor_match.group(2)),
    )


def _composition_intents(query: str) -> dict[str, bool]:
    lowered = query.casefold()
    return {
        "carry_to_project": bool(
            re.search(r"\b(?:carry|carries|carried|propagate|propagat\w*|apply)\b[^.]{0,60}\b(?:into|to)\b[^.]{0,30}\b(?:station|project)\b", lowered)
            or re.search(r"\b(?:station|project)\s+total\b", lowered)
            and re.search(r"\bdelta\b", lowered)
        ),
        "rank": bool(re.search(r"\b(?:rank|ranking|from low(?:est)? to high(?:est)?|order\w* (?:the )?(?:three|options|routes|alternatives))\b", lowered)),
        "compare": bool(re.search(r"\b(?:compare|versus|against|side by side|difference between)\b", lowered)),
        "threshold": bool(re.search(r"\b(?:exceeds?|threshold|decision threshold|at least)\s*({_NUMBER})\s*%".format(_NUMBER=_NUMBER), lowered)),
        "multiplier": bool(re.search(r"\b(?:multiplier|ratio|percentage change)\b", lowered)),
        "against_default": bool(re.search(r"\b(?:governed|current|station|concrete)\s+default\b", lowered)),
    }


def _assemble_compound_program(
    query: str,
    contract: QueryContract,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Generic operator-algebra assembly for compound contracts.

    Typed calculation intents (reduction / substitution / transport / reuse /
    sensitivity) harvested from the query become leaf operators; output-action
    wording selects composition operators (delta composition onto the project
    total, comparison, ranking, threshold, ratio, default validation).  Every
    step is a registered deterministic function and every dependency is an
    explicit ``$ref``; unknown signal combinations still fail closed.
    """

    steps: list[dict[str, Any]] = []

    def next_id() -> str:
        return f"CALC-{len(steps) + 1:03d}"

    overview = context.get("project_overview", {})
    project_total = _project_total(overview)
    delta_refs: dict[str, Any] = {}
    computation_ids: list[str] = []
    route_result_ids: dict[str, str] = {}

    substitution = _parse_substitution_intent(query)
    if substitution is not None:
        fraction, f_base, f_alt = substitution
        sid = next_id()
        steps.append(
            _step(
                sid,
                "material_substitution",
                {
                    "baseline_emission": _resource_baseline(context, "concrete"),
                    "substitution_fraction": fraction,
                    "baseline_factor": f_base,
                    "alternative_factor": f_alt,
                },
                "Calculate the substituted concrete option.",
            )
        )
        delta_refs["concrete_substitution"] = {"$ref": f"{sid}.delta"}
        computation_ids.append(sid)

    for target, fraction in _parse_reduction_intents(query):
        sid = next_id()
        if target in _REDUCTION_TARGET_STAGES:
            baseline = float(_stage_values(overview).get(
                "A4_TransportationStage" if target == "a4" else f"{target.upper()}_ProductStage",
                0.0,
            ))
        else:
            baseline = _resource_baseline(context, target)
        steps.append(
            _step(
                sid,
                "factor_reduction",
                {"baseline_emission": baseline, "reduction_fraction": fraction},
                f"Calculate the {target} reduction branch.",
            )
        )
        delta_refs[f"{target}_reduction"] = {"$ref": f"{sid}.delta"}
        computation_ids.append(sid)

    try:
        mass, unit, distances, factor = _transport_alternative_inputs(query)
    except CalculationPlanningError:
        mass, unit, distances, factor = None, None, [], None
    if distances:
        for distance in distances:
            sid = next_id()
            steps.append(
                _step(
                    sid,
                    "transport_emission",
                    {"mass": mass, "mass_unit": unit, "distance_km": distance, "factor_kgco2e_per_tkm": factor},
                    f"Calculate the {distance:g} km A4 option.",
                )
            )
            route_result_ids[f"{distance:g} km"] = sid
            computation_ids.append(sid)

    reuse = re.search(
        rf"({_NUMBER})\s*kg\s*co2e?\s*(?:of\s+)?formwork[^.]*?over\s+({_NUMBER})\s*(?:reuse\s*)?(?:uses|cycles)",
        query,
        re.I,
    )
    if reuse:
        sid = next_id()
        steps.append(
            _step(
                sid,
                "amortise_reusable_resource",
                {"baseline_emission": float(reuse.group(1)), "reuse_cycles": int(float(reuse.group(2)))},
                "Calculate the reusable-formwork allocation.",
            )
        )
        delta_refs["formwork_reuse"] = {"$ref": f"{sid}.delta"}
        computation_ids.append(sid)

    if not steps:
        # Allow a retrieval-only contract to continue without a calculation leaf.
        retrieval_shaped = (
            contract.mode != QueryMode.STRATEGIC_ANALYSIS
            and bool(
                re.search(
                    r"\b(?:compar\w*|reconcil\w*|side by side|show\w*|check\w*|against|versus)\b",
                    query,
                    re.I,
                )
            )
            and bool(contract.target_scopes or contract.target_stages)
        )
        if not retrieval_shaped:
            raise CalculationPlanningError(
                "UNSUPPORTED_COMPOUND_PROGRAM",
                "No typed calculation intent was recognised for this compound request.",
            )
        try:
            call = _contractual_ir_call(query, contract, context)
        except CalculationPlanningError:
            raise CalculationPlanningError(
                "UNSUPPORTED_COMPOUND_PROGRAM",
                "No typed calculation intent was recognised for this compound request.",
            )
        return {
            "compiler": "typed_dependency_compiler",
            "steps": [_step("CALC-001", call.tool_name, call.arguments, call.rationale)],
            "final_result_template": {
                "tool": "dependency_ordered_calculation_program",
                "program_specification": {"assembler": "operator_algebra_retrieval_fallback"},
                "status": "complete",
            },
        }

    composition = _composition_intents(query)

    composed_sid: str | None = None
    if composition["carry_to_project"] and delta_refs:
        composed_sid = next_id()
        steps.append(
            _step(
                composed_sid,
                "apply_named_deltas",
                {"baseline_value": project_total, "deltas": delta_refs, "unit": "kgCO2e"},
                "Compose every independent delta onto the governed project baseline once.",
                computation_ids,
            )
        )

    if composition["threshold"]:
        threshold_match = re.search(rf"({_NUMBER})\s*%", query[query.casefold().find("threshold") :] if "threshold" in query.casefold() else query, re.I) or re.search(
            rf"(?:exceeds?|at least)\s*({_NUMBER})\s*%", query, re.I
        )
        if threshold_match:
            source = composed_sid or computation_ids[0]
            steps.append(
                _step(
                    next_id(),
                    "evaluate_threshold",
                    {"value": {"$ref": f"{source}.reduction_fraction" if composed_sid else f"{source}.reduction"}, "threshold": float(threshold_match.group(1)) / 100.0, "comparison": "greater_than"},
                    "Test the explicit decision threshold.",
                    [source],
                )
            )

    if route_result_ids:
        rank_id = next_id()
        steps.append(
            _step(
                rank_id,
                "rank_named_values",
                {
                    "values": {"$step_results": {"steps": {label: sid for label, sid in route_result_ids.items()}, "field": "result"}},
                    "unit": "kgCO2e",
                    "descending": not composition["rank"] or "from low" not in query.casefold(),
                    "limit": len(route_result_ids),
                },
                "Rank the calculated alternatives deterministically.",
                list(route_result_ids.values()),
            )
        )
        if composition["against_default"]:
            default_rows = context.get("material_transport_defaults", {}).get("rows", [])
            concrete_rows = [row for row in default_rows if "concrete" in str(row.get("material_name", "")).lower()]
            reference_row = (concrete_rows or default_rows or [{}])[0]
            steps.append(
                _step(
                    next_id(),
                    "compare_to_reference",
                    {
                        "values": {label: float(label.split()[0]) for label in route_result_ids},
                        "reference_value": float(reference_row.get("default_distance_km", 0.0)),
                        "reference_name": str(reference_row.get("material_code") or "governed_concrete_default"),
                        "tolerance": 0.0,
                    },
                    "Validate every calculated option against the retrieved concrete default.",
                    [rank_id],
                )
            )

    if composition["compare"] and composed_sid:
        steps.append(
            _step(
                next_id(),
                "compare_named_values",
                {"values": {"baseline_project": project_total, "revised_project": {"$ref": f"{composed_sid}.alternative"}}, "unit": "kgCO2e", "baseline_name": "baseline_project"},
                "Compare baseline and revised project totals.",
                [composed_sid],
            )
        )
    elif composition["compare"] and substitution is not None and not composed_sid:
        sub_sid = computation_ids[0]
        steps.append(
            _step(
                next_id(),
                "compare_named_values",
                {
                    "values": {
                        "concrete_baseline": _resource_baseline(context, "concrete"),
                        "revised_option": {"$ref": f"{sub_sid}.alternative"},
                    },
                    "unit": "kgCO2e",
                    "baseline_name": "concrete_baseline",
                },
                "Compare the substituted option against its concrete baseline.",
                [sub_sid],
            )
        )
    elif composition["compare"] and len(route_result_ids) < 2 and not route_result_ids and len(computation_ids) >= 2:
        steps.append(
            _step(
                next_id(),
                "compare_named_values",
                {"values": {sid: {"$ref": f"{sid}.alternative"} for sid in computation_ids[:2]}, "unit": "kgCO2e"},
                "Compare the two governed options.",
                computation_ids[:2],
            )
        )

    if composition["multiplier"] and computation_ids:
        ratio_match = re.search(
            rf"(?:multiplier|ratio)\b[^.]*?({_NUMBER})\s*km[^.]*?({_NUMBER})\s*km", query, re.I
        )
        if ratio_match:
            steps.append(
                _step(
                    next_id(),
                    "calculate_ratio",
                    {"numerator": float(ratio_match.group(1)), "denominator": float(ratio_match.group(2)), "numerator_name": "new_distance_km", "denominator_name": "old_distance_km"},
                    "Calculate the requested distance multiplier.",
                )
            )

    specification = {
        "assembler": "operator_algebra",
        "leaf_operators": [step["tool_name"] for step in steps],
    }
    return {
        "compiler": "typed_dependency_compiler",
        "steps": steps,
        "final_result_template": {
            "tool": "dependency_ordered_calculation_program",
            "program_specification": specification,
            "status": "complete",
        },
    }


def _compound_program(
    query: str,
    contract: QueryContract,
    context: dict[str, Any],
) -> dict[str, Any]:
    template = str(contract.compound_template or "")
    try:
        return _template_compound_program(query, contract, context, template)
    except CalculationPlanningError as exc:
        # Try operator assembly when the selected template lacks the required inputs.
        if exc.code == "UNSUPPORTED_COMPOUND_PROGRAM":
            raise
        return _assemble_compound_program(query, contract, context)


def _template_compound_program(
    query: str,
    contract: QueryContract,
    context: dict[str, Any],
    template: str,
) -> dict[str, Any]:
    overview = context.get("project_overview", {})
    project_total = _project_total(overview)
    steps: list[dict[str, Any]] = []
    try:
        program_specification = _program_specification(template)
    except CalculationPlanningError:
        program_specification = {"assembler": "operator_algebra"}

    if template == "dual_reduction_threshold":
        concrete_fraction = _target_percentage(query, "concrete", "concrete reduction percentage")
        a4_fraction = _target_percentage(query, "a4", "A4 reduction percentage")
        threshold = _decision_threshold(query)
        program_specification = {
            "steps": [
                f"factor_reduction(concrete,{concrete_fraction:.12g})",
                f"factor_reduction(A4,{a4_fraction:.12g})",
                "aggregate",
                f"threshold({threshold:.12g})",
            ]
        }
        a4_baseline = float(_stage_values(overview).get("A4_TransportationStage", 0.0))
        steps = [
            _step("CALC-001", "factor_reduction", {"baseline_emission": _resource_baseline(context, "concrete"), "reduction_fraction": concrete_fraction}, "Calculate the concrete branch."),
            _step("CALC-002", "factor_reduction", {"baseline_emission": a4_baseline, "reduction_fraction": a4_fraction}, "Calculate the A4 branch."),
            _step("CALC-003", "apply_named_deltas", {"baseline_value": project_total, "deltas": {"concrete": {"$ref": "CALC-001.delta"}, "A4": {"$ref": "CALC-002.delta"}}, "unit": "kgCO2e"}, "Compose independent deltas once, without double counting.", ["CALC-001", "CALC-002"]),
            _step("CALC-004", "compare_named_values", {"values": {"baseline": project_total, "coupled_alternative": {"$ref": "CALC-003.alternative"}}, "unit": "kgCO2e", "baseline_name": "baseline"}, "Compare the coupled option with the governed project baseline.", ["CALC-003"]),
            _step("CALC-005", "evaluate_threshold", {"value": {"$ref": "CALC-003.reduction_fraction"}, "threshold": threshold, "comparison": "greater_than"}, "Test the explicit decision threshold.", ["CALC-003"]),
        ]
    elif template == "substitution_rank_project_delta":
        fraction = _match_number(rf"({_NUMBER})\s*%\s*ggbs", query, "substitution percentage") / 100.0
        factor_match = re.search(rf"factors?\s*({_NUMBER})\s*(?:and|,)\s*({_NUMBER})", query, re.I)
        if not factor_match:
            raise CalculationPlanningError("CALCULATION_PROGRAM_PARAMETER_MISSING", "Two compatible substitution factors are required.", ["baseline_factor", "alternative_factor"])
        concrete = _resource_baseline(context, "concrete")
        steps = [
            _step("CALC-001", "material_substitution", {"baseline_emission": concrete, "substitution_fraction": fraction, "baseline_factor": float(factor_match.group(1)), "alternative_factor": float(factor_match.group(2))}, "Calculate the revised concrete option."),
            _step("CALC-002", "apply_named_deltas", {"baseline_value": project_total, "deltas": {"concrete_substitution": {"$ref": "CALC-001.delta"}}, "unit": "kgCO2e"}, "Propagate only the concrete delta to project level.", ["CALC-001"]),
            _step("CALC-003", "compare_named_values", {"values": {"baseline_project": project_total, "revised_project": {"$ref": "CALC-002.alternative"}}, "unit": "kgCO2e", "baseline_name": "baseline_project"}, "Compare baseline and revised project totals.", ["CALC-002"]),
            _step("CALC-004", "rank_named_values", {"values": {"baseline_project": project_total, "revised_project": {"$ref": "CALC-002.alternative"}}, "unit": "kgCO2e", "descending": False, "limit": 2}, "Rank the two options from lower to higher carbon.", ["CALC-002"]),
        ]
    elif template == "sensitivity_argmin_post_retrieval":
        values = sorted({float(item) for item in re.findall(rf"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", query)})
        if not values:
            raise CalculationPlanningError("CALCULATION_PROGRAM_PARAMETER_MISSING", "Sensitivity points are required.", ["parameter_values"])
        foundation = overview.get("foundation_pit", {})
        parameters = overview.get("parameters", {}).get("foundation_pit", {})
        base = foundation.get("sensitivity", {}).get("base", {})
        steps = [
            _step("CALC-001", "one_at_a_time_sensitivity", {"baseline_value": float(base.get("combined_kgco2e", 0.0)), "baseline_parameter": float(base.get("plan_utilisation_factor", 0.9)), "parameter_values": values, "parameter_name": "plan_utilisation_factor", "result_unit": "kgCO2e"}, "Evaluate every requested utilisation point."),
            _step("CALC-002", "rank_named_values", {"values": {"$named_points": {"step_id": "CALC-001", "records_field": "points", "name_field": "parameter_value", "value_field": "result"}}, "unit": "kgCO2e", "descending": False, "limit": len(values)}, "Select the minimum-carbon point with a deterministic ranking.", ["CALC-001"]),
            _step("CALC-003", "project_foundation_geometry", {"gross_envelope_m3": float(foundation.get("geometry", {}).get("gross_envelope_m3", 0.0)), "plan_utilisation_factor": {"$ref": "CALC-002.ranked.0.name"}, "backfill_ratio": float(parameters.get("backfill_ratio", 0.0)), "loose_volume_factor": float(parameters.get("loose_volume_factor", 1.0))}, "Project geometry only for the selected minimum-carbon point.", ["CALC-002"]),
        ]
    elif template == "reuse_reduction_composition":
        baseline = _match_number(
            rf"({_NUMBER})\s*kg\s*co2e?\s*(?:of\s+)?formwork",
            query,
            "formwork baseline",
        )
        cycles = _match_number(
            rf"(?:allocated|amorti[sz]ed)?\s*over\s*({_NUMBER})\s*(?:reuse\s*)?(?:uses?|cycles?)",
            query,
            "reuse cycles",
        )
        steel_fraction = _target_percentage(
            query,
            r"steel(?:[\s-]*factor)?",
            "steel reduction percentage",
        )
        steps = [
            _step("CALC-001", "amortise_reusable_resource", {"baseline_emission": baseline, "reuse_cycles": cycles}, "Calculate the reusable-formwork allocation."),
            _step("CALC-002", "factor_reduction", {"baseline_emission": _resource_baseline(context, "steel"), "reduction_fraction": steel_fraction}, "Calculate the steel branch."),
            _step("CALC-003", "apply_named_deltas", {"baseline_value": project_total, "deltas": {"formwork_reuse": {"$ref": "CALC-001.delta"}, "steel_reduction": {"$ref": "CALC-002.delta"}}, "unit": "kgCO2e"}, "Compose both deltas on the project baseline.", ["CALC-001", "CALC-002"]),
            _step("CALC-004", "compare_named_values", {"values": {"formwork_only_reduction": {"$ref": "CALC-001.reduction"}, "steel_only_reduction": {"$ref": "CALC-002.reduction"}, "coupled_reduction": {"$ref": "CALC-003.reduction"}}, "unit": "kgCO2e"}, "Compare the coupled option with each single measure.", ["CALC-001", "CALC-002", "CALC-003"]),
        ]
    elif template == "multi_transport_rank_post_retrieval":
        mass, unit, distances, factor = _transport_alternative_inputs(query)
        for index, distance in enumerate(distances, start=1):
            steps.append(_step(f"CALC-{index:03d}", "transport_emission", {"mass": mass, "mass_unit": unit, "distance_km": distance, "factor_kgco2e_per_tkm": factor}, f"Calculate the {distance:g} km A4 option."))
        result_steps = {f"CALC-{index:03d}": f"{distance:g} km" for index, distance in enumerate(distances, start=1)}
        rank_id = f"CALC-{len(steps) + 1:03d}"
        steps.append(_step(rank_id, "rank_named_values", {"values": {"$step_results": {"steps": result_steps, "field": "result"}}, "unit": "kgCO2e", "descending": False, "limit": len(distances)}, "Rank alternatives by calculated A4 carbon.", list(result_steps)))
        default_rows = context.get("material_transport_defaults", {}).get("rows", [])
        concrete_rows = [row for row in default_rows if "concrete" in str(row.get("material_name", "")).lower()]
        reference_row = (concrete_rows or default_rows or [{}])[0]
        default_distance = float(reference_row.get("default_distance_km", 0.0))
        steps.append(_step(f"CALC-{len(steps) + 1:03d}", "compare_to_reference", {"values": {f"{value:g} km": value for value in distances}, "reference_value": default_distance, "reference_name": str(reference_row.get("material_code") or "governed_concrete_default"), "tolerance": 0.0}, "Validate every calculated option against the retrieved concrete default.", [rank_id]))
    else:
        return _assemble_compound_program(query, contract, context)

    return {
        "compiler": "typed_dependency_compiler",
        "steps": steps,
        "final_result_template": {
            "tool": "dependency_ordered_calculation_program",
            "program_specification": program_specification,
            "status": "complete",
        },
    }


def _correction_impact_program(
    query: str,
    contract: QueryContract,
    context: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    changes = overlay.get("changes", [])
    if not changes:
        # A transport comparison with no correction delta can still form a calculation program.
        try:
            _mass, _unit, distances, _factor = _transport_alternative_inputs(query)
        except CalculationPlanningError:
            distances = []
        if distances:
            return _assemble_compound_program(query, contract, context)
        raise CalculationPlanningError("MISSING_SCENARIO_CHANGE", "No correction delta was supplied.")
    change = changes[0]
    target = str(change.get("target", ""))
    new_distance = float(change.get("parameters", {}).get("distance_km"))
    default_rows = context.get("material_transport_defaults", {}).get("rows", [])

    contract_payload = contract.model_dump(mode="json")
    requested_labels = transport_material_labels(target, contract_payload)
    matching = matching_transport_defaults(
        default_rows,
        requested_labels,
        query=contract.query,
    )
    if len(matching) != 1:
        requested = ", ".join(dict.fromkeys(requested_labels)) or "the requested material"
        raise CalculationPlanningError(
            "TRANSPORT_DEFAULT_NOT_UNIQUE",
            f"Exactly one governed transport default is required for {requested}.",
            ["material_transport_default"],
        )
    resolved_row = matching[0]
    material_code = str(resolved_row["material_code"]).upper()
    material_name = str(resolved_row.get("material_name", material_code))
    old_distance = float(resolved_row["default_distance_km"])
    return {
        "compiler": "typed_dependency_compiler",
        "steps": [
            _step("CALC-001", "compare_named_values", {"values": {"current_default": old_distance, "proposed_default": new_distance}, "unit": "km", "baseline_name": "current_default"}, "Compare the proposed correction with the current governed default."),
            _step("CALC-002", "calculate_ratio", {"numerator": new_distance, "denominator": old_distance, "numerator_name": "new_distance_km", "denominator_name": "old_distance_km"}, "Calculate the requested distance multiplier.", ["CALC-001"]),
        ],
        "final_result_template": {
            "tool": "correction_impact_then_preview",
            "material_code": material_code,
            "old_distance_km": old_distance,
            "new_distance_km": new_distance,
            "distance_multiplier": {"$ref": "CALC-002.ratio"},
            "status": "pending_confirmation",
        },
        "presentation_extensions": {
            "material_name": material_name,
            "resolved_target": f"material_transport_default:{material_code}",
        },
    }


def compile_mas_calculation_program(
    query: str,
    mode: QueryMode,
    query_contract: dict[str, Any],
    context: dict[str, Any],
    overlay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a query contract into executable, dependency-ordered tool calls.

    This compiler is used only by the coordinated MAS Calculator.  It does not
    calculate carbon itself: it binds governed Reader/Extractor outputs to the
    registered deterministic functions and leaves every numerical transition
    visible in the shared calculation program.
    """

    contract = QueryContract.model_validate(query_contract)
    overlay = overlay or {}
    if contract.compound:
        return _compound_program(query, contract, context)
    if contract.correction_impact_preview:
        return _correction_impact_program(query, contract, context, overlay)
    call = _simple_call(query, mode, contract, context, overlay)
    return {
        "compiler": "typed_dependency_compiler",
        "steps": [_step("CALC-001", call.tool_name, call.arguments, call.rationale)],
        "final_result_template": {"$ref": "CALC-001"},
    }


def _lookup_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def resolve_program_bindings(value: Any, results: dict[str, dict[str, Any]]) -> Any:
    """Resolve explicit result references immediately before a tool invocation."""

    if isinstance(value, list):
        return [resolve_program_bindings(item, results) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$ref"}:
        reference = str(value["$ref"])
        step_id, _, path = reference.partition(".")
        if step_id not in results:
            raise CalculationPlanningError(
                "CALCULATION_DEPENDENCY_MISSING",
                f"Calculation result {step_id} is not available for {reference}.",
            )
        return _lookup_path(results[step_id], path)
    if set(value) == {"$named_points"}:
        specification = value["$named_points"]
        source = results[str(specification["step_id"])]
        records = _lookup_path(source, str(specification["records_field"]))
        return {
            str(record[specification["name_field"]]): float(record[specification["value_field"]])
            for record in records
        }
    if set(value) == {"$step_results"}:
        specification = value["$step_results"]
        field = str(specification["field"])
        return {
            str(label): float(_lookup_path(results[str(step_id)], field))
            for step_id, label in specification["steps"].items()
        }
    return {key: resolve_program_bindings(item, results) for key, item in value.items()}
