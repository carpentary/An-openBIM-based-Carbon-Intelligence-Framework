from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .agents import AgentServices
from .config import ProjectConfig
from .evaluation import GroundTruthSuite
from .runtime import build_workflow


FactInventory = dict[str, tuple[Any, str | None]]


def _number(value: Any) -> float:
    return float(value)


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _prediction_path_value(report: dict[str, Any], path: str) -> Any:
    if path == "$issue_codes":
        return sorted(
            {
                item.get("code")
                for item in report.get("issues", [])
                if item.get("code")
            }
        )
    current: Any = report
    for token in path.split("."):
        if isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
        elif isinstance(current, list):
            # Numeric path segments index lists of calculation steps.
            if not token.isdigit():
                return None
            index = int(token)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _task_graph_prediction(tdg: dict[str, Any]) -> dict[str, Any]:
    tasks = list(tdg.get("tasks", []))
    node_to_agent = {
        str(task.get("node_id")): str(task.get("agent"))
        for task in tasks
        if task.get("node_id") and task.get("agent")
    }
    edges = []
    for task in tasks:
        target = node_to_agent.get(str(task.get("node_id")))
        if not target:
            continue
        for dependency in task.get("dependencies", []):
            source = node_to_agent.get(str(dependency))
            if source:
                edges.append([source, target])
    return {
        "agents": [str(task.get("agent")) for task in tasks if task.get("agent")],
        "dependency_edges": edges,
    }


def _stage_totals(overview: dict[str, Any]) -> dict[str, float]:
    return {
        str(item["stage"]): _number(item["value"])
        for item in overview.get("totals", [])
        if item.get("value") is not None
    }


def _transport_leg(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": row.get("scope"),
        "material_code": row.get("material_code"),
        "material_name": row.get("material_name"),
        "source_quantity": _number(row.get("source_quantity")),
        "source_unit": row.get("source_unit"),
        "mass_t": _number(row.get("mass_t")),
        "distance_km": _number(row.get("distance_km")),
        "factor_kgco2e_per_tkm": _number(row.get("factor_kgco2e_per_tkm")),
        "vehicle_code": row.get("vehicle_code"),
        "emission_kgco2e": _number(row.get("emission_kgco2e")),
    }


def _scenario_facts(calculation: dict[str, Any], inventory: FactInventory) -> None:
    tool = calculation.get("tool")
    if tool == "factor_reduction":
        if not all(key in calculation for key in ("baseline", "alternative", "delta", "reduction", "percentage_change")):
            return
        value = {key: _number(calculation[key]) for key in ("baseline", "alternative", "delta", "reduction")}
        inventory["scenario.concrete_factor_reduction_10pct"] = (value, "kgCO2e")
        inventory["scenario.concrete_factor_reduction_10pct.percentage_change"] = (
            _number(calculation["percentage_change"]),
            "percent",
        )
    elif tool == "material_substitution":
        if not all(key in calculation for key in ("baseline", "alternative", "delta", "reduction", "percentage_change")):
            return
        value = {key: _number(calculation[key]) for key in ("baseline", "alternative", "delta", "reduction")}
        inventory["scenario.ggbs_substitution_30pct"] = (value, "kgCO2e")
        inventory["scenario.ggbs_substitution_30pct.percentage_change"] = (
            _number(calculation["percentage_change"]),
            "percent",
        )
    elif tool == "transport_emission":
        # Only extract a scenario fact if the result has the governed value field.
        if "result" in calculation:
            inventory["scenario.transport_50t_40km"] = (_number(calculation["result"]), "kgCO2e")


def build_fact_inventory(query: str, report: dict[str, Any]) -> FactInventory:
    data = report.get("data", {})
    context = data.get("retrieved_context", {})
    overview = context.get("project_overview", {})
    calculation = data.get("calculation_results", {})
    inventory: FactInventory = {}

    stages = _stage_totals(overview)
    if stages:
        inventory["project.stage_totals"] = (stages, "kgCO2e")
        for stage, value in stages.items():
            inventory[f"project.{stage}"] = (value, "kgCO2e")
        inventory["assessment.included_stages"] = (list(stages), None)
    if overview.get("combined_total_kgco2e") is not None:
        inventory["project.total_kgco2e"] = (_number(overview["combined_total_kgco2e"]), "kgCO2e")
    profile = overview.get("profile", {})
    if profile.get("applicability_scope") is not None:
        inventory["assessment.applicability_scope"] = (profile["applicability_scope"], None)
    if overview.get("formulae"):
        inventory["assessment.formulae"] = (dict(overview["formulae"]), None)

    scope_stages = {
        str(scope): {str(stage): _number(value) for stage, value in values.items()}
        for scope, values in overview.get("scope_stage_totals_kgco2e", {}).items()
    }
    if scope_stages:
        scope_totals = {scope: sum(values.values()) for scope, values in scope_stages.items()}
        inventory["systems.total_kgco2e"] = (scope_totals, "kgCO2e")
        for scope, values in scope_stages.items():
            inventory[f"systems.{scope}.stage_totals"] = (values, "kgCO2e")
            inventory[f"systems.{scope}.total"] = (scope_totals[scope], "kgCO2e")

    if calculation.get("tool") == "calculate_shares":
        shares = {item["name"]: _number(item["share_fraction"]) for item in calculation.get("shares", [])}
        if set(shares) == set(stages):
            inventory["project.stage_shares"] = (shares, "fraction")
        elif set(shares) == set(scope_stages):
            inventory["systems.total_shares"] = (shares, "fraction")
    if calculation.get("tool") == "compare_named_values":
        maximum = calculation.get("maximum", {})
        if maximum.get("name") in stages:
            inventory["project.largest_stage"] = (maximum["name"], None)
            inventory["project.largest_stage_value"] = (_number(maximum["value"]), "kgCO2e")
        elif maximum.get("name") in scope_stages:
            inventory["systems.largest_scope"] = (maximum["name"], None)
            inventory["systems.largest_scope_value"] = (_number(maximum["value"]), "kgCO2e")

    if context.get("ifc_inventory"):
        source = context["ifc_inventory"].get("data", {})
        counts = {row["role"]: int(row["scoped_product_count"]) for row in source.get("sources", [])}
        counts["combined"] = int(source.get("combined_product_count", 0))
        inventory["ifc.product_counts"] = (counts, "count")

    foundation = overview.get("foundation_pit", {})
    geometry = foundation.get("geometry", {})
    for field, unit in (
        ("excavation_m3", "m3"),
        ("gross_envelope_m3", "m3"),
        ("backfill_m3", "m3"),
        ("offsite_disposal_in_situ_m3", "m3"),
        ("offsite_transport_loose_m3", "m3"),
        ("offsite_haul_distance_km", "km"),
    ):
        if geometry.get(field) is not None:
            inventory[f"foundation.geometry.{field}"] = (_number(geometry[field]), unit)
    foundation_parameters = overview.get("parameters", {}).get("foundation_pit", {})
    parameter_fields = (
        "backfill_ratio",
        "high_plan_utilisation_factor",
        "loose_volume_factor",
        "low_plan_utilisation_factor",
        "offsite_haul_distance_km",
        "plan_utilisation_factor",
    )
    if foundation_parameters:
        inventory["foundation.parameters"] = (
            {field: _number(foundation_parameters[field]) for field in parameter_fields},
            None,
        )
    if foundation.get("sensitivity"):
        inventory["foundation.sensitivity"] = (foundation["sensitivity"], None)
    if foundation:
        inventory["foundation.process_counts"] = (
            {
                "work_package_count": int(foundation.get("work_package_count", 0)),
                "resource_flow_count": int(foundation.get("resource_flow_count", 0)),
            },
            "count",
        )
        inventory["foundation.stage_totals"] = (
            {key: _number(value) for key, value in foundation.get("stage_totals_kgco2e", {}).items()},
            "kgCO2e",
        )
    foundation_rows = context.get("foundation_cpd_efd", {}).get("rows", [])
    if foundation_rows:
        inventory["foundation.cpd_codes"] = (
            sorted({row["cpd_code"] for row in foundation_rows if row.get("cpd_code")}),
            None,
        )

    enclosure_data = overview.get("enclosure_structure", {})
    coverage = enclosure_data.get("mapping_coverage", {})
    if coverage:
        coverage_fields = (
            "total_elements",
            "applicable_elements",
            "mapped_applicable_elements",
            "not_applicable_elements",
            "missing_quantity_elements",
        )
        inventory["enclosure.coverage_counts"] = (
            {field: int(coverage[field]) for field in coverage_fields},
            "count",
        )
        inventory["enclosure.mapping_rate"] = (_number(coverage["mapping_rate"]), "fraction")
        inventory["enclosure.unmapped_applicable"] = (
            _number(coverage["applicable_elements"]) - _number(coverage["mapped_applicable_elements"]),
            "count",
        )
        inventory["enclosure.element_type_counts"] = (
            {key: int(value) for key, value in coverage.get("element_type_counts", {}).items()},
            "count",
        )
        inventory["enclosure.variant_counts"] = (
            {key: int(value) for key, value in coverage.get("variant_counts", {}).items()},
            "count",
        )
        inventory["enclosure.diaphragm_wall_volume_m3"] = (
            _number(coverage["diaphragm_wall_volume_m3"]),
            "m3",
        )
        inventory["enclosure.diaphragm_wall_length_m"] = (
            _number(coverage["diaphragm_wall_length_m"]),
            "m",
        )
        largest_type, largest_count = max(
            coverage.get("element_type_counts", {}).items(),
            key=lambda item: (int(item[1]), item[0]),
        )
        inventory["enclosure.largest_type"] = (largest_type, None)
        inventory["enclosure.largest_type_count"] = (_number(largest_count), "count")
        variants = coverage.get("variant_counts", {})
        inventory["enclosure.section_steel_strut_variants"] = (
            {
                key: int(value)
                for key, value in variants.items()
                if key.startswith("Type_SectionSteelStrut::")
            },
            "count",
        )
    if enclosure_data:
        inventory["enclosure.process_counts"] = (
            {
                "work_package_count": int(enclosure_data.get("work_package_count", 0)),
                "resource_flow_count": int(enclosure_data.get("resource_flow_count", 0)),
            },
            "count",
        )
    enclosure_rows = context.get("enclosure_cpd_efd", {}).get("rows", [])
    if enclosure_rows:
        inventory["enclosure.diaphragm_wall_cpd_codes"] = (
            sorted({row["cpd_code"] for row in enclosure_rows if row.get("cpd_code")}),
            None,
        )

    transport = overview.get("transport", {})
    if transport:
        inventory["transport.total_a4"] = (
            _number(stages.get("A4_TransportationStage", sum(transport.get("A4_by_scope_kgco2e", {}).values()))),
            "kgCO2e",
        )
        inventory["transport.coverage_counts"] = (
            {
                "leg_count": int(transport.get("leg_count", 0)),
                "material_default_count": int(transport.get("material_default_count", 0)),
                "vehicle_factor_count": int(transport.get("vehicle_factor_count", 0)),
                "exclusion_count": len(transport.get("exclusions", [])),
            },
            "count",
        )
        inventory["transport.a4_by_scope"] = (
            {key: _number(value) for key, value in transport.get("A4_by_scope_kgco2e", {}).items()},
            "kgCO2e",
        )
    legs = context.get("a4_transport_legs", {}).get("rows", [])
    if legs:
        governed_legs = [_transport_leg(row) for row in legs]
        governed_legs.sort(key=lambda row: (-row["emission_kgco2e"], row["scope"], row["material_code"]))
        inventory["transport.largest_leg"] = (governed_legs[0], None)
        inventory["transport.top_five_legs"] = (governed_legs[:5], None)
        distance_counts = Counter(str(_number(row["distance_km"])) for row in governed_legs)
        vehicle_counts = Counter(str(row["vehicle_code"]) for row in governed_legs)
        inventory["transport.distance_leg_counts"] = (dict(sorted(distance_counts.items())), "count")
        inventory["transport.vehicle_leg_counts"] = (dict(sorted(vehicle_counts.items())), "count")
        for row in governed_legs:
            if row["scope"] == "foundation_pit":
                inventory[f"transport.foundation.{row['material_code']}"] = (row, None)
            elif row["scope"] == "enclosure_structure":
                inventory[f"transport.enclosure.{row['material_code']}"] = (row, None)

    defaults = context.get("material_transport_defaults", {}).get("rows", [])
    vehicles = {
        row["vehicle_code"]: row
        for row in context.get("vehicle_factors", {}).get("rows", [])
    }
    for row in defaults:
        vehicle = vehicles.get(row.get("vehicle_code"), {})
        value = {
            "material_code": row.get("material_code"),
            "source_unit": row.get("source_unit"),
            "vehicle_code": row.get("vehicle_code"),
            "default_distance_km": _number(row.get("default_distance_km")),
            "mass_t_per_source_unit": _number(row.get("mass_t_per_source_unit")),
            "transport_applicable": _boolean(row.get("transport_applicable")),
            "factor_kgco2e_per_tkm": _number(vehicle.get("factor_kgco2e_per_tkm")),
        }
        inventory[f"transport.default.{row['material_code']}"] = (value, None)

    hotspots = context.get("hotspots", [])
    if hotspots:
        governed_hotspots = []
        for index, row in enumerate(hotspots[:10], start=1):
            governed_hotspots.append(
                {
                    "rank": int(row.get("rank", index)),
                    "element": row.get("element"),
                    "element_uri": row.get("element_uri"),
                    "value": _number(row.get("value")),
                    "unit": row.get("unit"),
                    "stages": list(row.get("stages", [])),
                }
            )
        inventory["ckg.top_ten_elements"] = (governed_hotspots, None)
        inventory["ckg.top_element"] = (governed_hotspots[0], None)
    trace_data = context.get("representative_element_trace", {})
    if trace_data:
        inventory["ckg.representative_trace"] = (
            {
                "element_uri": trace_data.get("element_uri"),
                "system_uri": trace_data.get("system_uri"),
                "semantic_route": trace_data.get("semantic_route"),
                "element_metadata": trace_data.get("element_metadata"),
                "chain_row_count": len(trace_data.get("chain_rows", [])),
                "distinct_work_packages": int(trace_data.get("distinct_work_packages", 0)),
                "distinct_resources": int(trace_data.get("distinct_resources", 0)),
                "distinct_emissions": int(trace_data.get("distinct_emissions", 0)),
                "emission_sum_in_returned_rows": _number(trace_data.get("emission_sum_in_returned_rows", 0)),
                "truncated": bool(trace_data.get("truncated", False)),
            },
            None,
        )
    resource = context.get("resource_breakdown", {})
    if resource:
        inventory["ckg.concrete_a1_a3_breakdown"] = (
            {
                "term": resource.get("term"),
                "items": resource.get("items", []),
                "total_emission": _number(resource.get("total_emission", 0)),
                "unit": resource.get("unit"),
            },
            None,
        )
    work_packages = context.get("work_packages", [])
    if work_packages:
        inventory["ckg.top_work_package_templates"] = (work_packages[:10], None)

    subontology = data.get("subontology", {})
    if len(subontology.get("profile_ids", [])) == 1:
        profile_id = subontology["profile_ids"][0]
        inventory[f"ontology.{profile_id}.required_terms"] = (
            list(subontology.get("ontology_terms", [])),
            None,
        )

    _scenario_facts(calculation, inventory)

    lowered = query.lower()
    if re.search(r"(?<![a-z0-9])b1(?![a-z0-9])", lowered):
        inventory["decision.B1_Use"] = (
            {"decision": "reject", "reason": "outside_active_boundary", "requested_stage": "B1_Use"},
            None,
        )
    if re.search(r"(?<![a-z0-9])c1(?![a-z0-9])", lowered):
        inventory["decision.C1_Deconstruction"] = (
            {"decision": "reject", "reason": "outside_active_boundary", "requested_stage": "C1_Deconstruction"},
            None,
        )
    if re.search(r"(?<![a-z0-9])(?:module\s*)?d(?![a-z0-9])", lowered) or "d模块" in lowered:
        inventory["decision.D_BeyondSystemBoundary"] = (
            {"decision": "reject", "reason": "outside_active_boundary", "requested_stage": "D_BeyondSystemBoundary"},
            None,
        )
    issue_codes = {issue.get("code") for issue in report.get("issues", [])}
    if "MISSING_A4_PARAMETERS" in issue_codes:
        inventory["decision.missing_a4_fields"] = (
            ["mass", "distance_km", "factor_kgco2e_per_tkm"],
            None,
        )
    if "MISSING_SUBSTITUTION_PARAMETERS" in issue_codes:
        inventory["decision.missing_substitution_fields"] = (
            ["substitution_fraction", "baseline_factor", "alternative_factor"],
            None,
        )
    if "MISSING_A4_PARAMETERS" in issue_codes and report.get("mode") == "KC":
        inventory["decision.missing_persistent_transport_fields"] = (
            ["target_entity", "distance_km"],
            None,
        )
    return inventory


def provenance_steps_from_report(
    report: dict[str, Any],
    *,
    complex_contract: bool = False,
    ontology_mode: str | None = None,
) -> list[str]:
    metadata_mode = ontology_mode or (
        report.get("data", {})
        .get("metadata_retrieval", {})
        .get("ontology_mode", "full")
    )
    method_contract: list[str] = []
    if complex_contract:
        method_contract = ["interpret_query_contract"]
        method_contract.append(
            "type_against_static_ontology_schema"
            if metadata_mode == "schema_only"
            else "resolve_query_specific_subontology"
        )
        method_contract.extend(
            [
                "retrieve_fingerprinted_sources",
                "execute_dependency_ordered_operations",
                "verify_answer_and_boundary",
            ]
        )
    if report.get("status") == "blocked":
        return list(dict.fromkeys(method_contract + [
            "identify_query_scope",
            "retrieve_governed_facts",
            "apply_required_semantic_path",
            "validate_unit_and_boundary",
            "assemble_grounded_answer",
        ]))
    calculation = report.get("data", {}).get("calculation_results", {})
    tool = calculation.get("tool")
    if tool == "factor_reduction":
        specific = [
            "resolve_concrete_baseline",
            "validate_reduction_fraction",
            "apply_factor_reduction",
            "compare_baseline_and_alternative",
            "keep_overlay_transient",
        ]
        return list(dict.fromkeys(method_contract + specific))
    if tool == "material_substitution":
        specific = [
            "resolve_concrete_baseline",
            "validate_substitution_fraction",
            "validate_factor_units",
            "apply_substitution_multiplier",
            "compare_baseline_and_alternative",
        ]
        return list(dict.fromkeys(method_contract + specific))
    if tool == "transport_emission":
        specific = [
            "validate_mass_and_unit",
            "validate_distance",
            "validate_tkm_factor",
            "normalise_mass_to_tonnes",
            "multiply_mass_distance_factor",
        ]
        return list(dict.fromkeys(method_contract + specific))
    return list(dict.fromkeys(method_contract + [
        "identify_query_scope",
        "retrieve_governed_facts",
        "apply_required_semantic_path",
        "validate_unit_and_boundary",
        "assemble_grounded_answer",
    ]))


def _runtime_evidence(state: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Summarise query, inference and validation evidence for one case."""

    calls = report.get("data", {}).get("tool_calls", [])
    parity: dict[str, int] = {}
    swrl_store: dict[str, Any] | None = None
    transaction_gate = "not_applicable"
    for call in calls:
        result = call.get("result") or {}
        tool = call.get("tool_name")
        if tool == "execute_ckg_sparql_template":
            note = result.get("sparql_parity") or {}
            status = str(note.get("status", "absent"))
            parity[status] = parity.get(status, 0) + 1
        if tool == "check_knowledge_readiness":
            swrl_store = result.get("swrl_inference_store") or swrl_store
        error = str(call.get("error") or "")
        if "TRANSACTION_SHACL_VIOLATION" in error:
            transaction_gate = "violation_blocked"
        elif "TRANSACTION_SHACL_GATE_UNAVAILABLE" in error:
            transaction_gate = "unavailable"
    if report.get("transaction_preview") is not None and transaction_gate == "not_applicable":
        transaction_gate = "passed"
    contract = state.get("query_contract", {}) or {}
    return {
        "sparql_parity": parity,
        "swrl_inference_store": swrl_store,
        "transaction_shacl_gate": transaction_gate,
        "discarded_semantic_risks": contract.get("discarded_semantic_risks", []),
        "llm_cache_split": {
            key: value
            for key, value in (next(
                (
                    event.get("details", {}).get("llm_api_usage", {})
                    for event in report.get("provenance", [])
                    if event.get("agent") == "orchestrator"
                ),
                {},
            )).items()
            if "cache" in key or key in {"prompt_tokens", "completion_tokens", "request_count"}
        },
    }


def run_query_suite(
    config: ProjectConfig,
    services: AgentServices,
    suite: GroundTruthSuite,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = build_workflow(config=config, services=services)
    predictions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    complex_contract = any(case.reasoning_ground_truth.steps for case in suite.cases)
    for case in suite.cases:
        variant = next(item for item in case.query_variants if item.language == language)
        _case_started = time.perf_counter()
        state = workflow.invoke({"query": variant.text})
        _case_duration = round(time.perf_counter() - _case_started, 3)
        report = state["final_report"]
        orchestrator_details = next(
            (
                event.get("details", {})
                for event in report.get("provenance", [])
                if event.get("agent") == "orchestrator"
            ),
            {},
        )
        metadata_details = next(
            (
                event.get("details", {})
                for event in report.get("provenance", [])
                if event.get("agent") == "metadata"
            ),
            {},
        )
        metadata_backend = metadata_details.get("backend", {})
        inventory = build_fact_inventory(variant.text, report)
        facts = []
        missing_fact_ids = []
        for assertion in case.assertions:
            if assertion.prediction_path:
                value = _prediction_path_value(report, assertion.prediction_path)
                predicted = None if value is None else (value, assertion.unit)
            else:
                predicted = inventory.get(assertion.fact_key or assertion.assertion_id)
            if predicted is None:
                missing_fact_ids.append(assertion.assertion_id)
                value, unit = None, None
            else:
                value, unit = predicted
            facts.append({"assertion_id": assertion.assertion_id, "value": value, "unit": unit})
        subontology = report.get("data", {}).get("subontology", {})
        calls = report.get("data", {}).get("tool_calls", [])
        task_graph = _task_graph_prediction(state.get("tdg", {}))
        prediction = {
            "case_id": case.case_id,
            "status": report.get("status"),
            "issue_codes": sorted({item.get("code") for item in report.get("issues", []) if item.get("code")}),
            "facts": facts,
            "retrieval": {
                "ontology_terms": subontology.get("ontology_terms", []),
                "semantic_path_ids": subontology.get("semantic_path_ids", []),
                "data_source_ids": subontology.get("data_source_ids", []),
            },
            "provenance_step_ids": provenance_steps_from_report(
                report,
                complex_contract=complex_contract,
            ),
            "observed_mode": report.get("mode"),
            "selected_plan_id": report.get("data", {}).get("selected_plan_id"),
            "tool_names": [call.get("tool_name") for call in calls],
            "task_graph": task_graph,
            "requires_human_confirmation": bool(report.get("requires_human_confirmation")),
            "calculation": report.get("data", {}).get("calculation_results", {}),
        }
        predictions.append(prediction)
        semantic_contract = state.get("query_contract", {})
        executions.append(
            {
                "case_id": case.case_id,
                "language": language,
                "query": variant.text,
                "duration_seconds": _case_duration,
                "runtime_evidence": _runtime_evidence(state, report),
                "expected_mode": case.expected_mode,
                "observed_mode": report.get("mode"),
                "expected_status": case.expected_status,
                "observed_status": report.get("status"),
                "observed_issue_codes": prediction["issue_codes"],
                "missing_fact_ids": missing_fact_ids,
                "tool_names": prediction["tool_names"],
                "all_tool_calls_succeeded": all(call.get("status") == "success" for call in calls),
                "failed_tool_calls": [
                    {
                        "call_id": call.get("call_id"),
                        "agent": call.get("agent"),
                        "tool_name": call.get("tool_name"),
                        "error": call.get("error"),
                    }
                    for call in calls
                    if call.get("status") != "success"
                ],
                "selected_plan_id": prediction["selected_plan_id"],
                "task_graph": task_graph,
                "required_tool_names": case.task_ground_truth.required_tool_names,
                "required_tools_present": set(case.task_ground_truth.required_tool_names).issubset(
                    set(prediction["tool_names"])
                ),
                "semantic_contract": {
                    "task_family": semantic_contract.get("task_family"),
                    "primary_operator": semantic_contract.get("primary_operator"),
                    "operation_ids": semantic_contract.get("operation_ids", []),
                    "target_scopes": semantic_contract.get("target_scopes", []),
                    "target_stages": semantic_contract.get("target_stages", []),
                    "resource_targets": semantic_contract.get("resource_targets", []),
                    "explicit_identifiers": semantic_contract.get("explicit_identifiers", []),
                    "planner_profile_ids": semantic_contract.get("planner_profile_ids", []),
                    "required_capabilities": semantic_contract.get("required_capabilities", []),
                    "semantic_risks": semantic_contract.get("semantic_risks", []),
                    "parser_backend": semantic_contract.get("parser_backend"),
                },
                "planning_route_source": orchestrator_details.get("route_source"),
                "model_proposed_mode": orchestrator_details.get("model_proposed_mode"),
                "planning_arbitration": orchestrator_details.get("planning_arbitration"),
                "structured_output_attempts": orchestrator_details.get(
                    "structured_output_attempts", 0
                ),
                "transport_request_attempts": orchestrator_details.get(
                    "transport_request_attempts", 0
                ),
                "transport_retry_count": orchestrator_details.get(
                    "transport_retry_count", 0
                ),
                "llm_api_usage": orchestrator_details.get("llm_api_usage", {}),
                "llm_cumulative_api_usage": orchestrator_details.get(
                    "llm_cumulative_api_usage", {}
                ),
                "llm_token_budget": orchestrator_details.get("llm_token_budget"),
                "metadata_dense_backend": metadata_backend.get("dense_backend"),
                "metadata_reranker_backend": metadata_backend.get("reranker_backend"),
                "metadata_fallback_used": metadata_backend.get("fallback_used"),
            }
        )
    model_routed = [item for item in executions if item["model_proposed_mode"] is not None]
    metadata_routed = [item for item in executions if item["metadata_dense_backend"] is not None]
    payload = {
        "dataset_id": suite.dataset_id,
        "dataset_digest": suite.dataset_digest,
        "prediction_schema_version": "1.0.0",
        "language": language,
        "predictions": predictions,
    }
    execution_report = {
        "language": language,
        "case_count": len(executions),
        "mode_accuracy": sum(item["expected_mode"] == item["observed_mode"] for item in executions) / len(executions),
        "status_accuracy": sum(item["expected_status"] == item["observed_status"] for item in executions) / len(executions),
        "complete_fact_coverage": sum(not item["missing_fact_ids"] for item in executions) / len(executions),
        "tool_call_success_rate": sum(item["all_tool_calls_succeeded"] for item in executions) / len(executions),
        "required_tool_recall": sum(item["required_tools_present"] for item in executions) / len(executions),
        "model_route_proposal_agreement": (
            sum(item["model_proposed_mode"] == item["observed_mode"] for item in model_routed)
            / len(model_routed)
            if model_routed
            else None
        ),
        "governed_route_override_count": sum(
            item["planning_arbitration"] == "governed_deterministic_override"
            for item in model_routed
        ),
        "structured_output_retry_count": sum(
            item["structured_output_attempts"] > 1 for item in model_routed
        ),
        "transport_retry_count": sum(
            item["transport_retry_count"] for item in model_routed
        ),
        "transport_retried_case_count": sum(
            item["transport_retry_count"] > 0 for item in model_routed
        ),
        "llm_prompt_tokens": sum(
            item["llm_api_usage"].get("prompt_tokens", 0) for item in model_routed
        ),
        "llm_completion_tokens": sum(
            item["llm_api_usage"].get("completion_tokens", 0) for item in model_routed
        ),
        "llm_total_tokens": sum(
            item["llm_api_usage"].get("total_tokens", 0) for item in model_routed
        ),
        "llm_prompt_cache_hit_tokens": sum(
            item["llm_api_usage"].get("prompt_cache_hit_tokens", 0) for item in model_routed
        ),
        "llm_prompt_cache_miss_tokens": sum(
            item["llm_api_usage"].get("prompt_cache_miss_tokens", 0) for item in model_routed
        ),
        "bge_backend_rate": (
            sum(
                item["metadata_dense_backend"] == "BGE-M3"
                and item["metadata_reranker_backend"] == "BGE-reranker-v2-m3"
                and item["metadata_fallback_used"] is False
                for item in metadata_routed
            )
            / len(metadata_routed)
            if metadata_routed
            else None
        ),
        "failures": [
            item
            for item in executions
            if item["expected_mode"] != item["observed_mode"]
            or item["expected_status"] != item["observed_status"]
            or item["missing_fact_ids"]
            or not item["all_tool_calls_succeeded"]
            or not item["required_tools_present"]
        ],
        "per_case": executions,
    }
    return payload, execution_report


def load_query_suite(dataset_path: Path) -> GroundTruthSuite:
    return GroundTruthSuite.model_validate(json.loads(dataset_path.read_text(encoding="utf-8")))
