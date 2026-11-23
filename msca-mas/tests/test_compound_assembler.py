"""Operator-algebra compound assembler: unknown combinations now compile."""

import pytest

from msca_mas.mas_program import (
    CalculationPlanningError,
    compile_mas_calculation_program,
)
from msca_mas.models import QueryMode
from msca_mas.query_contract import parse_query_contract


def _context() -> dict:
    return {
        "project_overview": {
            "totals": [
                {"stage": "A1-A3_ProductStage", "value": 1000.0, "unit": "Unit_KgCO2e"},
                {"stage": "A4_TransportationStage", "value": 200.0, "unit": "Unit_KgCO2e"},
                {"stage": "A5_ConstructionStage", "value": 100.0, "unit": "Unit_KgCO2e"},
            ],
            "combined_total_kgco2e": 1300.0,
            "scope_stage_totals_kgco2e": {"main_structure": {"A1-A3_ProductStage": 900.0}},
        },
        "resource_breakdowns": {
            "concrete": {"total_emission": 800.0},
        },
        "resource_breakdown": {"term": "concrete", "total_emission": 800.0},
        "material_transport_defaults": {
            "rows": [
                {"material_code": "M-3-1", "material_name": "Concrete C30", "default_distance_km": 40.0}
            ]
        },
    }


def _compile(query: str, template: str | None = None):
    contract = parse_query_contract(query, QueryMode.STRATEGIC_ANALYSIS)
    payload = contract.model_dump(mode="json")
    if template is not None:
        payload["compound"] = True
        payload["compound_template"] = template
    return compile_mas_calculation_program(query, QueryMode.STRATEGIC_ANALYSIS, payload, _context())


def test_known_templates_keep_their_exact_step_shape():
    program = _compile(
        "For a revised concrete option, 37% GGBS substitution with factors "
        "342.028841 and 96 kgCO2e per tonne. Calculate the substituted option, "
        "carry only its delta into the station total, compare and rank the "
        "baseline and revised project totals.",
        template="substitution_rank_project_delta",
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools == ["material_substitution", "apply_named_deltas", "compare_named_values", "rank_named_values"]


def test_unknown_template_with_transport_intents_assembles():
    program = _compile(
        "A logistics workshop has one 96.5 t concrete delivery and three A4 "
        "routes of 34, 67.5 and 112 km, each using 0.4745 kgCO2e per t-km. "
        "Calculate the A4 result for each route and rank the three from lowest.",
        template="compound_scenario_unknown",
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools[:3] == ["transport_emission"] * 3
    assert "rank_named_values" in tools
    ids = {step["step_id"] for step in program["steps"]}
    for step in program["steps"]:
        for dependency in step.get("dependencies", []):
            assert dependency in ids


def test_unknown_template_with_reduction_intents_assembles_and_composes():
    program = _compile(
        "The coupled option reduces concrete by 10% and lowers the A4 stage "
        "by 15%; carry both deltas into the station total and compare the "
        "baseline and revised project totals.",
        template="compound_scenario_unknown",
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools.count("factor_reduction") == 2
    assert "apply_named_deltas" in tools
    assert "compare_named_values" in tools
    compose = next(s for s in program["steps"] if s["tool_name"] == "apply_named_deltas")
    assert len(compose["arguments"]["deltas"]) == 2
    assert compose["dependencies"] == [
        s["step_id"] for s in program["steps"] if s["tool_name"] == "factor_reduction"
    ]


def test_no_calculation_intent_still_fails_closed():
    with pytest.raises(CalculationPlanningError) as excinfo:
        _compile("Please summarise the ontology classes.", template="compound_scenario_unknown")
    assert excinfo.value.code == "UNSUPPORTED_COMPOUND_PROGRAM"


def test_substitution_variant_without_unit_word_assembles():
    program = _compile(
        "Work through the coupled option applying 32% GGBS substitution with "
        "factors 342.028841 and 88, then rank the revised concrete option "
        "against the project baseline.",
        template="compound_scenario_variant",
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools[0] == "material_substitution"
    # Reference the project baseline without creating another scenario.
    assert tools[1] == "compare_named_values"


def test_transport_against_default_appends_reference_validation():
    program = _compile(
        "A 96.5 t delivery over routes of 34, 67.5 and 112 km at 0.4745 "
        "kgCO2e per t-km; rank them and validate against the concrete default.",
        template="compound_scenario_unknown",
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools[-1] == "compare_to_reference"


def _context_full() -> dict:
    context = _context()
    context["project_overview"]["scope_stage_totals_kgco2e"] = {
        "foundation_pit": {"A1-A3_ProductStage": 1.0, "A4_TransportationStage": 2.0, "A5_ConstructionStage": 3.0}
    }
    return context


def test_compound_over_governed_retrieval_degrades_to_ir_assembly():
    program = compile_mas_calculation_program(
        "Show the stored foundation-pit low/base/high sensitivity results and "
        "reconcile the base result with the foundation stage-total sum.",
        QueryMode.INFORMATION_RETRIEVAL,
        {"query": "Show the stored foundation-pit low/base/high sensitivity results and "
                  "reconcile the base result with the foundation stage-total sum.",
         "mode": "IR", "task_family": "retrieval",
         "compound": True, "compound_template": "compound_scenario_unknown",
         "target_scopes": ["foundation_pit"], "target_stages": []},
        _context_full(),
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools == ["compare_named_values"] or tools == ["aggregate_named_values"]


def test_compound_with_no_values_and_no_intent_still_fails_closed():
    with pytest.raises(CalculationPlanningError) as excinfo:
        compile_mas_calculation_program(
            "Please summarise the ontology classes.",
            QueryMode.INFORMATION_RETRIEVAL,
            {"query": "Please summarise the ontology classes.",
             "mode": "IR", "task_family": "retrieval",
             "compound": True, "compound_template": "compound_scenario_unknown"},
            _context(),
        )
    assert excinfo.value.code == "UNSUPPORTED_COMPOUND_PROGRAM"


def test_correction_route_with_transport_intents_assembles_compound():
    program = compile_mas_calculation_program(
        "A logistics workshop has one 96.5 t concrete delivery and three A4 "
        "routes of 34, 67.5 and 112 km at 0.4745 kgCO2e per t-km; calculate "
        "the A4 result for each route and rank the three from lowest.",
        QueryMode.KNOWLEDGE_CORRECTION,
        {"query": "A logistics workshop has one 96.5 t concrete delivery and three A4 "
                  "routes of 34, 67.5 and 112 km at 0.4745 kgCO2e per t-km; calculate "
                  "the A4 result for each route and rank the three from lowest.",
         "mode": "KC", "task_family": "correction_preview",
         "correction_impact_preview": True, "compound": False},
        _context(),
        overlay={"changes": [], "ready": True},
    )
    tools = [step["tool_name"] for step in program["steps"]]
    assert tools.count("transport_emission") == 3
    assert "rank_named_values" in tools
