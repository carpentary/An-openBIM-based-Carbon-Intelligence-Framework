"""Check published numerical references; this does not run an LLM benchmark."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from msca_mas.calculations import (
    calculate_shares, factor_reduction, one_at_a_time_sensitivity,
    rank_named_values, transport_emission,
)
from msca_mas.evaluation import GroundTruthSuite, _assertion_matches
from msca_mas.util import stable_digest

HERE = Path(__file__).resolve().parent


def main() -> None:
    raw = json.loads((HERE / "benchmark/queries_and_ground_truth.json").read_text("utf8"))
    if raw["dataset_digest"] != stable_digest({**raw, "dataset_digest": ""}):
        raise ValueError("The representative suite digest does not match")
    suite = GroundTruthSuite.model_validate(raw)
    inputs = json.loads((HERE / "benchmark/reference_inputs.json").read_text("utf8"))
    if len(suite.cases) != 15:
        raise ValueError("Expected 15 representative cases")
    for mode in ("IR", "SA", "KC"):
        cases = [c for c in suite.cases if c.expected_mode == mode]
        assert len(cases) == 5
        assert sum("trap" in c.tags for c in cases) == 2

    calculations = {
        "IR-001": rank_named_values(inputs["scope_totals"]),
        "IR-002": rank_named_values(inputs["stage_totals"]),
        "IR-003": calculate_shares(inputs["scope_a4"]),
        "SA-001": factor_reduction(inputs["concrete_a1_a3"], 0.13),
        "SA-002": transport_emission(185000, "kg", 64, 0.162),
        "SA-003": one_at_a_time_sensitivity(
            inputs["foundation_reference"]["carbon"],
            inputs["foundation_reference"]["plan_utilisation_factor"],
            [0.74, 0.82, 0.9, 0.98], parameter_name="plan_utilisation_factor",
        ),
    }
    checked = 0
    for case in suite.cases:
        if "trap" in case.tags:
            assert case.expected_status == "blocked" and case.expected_issue_codes
            continue
        query = next(q.text for q in case.query_variants if q.language == "en")
        for assertion in case.assertions:
            path = assertion.prediction_path
            if path == "data.calculation_results":
                value = calculations[case.case_id]
            elif path == "data.retrieved_context.project_overview.combined_total_kgco2e":
                value = inputs["project_total"]
            elif case.expected_mode == "KC":
                row = next(r for r in inputs["transport_defaults"]
                           if r["name"].casefold() in query.casefold())
                proposed = float(re.search(r"hauled (\d+(?:\.\d+)?) km", query)[1])
                fields = {
                    "operation": "upsert_material_transport_parameter",
                    "target_uri": "https://example.org/project#MaterialTransportDefault_" + row["material_code"].replace("-", "_"),
                    "predicate_uri": "http://bimlca.uni.edu/ontology#hasTransportDistance",
                    "new_value": proposed,
                    "source_graph_mutated": False,
                }
                value = fields[path.split(".")[-1]]
            else:
                raise ValueError(f"Unsupported reference check: {path}")
            passed, reason = _assertion_matches(assertion, {"value": value, "unit": assertion.unit})
            if not passed:
                raise AssertionError(f"{case.case_id}: {reason}")
        checked += 1
    print(json.dumps({
        "query_records": len(suite.cases),
        "normal_case_reference_checks": checked,
        "trap_annotation_records": 6,
        "scope": "Arithmetic and record-format checks only, not MAS accuracy or human annotation agreement.",
    }, indent=2))


if __name__ == "__main__":
    main()
