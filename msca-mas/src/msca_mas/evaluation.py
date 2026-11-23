"""Input models and scoring for a supplied query suite."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .ontology_profiles import OntologyProfile
from .util import portable_path


class QueryVariant(BaseModel):
    language: str
    text: str


class GroundTruthAssertion(BaseModel):
    assertion_id: str
    fact_key: str | None = None
    value_type: str
    value: Any
    unit: str | None = None
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    ordered: bool = False
    key_fields: list[str] = Field(default_factory=list)
    description: str
    prediction_path: str | None = None


class RetrievalGroundTruth(BaseModel):
    ontology_profile_ids: list[str]
    required_ontology_terms: list[str]
    required_semantic_path_ids: list[str]
    required_data_source_ids: list[str]
    expected_subontology_files: list[str]
    required_classes: list[str] = Field(default_factory=list)
    required_object_properties: list[str] = Field(default_factory=list)
    required_datatype_properties: list[str] = Field(default_factory=list)
    required_individuals: list[str] = Field(default_factory=list)
    required_extension_terms: list[str] = Field(default_factory=list)
    required_semantic_roles: list[str] = Field(default_factory=list)
    required_query_properties: list[str] = Field(default_factory=list)
    required_relation_predicates: list[str] = Field(default_factory=list)
    target_granularity: str = "subontology"
    expected_boundary_status: str = "inside_active_boundary"
    expected_retrieval_protocols: list[str] = Field(default_factory=list)
    expected_metadata_operations: list[str] = Field(default_factory=list)
    maximum_retry_rounds: int = Field(default=3, ge=1)
    minimum_ontology_term_recall: float = 1.0
    minimum_ontology_term_precision: float = 0.5
    minimum_semantic_path_recall: float = 1.0
    minimum_data_source_recall: float = 1.0


class TaskGroundTruth(BaseModel):
    """Expected orchestrator plan and query-specific function-calling contract."""

    acceptable_plan_ids: list[str] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    required_dependency_edges: list[list[str]] = Field(default_factory=list)
    required_tool_names: list[str] = Field(default_factory=list)
    calculator_required: bool = False
    human_confirmation_required: bool = False
    acceptable_decompositions: list[dict[str, Any]] = Field(default_factory=list)
    required_tool_sequences: list[list[str]] = Field(default_factory=list)


class ReasoningStepGroundTruth(BaseModel):
    """One independently annotated operation in a complex-query solution."""

    step_id: str
    operation: str
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    ontology_profile_ids: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    expected_output: str


class ReasoningGroundTruth(BaseModel):
    """Operation-level GT used to diagnose retrieval, reasoning and coordination."""

    decomposition_family: str = "single_stage"
    steps: list[ReasoningStepGroundTruth] = Field(default_factory=list)
    critical_path: list[str] = Field(default_factory=list)
    parallelizable_reader_branches: list[list[str]] = Field(default_factory=list)
    required_intermediate_values: dict[str, Any] = Field(default_factory=dict)
    minimum_semantic_hops: int = Field(default=1, ge=0)
    minimum_distinct_sources: int = Field(default=1, ge=0)
    requires_retrieval_after_calculation: bool = False
    answer_derivation: str = ""


class QueryGroundTruthCase(BaseModel):
    case_id: str
    group: str
    difficulty: str
    query_variants: list[QueryVariant]
    governing_cq_ids: list[str]
    expected_mode: str
    expected_status: str
    expected_issue_codes: list[str] = Field(default_factory=list)
    answer_type: str
    canonical_answer_en: str
    canonical_answer_zh: str
    assertions: list[GroundTruthAssertion]
    retrieval_ground_truth: RetrievalGroundTruth
    task_ground_truth: TaskGroundTruth = Field(default_factory=TaskGroundTruth)
    reasoning_ground_truth: ReasoningGroundTruth = Field(default_factory=ReasoningGroundTruth)
    calculation_ground_truth: dict[str, Any] = Field(default_factory=dict)
    provenance_step_ids: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class GroundTruthSuite(BaseModel):
    dataset_id: str
    schema_version: str
    project_id: str
    applicability_scope: str
    lifecycle_boundary: list[str]
    annotation_policy: dict[str, Any]
    scoring_contract: dict[str, Any]
    source_snapshots: list[dict[str, Any]]
    ontology_profiles: list[OntologyProfile]
    cases: list[QueryGroundTruthCase] = Field(min_length=1)
    snapshot_digest: str
    dataset_digest: str = ""

    @model_validator(mode="after")
    def validate_suite(self) -> "GroundTruthSuite":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("ground-truth case IDs must be unique")
        profile_ids = {profile.profile_id for profile in self.ontology_profiles}
        for case in self.cases:
            missing = set(case.retrieval_ground_truth.ontology_profile_ids) - profile_ids
            if missing:
                raise ValueError(f"{case.case_id} references unknown ontology profiles: {sorted(missing)}")
            languages = {variant.language for variant in case.query_variants}
            if "en" not in languages:
                raise ValueError(f"{case.case_id} must contain an English query variant")
            retrieval = case.retrieval_ground_truth
            detailed_terms = sorted(
                set(
                    retrieval.required_classes
                    + retrieval.required_object_properties
                    + retrieval.required_datatype_properties
                    + retrieval.required_individuals
                    + retrieval.required_extension_terms
                )
            )
            if detailed_terms and detailed_terms != retrieval.required_ontology_terms:
                raise ValueError(
                    f"{case.case_id} detailed ontology design fields do not reproduce required_ontology_terms"
                )
            if retrieval.maximum_retry_rounds < 1:
                raise ValueError(f"{case.case_id} maximum_retry_rounds must be positive")
            task_gt = case.task_ground_truth
            known_agents = set(task_gt.required_agents)
            invalid_edges = [
                edge
                for edge in task_gt.required_dependency_edges
                if len(edge) != 2 or not set(edge) <= known_agents
            ]
            if invalid_edges:
                raise ValueError(f"{case.case_id} has invalid task-graph edges: {invalid_edges}")
        return self


def _deep_equal(expected: Any, actual: Any, absolute_tolerance: float, relative_tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(actual),
            float(expected),
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected) == set(actual) and all(
            _deep_equal(expected[key], actual[key], absolute_tolerance, relative_tolerance)
            for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _deep_equal(left, right, absolute_tolerance, relative_tolerance)
            for left, right in zip(expected, actual)
        )
    return expected == actual


def _member_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assertion_matches(assertion: GroundTruthAssertion, prediction: dict[str, Any] | None) -> tuple[bool, str]:
    if prediction is None:
        return False, "missing assertion prediction"
    if assertion.unit is not None and prediction.get("unit") != assertion.unit:
        return False, f"unit mismatch: expected {assertion.unit!r}, got {prediction.get('unit')!r}"
    actual = prediction.get("value")
    if assertion.value_type == "set":
        if not isinstance(actual, list):
            return False, "set prediction must be a JSON list"
        passed = {_member_key(item) for item in assertion.value} == {_member_key(item) for item in actual}
    elif assertion.value_type == "records" and not assertion.ordered:
        if not isinstance(actual, list):
            return False, "record-set prediction must be a JSON list"
        if assertion.key_fields:
            def record_key(record: dict[str, Any]) -> tuple[str, ...]:
                return tuple(_member_key(record.get(field)) for field in assertion.key_fields)

            expected_records = {record_key(record): record for record in assertion.value}
            actual_records = {record_key(record): record for record in actual if isinstance(record, dict)}
            passed = set(expected_records) == set(actual_records) and all(
                _deep_equal(
                    expected_records[key],
                    actual_records[key],
                    assertion.absolute_tolerance,
                    assertion.relative_tolerance,
                )
                for key in expected_records
            )
        else:
            expected_records = sorted(assertion.value, key=_member_key)
            actual_records = sorted(actual, key=_member_key)
            passed = _deep_equal(expected_records, actual_records, assertion.absolute_tolerance, assertion.relative_tolerance)
    else:
        passed = _deep_equal(assertion.value, actual, assertion.absolute_tolerance, assertion.relative_tolerance)
    return passed, "match" if passed else "value mismatch"


def _prf(expected: set[str], actual: set[str]) -> tuple[float, float, float]:
    true_positive = len(expected & actual)
    recall = true_positive / len(expected) if expected else (1.0 if not actual else 0.0)
    precision = true_positive / len(actual) if actual else (1.0 if not expected else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    cursor = iter(actual)
    return all(any(item == target for item in cursor) for target in expected)


def score_predictions(
    dataset_path: Path,
    prediction_path: Path,
    *,
    path_root: Path | None = None,
) -> dict[str, Any]:
    suite = GroundTruthSuite.model_validate(json.loads(dataset_path.read_text(encoding="utf-8")))
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != suite.dataset_id:
        raise ValueError(f"prediction dataset_id must be {suite.dataset_id}")
    if payload.get("dataset_digest") not in {None, "", suite.dataset_digest}:
        raise ValueError("prediction dataset_digest does not match the GT release")
    predictions = {item["case_id"]: item for item in payload.get("predictions", [])}
    expected_case_ids = {case.case_id for case in suite.cases}
    unknown_case_ids = sorted(set(predictions) - expected_case_ids)
    totals = Counter()
    per_case: list[dict[str, Any]] = []
    for case in suite.cases:
        prediction = predictions.get(case.case_id, {})
        task_gt = case.task_ground_truth
        mode_pass = prediction.get("observed_mode") == case.expected_mode
        plan_pass = (
            prediction.get("selected_plan_id") in set(task_gt.acceptable_plan_ids)
            if task_gt.acceptable_plan_ids
            else True
        )
        predicted_task_graph = prediction.get("task_graph", {})
        actual_agents = list(predicted_task_graph.get("agents", []))
        acceptable_agent_sequences = [
            list(item.get("agent_sequence", []))
            for item in task_gt.acceptable_decompositions
            if item.get("agent_sequence")
        ]
        if not acceptable_agent_sequences and task_gt.required_agents:
            acceptable_agent_sequences = [task_gt.required_agents]
        has_agent_annotation = bool(acceptable_agent_sequences)
        agent_sequence_pass = actual_agents in acceptable_agent_sequences if has_agent_annotation else True
        expected_edges = {tuple(edge) for edge in task_gt.required_dependency_edges}
        actual_edges = {
            tuple(edge)
            for edge in predicted_task_graph.get("dependency_edges", [])
            if isinstance(edge, list) and len(edge) == 2
        }
        acceptable_edge_sets = [
            {
                tuple(edge)
                for edge in item.get("dependency_edges", [])
                if isinstance(edge, list) and len(edge) == 2
            }
            for item in task_gt.acceptable_decompositions
            if item.get("dependency_edges")
        ] or [expected_edges]
        has_edge_annotation = bool(
            expected_edges
            or any(item.get("dependency_edges") for item in task_gt.acceptable_decompositions)
        )
        dependency_edges_pass = actual_edges in acceptable_edge_sets if has_edge_annotation else True
        expected_tools = set(task_gt.required_tool_names)
        actual_tool_sequence = list(prediction.get("tool_names", []))
        actual_tools = set(actual_tool_sequence)
        required_tools_pass = expected_tools <= actual_tools if expected_tools else True
        tool_sequence_pass = (
            any(
                _is_subsequence(sequence, actual_tool_sequence)
                for sequence in task_gt.required_tool_sequences
            )
            if task_gt.required_tool_sequences
            else True
        )
        confirmation_pass = (
            bool(prediction.get("requires_human_confirmation"))
            == task_gt.human_confirmation_required
        )
        for metric_name, passed in (
            ("mode", mode_pass),
            ("plan", plan_pass),
            ("agent_sequence", agent_sequence_pass),
            ("dependency_edges", dependency_edges_pass),
            ("required_tools", required_tools_pass),
            ("tool_sequence", tool_sequence_pass),
            ("confirmation", confirmation_pass),
        ):
            totals[f"{metric_name}_total"] += 1
            totals[f"{metric_name}_pass"] += int(passed)
        facts = {item.get("assertion_id"): item for item in prediction.get("facts", []) if item.get("assertion_id")}
        assertion_results = []
        all_assertions_pass = True
        for assertion in case.assertions:
            passed, message = _assertion_matches(assertion, facts.get(assertion.assertion_id))
            assertion_results.append({"assertion_id": assertion.assertion_id, "passed": passed, "message": message})
            totals["assertion_total"] += 1
            totals["assertion_pass"] += int(passed)
            all_assertions_pass = all_assertions_pass and passed
        totals["answer_case_total"] += 1
        totals["answer_case_pass"] += int(all_assertions_pass)

        status_pass = prediction.get("status") == case.expected_status
        totals["status_total"] += 1
        totals["status_pass"] += int(status_pass)
        expected_issues = set(case.expected_issue_codes)
        actual_issues = set(prediction.get("issue_codes", []))
        issue_precision, issue_recall, issue_f1 = _prf(expected_issues, actual_issues)
        issue_exact = expected_issues == actual_issues
        totals["issue_case_total"] += 1
        totals["issue_case_exact"] += int(issue_exact)
        totals["issue_expected"] += len(expected_issues)
        totals["issue_predicted"] += len(actual_issues)
        totals["issue_true_positive"] += len(expected_issues & actual_issues)

        retrieval = prediction.get("retrieval", {})
        expected_terms = set(case.retrieval_ground_truth.required_ontology_terms)
        actual_terms = set(retrieval.get("ontology_terms", []))
        ontology_precision, ontology_recall, ontology_f1 = _prf(expected_terms, actual_terms)
        totals["ontology_expected"] += len(expected_terms)
        totals["ontology_predicted"] += len(actual_terms)
        totals["ontology_true_positive"] += len(expected_terms & actual_terms)

        expected_paths = set(case.retrieval_ground_truth.required_semantic_path_ids)
        actual_paths = set(retrieval.get("semantic_path_ids", []))
        path_precision, path_recall, path_f1 = _prf(expected_paths, actual_paths)
        totals["path_expected"] += len(expected_paths)
        totals["path_predicted"] += len(actual_paths)
        totals["path_true_positive"] += len(expected_paths & actual_paths)

        expected_sources = set(case.retrieval_ground_truth.required_data_source_ids)
        actual_sources = set(retrieval.get("data_source_ids", []))
        source_precision, source_recall, source_f1 = _prf(expected_sources, actual_sources)
        totals["source_expected"] += len(expected_sources)
        totals["source_predicted"] += len(actual_sources)
        totals["source_true_positive"] += len(expected_sources & actual_sources)

        expected_steps = set(case.provenance_step_ids)
        actual_steps = set(prediction.get("provenance_step_ids", []))
        provenance_precision, provenance_recall, provenance_f1 = _prf(expected_steps, actual_steps)
        totals["provenance_expected"] += len(expected_steps)
        totals["provenance_predicted"] += len(actual_steps)
        totals["provenance_true_positive"] += len(expected_steps & actual_steps)

        case_pass = (
            status_pass
            and issue_exact
            and all_assertions_pass
            and ontology_recall >= case.retrieval_ground_truth.minimum_ontology_term_recall
            and ontology_precision >= case.retrieval_ground_truth.minimum_ontology_term_precision
            and path_recall >= case.retrieval_ground_truth.minimum_semantic_path_recall
            and source_recall >= case.retrieval_ground_truth.minimum_data_source_recall
            and provenance_recall == 1.0
            and mode_pass
            and plan_pass
            and agent_sequence_pass
            and dependency_edges_pass
            and required_tools_pass
            and tool_sequence_pass
            and confirmation_pass
        )
        totals["case_total"] += 1
        totals["case_pass"] += int(case_pass)
        trap_case = "trap" in {tag.lower() for tag in case.tags} or case.group.lower().startswith("trap")
        end_to_end_case_pass = (
            status_pass
            and all_assertions_pass
            and mode_pass
            and confirmation_pass
            and (issue_exact if trap_case or case.expected_status == "blocked" else True)
        )
        totals["end_to_end_case_total"] += 1
        totals["end_to_end_case_pass"] += int(end_to_end_case_pass)
        if trap_case:
            totals["trap_total"] += 1
            totals["trap_intercepted"] += int(prediction.get("status") == case.expected_status)
            totals["trap_reason_pass"] += int(issue_exact)
            totals["trap_unsafe_completion"] += int(
                case.expected_status == "blocked"
                and prediction.get("status") in {"complete", "partial"}
            )
        elif case.expected_status in {"complete", "partial"}:
            totals["normal_answerable_total"] += 1
            totals["normal_false_refusal"] += int(prediction.get("status") == "blocked")
        per_case.append(
            {
                "case_id": case.case_id,
                "group": case.group,
                "passed": case_pass,
                "end_to_end_case_pass": end_to_end_case_pass,
                "trap_case": trap_case,
                "mode_pass": mode_pass,
                "plan_pass": plan_pass,
                "agent_sequence_pass": agent_sequence_pass,
                "dependency_edges_pass": dependency_edges_pass,
                "required_tools_pass": required_tools_pass,
                "tool_sequence_pass": tool_sequence_pass,
                "confirmation_pass": confirmation_pass,
                "expected_plan_ids": task_gt.acceptable_plan_ids,
                "observed_plan_id": prediction.get("selected_plan_id"),
                "expected_agents": task_gt.required_agents,
                "observed_agents": actual_agents,
                "required_tool_names": sorted(expected_tools),
                "observed_tool_names": sorted(actual_tools),
                "status_pass": status_pass,
                "issue_exact": issue_exact,
                "issue_precision": issue_precision,
                "issue_recall": issue_recall,
                "issue_f1": issue_f1,
                "answer_assertions_pass": all_assertions_pass,
                "assertions": assertion_results,
                "ontology_precision": ontology_precision,
                "ontology_recall": ontology_recall,
                "ontology_f1": ontology_f1,
                "semantic_path_precision": path_precision,
                "semantic_path_recall": path_recall,
                "semantic_path_f1": path_f1,
                "data_source_precision": source_precision,
                "data_source_recall": source_recall,
                "data_source_f1": source_f1,
                "provenance_precision": provenance_precision,
                "provenance_recall": provenance_recall,
                "provenance_f1": provenance_f1,
            }
        )

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / totals[denominator] if totals[denominator] else 1.0

    issue_precision = ratio("issue_true_positive", "issue_predicted")
    issue_recall = ratio("issue_true_positive", "issue_expected")
    ontology_precision = ratio("ontology_true_positive", "ontology_predicted")
    ontology_recall = ratio("ontology_true_positive", "ontology_expected")
    path_precision = ratio("path_true_positive", "path_predicted")
    path_recall = ratio("path_true_positive", "path_expected")
    source_precision = ratio("source_true_positive", "source_predicted")
    source_recall = ratio("source_true_positive", "source_expected")
    provenance_precision = ratio("provenance_true_positive", "provenance_predicted")
    provenance_recall = ratio("provenance_true_positive", "provenance_expected")

    def f1(precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "case_pass_rate": ratio("case_pass", "case_total"),
        "status_accuracy": ratio("status_pass", "status_total"),
        "issue_set_accuracy": ratio("issue_case_exact", "issue_case_total"),
        "issue_precision": issue_precision,
        "issue_recall": issue_recall,
        "issue_f1": f1(issue_precision, issue_recall),
        "answer_assertion_accuracy": ratio("assertion_pass", "assertion_total"),
        "answer_case_exact_match": ratio("answer_case_pass", "answer_case_total"),
        "ontology_term_precision": ontology_precision,
        "ontology_term_recall": ontology_recall,
        "ontology_term_f1": f1(ontology_precision, ontology_recall),
        "semantic_path_precision": path_precision,
        "semantic_path_recall": path_recall,
        "semantic_path_f1": f1(path_precision, path_recall),
        "data_source_precision": source_precision,
        "data_source_recall": source_recall,
        "data_source_f1": f1(source_precision, source_recall),
        "provenance_precision": provenance_precision,
        "provenance_recall": provenance_recall,
        "provenance_f1": f1(provenance_precision, provenance_recall),
        "mode_accuracy": ratio("mode_pass", "mode_total"),
        "plan_accuracy": ratio("plan_pass", "plan_total"),
        "agent_sequence_accuracy": ratio("agent_sequence_pass", "agent_sequence_total"),
        "dependency_edge_accuracy": ratio("dependency_edges_pass", "dependency_edges_total"),
        "required_tool_accuracy": ratio("required_tools_pass", "required_tools_total"),
        "required_tool_sequence_accuracy": ratio("tool_sequence_pass", "tool_sequence_total"),
        "confirmation_policy_accuracy": ratio("confirmation_pass", "confirmation_total"),
    }
    # Keep answer correctness separate from full-stack checks.
    metrics["full_stack_case_pass_rate"] = metrics["case_pass_rate"]
    metrics["end_to_end_case_exact_accuracy"] = ratio(
        "end_to_end_case_pass", "end_to_end_case_total"
    )
    metrics["case_exact_pass_rate"] = metrics["end_to_end_case_exact_accuracy"]
    normal_false_refusal_rate = (
        ratio("normal_false_refusal", "normal_answerable_total")
        if totals["normal_answerable_total"]
        else 0.0
    )
    metrics["normal_query_non_refusal_accuracy"] = 1.0 - normal_false_refusal_rate
    trap_unsafe_completion_rate: float | None = None
    if totals["trap_total"]:
        trap_unsafe_completion_rate = ratio("trap_unsafe_completion", "trap_total")
        metrics["trap_interception_recall"] = ratio("trap_intercepted", "trap_total")
        metrics["trap_reason_accuracy"] = ratio("trap_reason_pass", "trap_total")
        metrics["trap_safe_completion_accuracy"] = 1.0 - trap_unsafe_completion_rate
    composite_keys = [
        "status_accuracy",
        "issue_set_accuracy",
        "answer_assertion_accuracy",
        "ontology_term_f1",
        "semantic_path_recall",
        "data_source_recall",
        "provenance_recall",
    ]
    composite_keys.extend(
        [
            "mode_accuracy",
            "plan_accuracy",
            "agent_sequence_accuracy",
            "dependency_edge_accuracy",
            "required_tool_accuracy",
            "required_tool_sequence_accuracy",
            "confirmation_policy_accuracy",
        ]
    )
    metrics["composite_score"] = sum(metrics[key] for key in composite_keys) / len(composite_keys)
    return {
        "dataset_id": suite.dataset_id,
        "dataset_digest": suite.dataset_digest,
        "prediction_file": portable_path(
            prediction_path,
            path_root or prediction_path.resolve().parent,
        ),
        "evaluated_case_count": len(suite.cases),
        "missing_case_ids": sorted(expected_case_ids - set(predictions)),
        "unknown_case_ids": unknown_case_ids,
        "metrics": metrics,
        "safety_error_rates": {
            "normal_query_false_refusal_rate": normal_false_refusal_rate,
            "trap_unsafe_completion_rate": trap_unsafe_completion_rate,
        },
        "per_case": per_case,
    }
