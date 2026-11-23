from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .analysis import CalculationPlanningError
from .config import ProjectConfig
from .cq import CQCatalogue
from .entity_resolution import matching_transport_defaults, transport_material_labels
from .knowledge import KnowledgeIndex
from .intent import analyse_query_intent
from .metadata_pipeline import MetadataPipeline, build_metadata_pipeline
from .mas_program import resolve_program_bindings
from .models import (
    AgentName,
    AgentUpdate,
    FinalReport,
    PersistenceMode,
    ProvenanceEvent,
    QueryMode,
    ReadinessIssue,
    RunStatus,
    Severity,
    SharedGlobalState,
    TaskDependencyGraph,
    TaskNode,
    ToolCallStatus,
)
from .ontology_profiles import ontology_profiles
from .reasoning import PlanningReasoner
from .semantic_models import (
    BackboneVerification,
    CalculationProgram,
    CalculationProgramStep,
    CalculationStateTransition,
    CompositeDataContext,
    DeltaGraphSpecification,
    DeltaOverride,
    EpistemicValidation,
    MetadataRetrievalContext,
    MetadataRetrievalRound,
    OntologyEntityCandidate,
    OntologyRelation,
    QueryInterpretation,
    QueryContract,
    ReaderRetrievalPlan,
    ReferenceDemand,
    ResolvedSemanticPath,
    RetrievalStep,
    SharedGlobalSpace,
    TaskControlBlock,
)
from .tooling import (
    ONTOLOGY_REASONING_TOOL_NAMES,
    ToolRegistry,
    append_tool_call,
    build_tool_registry,
)
from .safety import assess_query_contract_safety
from .transactions import TransactionService
from .util import stable_digest


class KnowledgeReader(Protocol):
    def is_current(self) -> bool: ...

    def project_overview(self) -> dict[str, Any]: ...

    def top_elements(self, limit: int = 10, stage: str | None = None) -> list[dict[str, Any]]: ...

    def resource_breakdown(self, term: str, stage: str | None = None, limit: int = 30) -> dict[str, Any]: ...

    def work_package_breakdown(self, term: str, limit: int = 30) -> list[dict[str, Any]]: ...

    def element_trace(self, element_uri: str, limit: int = 250) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AgentServices:
    config: ProjectConfig
    cq_catalogue: CQCatalogue
    knowledge: KnowledgeReader
    transactions: TransactionService
    metadata_pipeline: MetadataPipeline | None = None
    planning_reasoner: PlanningReasoner | None = None
    tools: ToolRegistry | None = None
    ontology_mode: Literal["full", "schema_only"] = "full"

    def tool_registry(self) -> ToolRegistry:
        if self.tools is not None:
            return self.tools
        registry = build_tool_registry(
            self.config,
            self.cq_catalogue,
            self.knowledge,
            self.transactions,
            self.metadata_pipeline,
        )
        if self.ontology_mode == "schema_only":
            from .ontology_policy import restrict_ontology_capabilities

            registry = restrict_ontology_capabilities(
                registry.restricted(
                    excluded_names=ONTOLOGY_REASONING_TOOL_NAMES
                ),
                ontology_mode=self.ontology_mode,
            )
        return registry

    @classmethod
    def create(
        cls,
        config: ProjectConfig,
        knowledge: KnowledgeReader | None = None,
        planning_reasoner: PlanningReasoner | None = None,
        ontology_mode: Literal["full", "schema_only"] = "full",
    ) -> "AgentServices":
        cq_catalogue = CQCatalogue.load(config.cq_catalogue)
        governed_knowledge = knowledge or KnowledgeIndex(config)
        transactions = TransactionService(config.transaction_dir, config=config)
        metadata_pipeline = build_metadata_pipeline(config, cq_catalogue)
        registry = build_tool_registry(
            config,
            cq_catalogue,
            governed_knowledge,
            transactions,
            metadata_pipeline,
        )
        if ontology_mode == "schema_only":
            from .ontology_policy import restrict_ontology_capabilities

            registry = restrict_ontology_capabilities(
                registry.restricted(
                    excluded_names=ONTOLOGY_REASONING_TOOL_NAMES
                ),
                ontology_mode=ontology_mode,
            )
        return cls(
            config=config,
            cq_catalogue=cq_catalogue,
            knowledge=governed_knowledge,
            transactions=transactions,
            metadata_pipeline=metadata_pipeline,
            planning_reasoner=planning_reasoner,
            ontology_mode=ontology_mode,
            tools=registry,
        )


def _provenance(
    state: SharedGlobalState,
    agent: AgentName,
    action: str,
    sources: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    existing = list(state.get("provenance", []))
    event = ProvenanceEvent(
        sequence=len(existing) + 1,
        agent=agent,
        action=action,
        sources=sources or [],
        details=details or {},
    )
    return existing + [event.model_dump(mode="json")]


def _metrics(state: SharedGlobalState, agent: AgentName, started: float, **extra: Any) -> dict[str, Any]:
    metrics = dict(state.get("metrics", {}))
    entry = {"duration_seconds": round(time.perf_counter() - started, 6), **extra}
    previous = metrics.get(agent.value)
    if isinstance(previous, dict):
        # Keep timings for each visit to a repeated node.
        rounds = [round for round in previous.get("_rounds", [previous]) if isinstance(round, dict)]
        entry = {**entry, "_rounds": [*rounds, entry]}
    metrics[agent.value] = entry
    return metrics


def _deduplicate_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one governed issue per semantic code and merge its evidence.

    The same safety condition can be detected independently by Reader and by
    the final contract gate.  Repeating it makes the report look like two
    failures even though it is one condition.  Tool failures and crashes remain
    distinct because their messages identify different failed calls.
    """

    output: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    severity_rank = {
        Severity.INFO.value: 0,
        Severity.WARNING.value: 1,
        Severity.BLOCKING.value: 2,
    }
    distinct_message_codes = {"TOOL_CALL_FAILED", "TOOL_CALL_CRASH"}
    for issue in issues:
        code = str(issue.get("code"))
        message = str(issue.get("message"))
        key = (code, message if code in distinct_message_codes else "")
        if key not in positions:
            normalised = dict(issue)
            normalised["missing_fields"] = list(
                dict.fromkeys(str(value) for value in issue.get("missing_fields", []))
            )
            positions[key] = len(output)
            output.append(normalised)
            continue

        position = positions[key]
        existing = dict(output[position])
        existing["missing_fields"] = list(
            dict.fromkeys(
                [
                    *(str(value) for value in existing.get("missing_fields", [])),
                    *(str(value) for value in issue.get("missing_fields", [])),
                ]
            )
        )
        if severity_rank.get(str(issue.get("severity")), 0) > severity_rank.get(
            str(existing.get("severity")), 0
        ):
            existing["severity"] = issue.get("severity")
        if issue.get("source") in {
            "independent_safety_gate",
            "independent_semantic_risk_gate",
        }:
            existing["message"] = issue.get("message", existing.get("message"))
            existing["source"] = issue.get("source")
        output[position] = existing
    return output


def _prioritise_root_cause_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suppress derivative user-facing symptoms when a grounded root cause exists.

    Downstream tool failures remain available in the immutable tool-call trace and
    execution metrics.  Once an independent semantic or safety gate has already
    established why execution must stop, however, exposing a dependent Reader or
    Calculator failure as a second user-facing cause obscures the actionable root
    condition.  This is DAG failure propagation, not query-specific issue matching.
    """

    semantic_root_codes = {
        str(issue.get("code"))
        for issue in issues
        if issue.get("source") == "independent_semantic_risk_gate"
    }
    independent_root_codes = {
        str(issue.get("code"))
        for issue in issues
        if issue.get("source") == "independent_safety_gate"
    }
    root_codes = semantic_root_codes | independent_root_codes
    suppress: set[str] = set()
    if "CALCULATION_PRECONDITION_FAILED" in root_codes:
        suppress.update(
            {
                "REFERENCE_DEMAND_UNRESOLVED",
                "MISSING_A4_PARAMETERS",
                "MISSING_SUBSTITUTION_PARAMETERS",
                "MISSING_SENSITIVITY_PARAMETERS",
                "MISSING_REUSE_PARAMETERS",
                "CALCULATION_PROGRAM_COMPILATION_FAILED",
                "UNMODELLED_CHANGE",
            }
        )
    # Independent safety findings take precedence over model risk labels.
    unsupported_from_safety_gate = "UNSUPPORTED_PERSISTENT_WRITE" in independent_root_codes
    if "REFERENCE_DEMAND_UNRESOLVED" in root_codes and not unsupported_from_safety_gate:
        suppress.update(
            {
                "MISSING_A4_PARAMETERS",
                "AMBIGUOUS_PERSISTENT_CORRECTION",
                "UNSUPPORTED_PERSISTENT_WRITE",
            }
        )
    if "AMBIGUOUS_PERSISTENT_CORRECTION" in root_codes and not unsupported_from_safety_gate:
        suppress.update(
            {
                "MISSING_A4_PARAMETERS",
                "UNSUPPORTED_PERSISTENT_WRITE",
            }
        )
    if "OUTSIDE_A1_A5_SCOPE" in root_codes:
        suppress.update(
            {
                "CALCULATION_PRECONDITION_FAILED",
                "PARAMETER_ROLE_UNIT_MISMATCH",
                "PARAMETER_VALUE_UNGROUNDED",
            }
        )
    if "CALCULATION_PRECONDITION_FAILED" in root_codes:
        suppress.update(
            {
                "PARAMETER_ROLE_UNIT_MISMATCH",
                "PARAMETER_VALUE_UNGROUNDED",
            }
        )
    if "MISSING_A4_PARAMETERS" in root_codes:
        suppress.add("AMBIGUOUS_PERSISTENT_CORRECTION")
    elif "AMBIGUOUS_PERSISTENT_CORRECTION" in root_codes:
        suppress.add("MISSING_A4_PARAMETERS")
    if "UNSUPPORTED_PERSISTENT_WRITE" in root_codes:
        suppress.update(
            {
                "CALCULATION_PRECONDITION_FAILED",
                "CALCULATION_PROGRAM_COMPILATION_FAILED",
                "MISSING_A4_PARAMETERS",
            }
        )
        # A model label cannot suppress an independent safety finding.
        if unsupported_from_safety_gate:
            suppress.update(
                {
                    "REFERENCE_DEMAND_UNRESOLVED",
                    "AMBIGUOUS_PERSISTENT_CORRECTION",
                }
            )
    derivative_failure_codes = {
        "TOOL_CALL_FAILED",
        "CALCULATION_TOOL_FAILED",
    }
    derivative_execution_sources = {
        AgentName.METADATA.value,
        AgentName.EXTRACTOR.value,
        AgentName.READER.value,
        AgentName.CALCULATOR.value,
    }
    return [
        issue
        for issue in issues
        if str(issue.get("code")) not in suppress
        and not (
            root_codes
            and str(issue.get("code")) in derivative_failure_codes
            and str(issue.get("source")) in derivative_execution_sources
        )
    ]


def _tool_failure_issue(agent: AgentName, tool_name: str, error: str | None, *, error_kind: str | None = None) -> dict[str, Any]:
    # Keep internal tool crashes separate from domain failures.
    code = "TOOL_CALL_CRASH" if error_kind == "crash" else "TOOL_CALL_FAILED"
    return ReadinessIssue(
        code=code,
        severity=Severity.BLOCKING,
        message=f"{tool_name} failed: {error or 'unknown tool error'}",
        source=agent.value,
    ).model_dump(mode="json")


def _shared_space(state: SharedGlobalState) -> SharedGlobalSpace:
    """Validate the current five-category shared blackboard or create it."""

    return SharedGlobalSpace.model_validate(state.get("shared_global_space") or {})


def _update_shared_space(
    state: SharedGlobalState,
    *,
    task_control: dict[str, Any] | None = None,
    entity_registry: dict[str, Any] | None = None,
    baseline_context: dict[str, Any] | None = None,
    delta_specifications: dict[str, Any] | None = None,
    analytics_and_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a typed partial update to the paper's Shared Global Space."""

    current = _shared_space(state)
    updates: dict[str, Any] = {}
    if task_control is not None:
        updates["task_control_block"] = task_control
    if entity_registry is not None:
        merged = dict(current.entity_pointer_registry)
        merged.update(entity_registry)
        updates["entity_pointer_registry"] = merged
    if baseline_context is not None:
        merged = dict(current.baseline_factual_context)
        merged.update(baseline_context)
        updates["baseline_factual_context"] = merged
    if delta_specifications is not None:
        merged = dict(current.transient_delta_specifications)
        merged.update(delta_specifications)
        updates["transient_delta_specifications"] = merged
    if analytics_and_provenance is not None:
        merged = dict(current.analytical_metrics_and_provenance)
        merged.update(analytics_and_provenance)
        updates["analytical_metrics_and_provenance"] = merged
    return current.model_copy(update=updates).model_dump(mode="json")


def _deduplicate_semantic_candidates(
    values: list[OntologyEntityCandidate],
) -> list[OntologyEntityCandidate]:
    selected: dict[str, OntologyEntityCandidate] = {}
    for value in values:
        current = selected.get(value.uri)
        if current is None or value.final_score > current.final_score or (value.seed and not current.seed):
            selected[value.uri] = value
    return sorted(
        selected.values(),
        key=lambda value: (not value.seed, value.discovered_at_hop, -value.final_score, value.uri),
    )


def _deduplicate_semantic_relations(values: list[OntologyRelation]) -> list[OntologyRelation]:
    selected: dict[str, OntologyRelation] = {}
    for value in values:
        key = f"{value.relation_id}|{value.direction}"
        current = selected.get(key)
        if current is None or value.hop < current.hop:
            selected[key] = value
    return sorted(
        selected.values(),
        key=lambda value: (value.hop, value.subject_uri, value.predicate_uri, value.object_uri),
    )


def _minimal_profile_projection(
    planner_profile_ids: list[str],
    signature_profile_ids: list[str],
) -> list[str]:
    """Project planner candidates to the smallest governed profile closure."""

    planner = list(dict.fromkeys(planner_profile_ids))
    if not planner:
        return list(dict.fromkeys(signature_profile_ids))
    available = set(planner)

    def selected(*profile_ids: str) -> list[str]:
        return [profile_id for profile_id in profile_ids if profile_id in available]

    if "ONT-SYSTEM-CARBON" in available:
        return selected("ONT-PROJECT-CARBON", "ONT-SYSTEM-CARBON")
    if "ONT-ENCLOSURE-CPD" in available:
        return selected("ONT-ENCLOSURE-CPD", "ONT-SEPRC-PROVENANCE")
    if "ONT-FOUNDATION-EARTHWORK" in available:
        return selected("ONT-FOUNDATION-EARTHWORK", "ONT-SEPRC-PROVENANCE")
    if "ONT-SCENARIO" in available:
        if "ONT-A4-TRANSPORT" in available:
            return selected("ONT-SCENARIO", "ONT-A4-TRANSPORT")
        if "ONT-PROJECT-CARBON" in available:
            return selected("ONT-SCENARIO", "ONT-PROJECT-CARBON")
        return selected("ONT-SCENARIO", "ONT-SCOPE-QUALITY") or ["ONT-SCENARIO"]
    if "ONT-A4-TRANSPORT" in available:
        return selected("ONT-A4-TRANSPORT", "ONT-SEPRC-PROVENANCE")
    return planner


def _coverage_preserving_profile_projection(
    planner_profile_ids: list[str],
    signature_profile_ids: list[str],
) -> list[str]:
    """Keep profiles that add a term, path, or source to the query contract."""

    catalogue = ontology_profiles()
    candidates = list(
        dict.fromkeys([*planner_profile_ids, *signature_profile_ids])
    )
    unknown = sorted(set(candidates) - set(catalogue))
    if unknown:
        raise ValueError(f"unknown ontology profiles in semantic closure: {unknown}")

    selected: list[str] = []
    covered_features: set[str] = set()
    for profile_id in candidates:
        profile = catalogue[profile_id]
        features = {
            *(f"term:{term}" for term in profile.required_terms),
            *(
                f"path:{path['path_id']}"
                for path in profile.semantic_paths
                if path.get("path_id")
            ),
            *(f"source:{source}" for source in profile.source_modules),
        }
        if features and features.issubset(covered_features):
            continue
        selected.append(profile_id)
        covered_features.update(features)
    return selected


class OrchestratorAgent:
    name = AgentName.ORCHESTRATOR

    def __init__(
        self,
        planning_reasoner: PlanningReasoner | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.planning_reasoner = planning_reasoner
        self.tools = tools

    @staticmethod
    def classify(query: str) -> QueryMode:
        """Return the rule-based intent used by the planner."""

        return analyse_query_intent(query).rule_mode

    def _planning_failure_state(
        self,
        state: SharedGlobalState,
        query: str,
        mode: QueryMode,
        started: float,
        tool_calls: list[dict[str, Any]],
        issue: dict[str, Any],
        failure_stage: str,
    ) -> AgentUpdate:
        """Fail closed when planning cannot produce a governed contract or plan.

        A planning defect must reach the user as a BLOCKED report through the
        Synthesis gate instead of crashing the whole run.
        """

        contract = QueryContract.model_validate(
            {"query": query or "unavailable", "mode": mode.value, "task_family": "retrieval"}
        )
        return {
            "mode": mode.value,
            "query_contract": contract.model_dump(mode="json"),
            "tdg": {},
            "pending_agents": [AgentName.SYNTHESIS.value],
            "shared_global_space": {},
            "analysis_required": False,
            "planning_candidates": [],
            "selected_plan_id": "",
            "cq_matches": [],
            "entity_pointers": [],
            "ontology_context": {},
            "retrieved_context": {},
            "scenario_overlay": {},
            "calculation_results": {},
            "issues": [issue],
            "tool_calls": tool_calls,
            "provenance": _provenance(
                state,
                self.name,
                "failed closed during planning; routed to the synthesis gate",
                details={"failure_stage": failure_stage},
            ),
            "metrics": _metrics(state, self.name, started, tools_called=0),
        }

    def run(self, state: SharedGlobalState) -> AgentUpdate:
        started = time.perf_counter()
        query = state.get("query", "").strip()
        if not query:
            raise ValueError("query must not be empty")
        planning_details: dict[str, Any] = {}
        deterministic_mode = self.classify(query)
        rule_signals = analyse_query_intent(query)
        if self.planning_reasoner is None:
            mode = deterministic_mode
            route_source = "deterministic_governed_classifier"
            structured_output_attempts = 0
            transport_request_attempts = 0
            transport_retry_count = 0
            model_proposed_mode = None
            planning_arbitration = "deterministic_only"
            rule_signal_mode = deterministic_mode
            arbitration_confidence = rule_signals.confidence
            arbitration_evidence = list(rule_signals.evidence)
            api_usage: dict[str, int] = {}
            cumulative_api_usage: dict[str, int] = {}
            token_budget = None
            rationale_summary = (
                f"Governed intent classifier selected {mode.value}; the corresponding validated workflow template was instantiated."
            )
            advisory_task_family = None
            advisory_operation_hints: list[str] = []
            advisory_semantic_anchors: list[str] = []
            semantic_contract_proposal: dict[str, Any] = {}
        else:
            planning_failure: str | None = None
            try:
                decision = self.planning_reasoner.decide(query, deterministic_mode)
            except Exception as exc:  # a failed planning call degrades to the
                # Record planner failure and fall back to the rule-based route.
                planning_failure = f"{type(exc).__name__}: {exc}"
                decision = None
            if decision is None:
                mode = deterministic_mode
                route_source = "planning_failed_deterministic_fallback"
                rationale_summary = (
                    f"Planning failed ({planning_failure}); the governed rule classifier "
                    f"selected {mode.value}."
                )[:500]
                structured_output_attempts = 0
                transport_request_attempts = 0
                transport_retry_count = 0
                model_proposed_mode = None
                planning_arbitration = "deterministic_only_after_planning_failure"
                rule_signal_mode = deterministic_mode
                arbitration_confidence = rule_signals.confidence
                arbitration_evidence = list(rule_signals.evidence)
                api_usage = {}
                cumulative_api_usage = {}
                token_budget = None
                advisory_task_family = None
                advisory_operation_hints = []
                advisory_semantic_anchors = []
                semantic_contract_proposal = {}
                planning_details = {"planning_failure": planning_failure[:400]}
            else:  # noqa: PLR5501 - explicit success branch keeps the trace readable
                mode = decision.mode
                route_source = decision.route_source
                rationale_summary = decision.rationale_summary
                structured_output_attempts = decision.structured_output_attempts
                transport_request_attempts = decision.transport_request_attempts
                transport_retry_count = decision.transport_retry_count
                model_proposed_mode = decision.model_proposed_mode
                planning_arbitration = decision.planning_arbitration
                rule_signal_mode = decision.rule_signal_mode or deterministic_mode
                arbitration_confidence = decision.arbitration_confidence
                arbitration_evidence = list(decision.arbitration_evidence)
                api_usage = decision.api_usage
                cumulative_api_usage = decision.cumulative_api_usage
                token_budget = decision.token_budget
                advisory_task_family = decision.task_family
                advisory_operation_hints = list(decision.operation_hints)
                advisory_semantic_anchors = list(decision.semantic_anchors)
                semantic_contract_proposal = decision.model_dump(mode="json")
                planning_details = {}
        if self.tools is None:
            return self._planning_failure_state(
                state,
                query,
                mode,
                started,
                list(state.get("tool_calls", [])),
                ReadinessIssue(
                    code="ORCHESTRATOR_NOT_CONFIGURED",
                    severity=Severity.BLOCKING,
                    message="orchestrator tool registry is not configured",
                    source=self.name.value,
                ).model_dump(mode="json"),
                "tool_registry",
            )
        query_contract, tool_calls, contract_call = append_tool_call(
            state.get("tool_calls"),
            self.tools,
            self.name,
            "parse_query_contract",
            {
                "query": query,
                "mode": mode.value,
                "advisory_task_family": advisory_task_family,
                "advisory_operation_hints": advisory_operation_hints,
                "advisory_semantic_anchors": advisory_semantic_anchors,
                "advisory_contract": semantic_contract_proposal,
            },
        )
        if contract_call.status == ToolCallStatus.ERROR:
            return self._planning_failure_state(
                state,
                query,
                mode,
                started,
                tool_calls,
                _tool_failure_issue(
                    self.name,
                    contract_call.tool_name,
                    contract_call.error,
                    error_kind=contract_call.error_kind,
                ),
                "parse_query_contract",
            )
        analysis_required = bool(query_contract.get("requires_calculator"))
        plan_result, tool_calls, plan_call = append_tool_call(
            tool_calls,
            self.tools,
            self.name,
            "build_execution_plan",
            {
                "mode": mode.value,
                "analysis_required": analysis_required,
                "query": query,
                "query_contract": query_contract,
            },
        )
        if plan_call.status == ToolCallStatus.ERROR:
            return self._planning_failure_state(
                state,
                query,
                mode,
                started,
                tool_calls,
                _tool_failure_issue(
                    self.name,
                    plan_call.tool_name,
                    plan_call.error,
                    error_kind=plan_call.error_kind,
                ),
                "build_execution_plan",
            )
        agents = [AgentName(value) for value in plan_result["agents"]]
        selected_plan_id = str(plan_result["selected_plan_id"])
        planning_candidates = list(plan_result["candidates"])
        selected_candidate = next(
            (item for item in planning_candidates if item["selected"]), None
        )
        if selected_candidate is None:
            return self._planning_failure_state(
                state,
                query,
                mode,
                started,
                tool_calls,
                ReadinessIssue(
                    code="PLAN_SELECTION_FAILED",
                    severity=Severity.BLOCKING,
                    message="build_execution_plan returned no selected candidate",
                    source=self.name.value,
                ).model_dump(mode="json"),
                "plan_selection",
            )
        tasks: list[TaskNode] = []
        task_contracts: dict[AgentName, tuple[str, list[str], list[str], list[str]]] = {
            AgentName.METADATA: (
                "resolve_query_specific_subontology",
                ["query_interpretation", "ontology_entities", "ontology_relations", "semantic_paths", "sufficiency_decision"],
                ["S", "E", "P", "R", "C", "assessment boundary"],
                ["hybrid retrieval then rerank", "recall adjacent relations", "bounded local retry"],
            ),
            AgentName.EXTRACTOR: (
                "construct_typed_delta_graph",
                ["delta_overrides", "reference_demands", "feasibility_status"],
                ["scenario override", "explicit parameter", "reference demand"],
                ["explicit values override references", "no silent assumptions", "persistent changes require confirmation"],
            ),
            AgentName.READER: (
                "retrieve_composite_data_context",
                ["baseline_facts", "reference_parameters", "source_provenance"],
                ["CKG SPARQL template", "CPD", "EFD", "transport default"],
                ["preverified CKG templates", "hard region/time/category filters", "no source mutation"],
            ),
            AgentName.CALCULATOR: (
                "execute_deterministic_calculation_program",
                ["calculation_steps", "state_transitions", "final_result", "units"],
                ["baseline plus delta", "unit conversion", "atomic function"],
                ["schema-validated arguments", "prevalidated functions only", "fail closed on missing inputs"],
            ),
            AgentName.SYNTHESIS: (
                "validate_epistemic_completeness_and_report",
                ["executive_summary", "execution_log", "provenance", "recommendations"],
                ["query-answer alignment", "epistemic completeness", "clarification"],
                ["do not recompute tool results", "block unsupported claims", "preview persistent writes"],
            ),
        }
        node_ids = {
            agent: f"task_{index:02d}_{agent.value}"
            for index, agent in enumerate(agents, start=1)
        }

        def task_dependencies(agent: AgentName) -> list[str]:
            metadata = node_ids.get(AgentName.METADATA)
            extractor = node_ids.get(AgentName.EXTRACTOR)
            reader = node_ids.get(AgentName.READER)
            calculator = node_ids.get(AgentName.CALCULATOR)
            if agent == AgentName.METADATA:
                return []
            if agent == AgentName.EXTRACTOR:
                return [metadata] if metadata else []
            if agent == AgentName.READER:
                # Run extraction before reference retrieval if the scenario can add a demand.
                predecessor = extractor or metadata
                return [predecessor] if predecessor else []
            if agent == AgentName.CALCULATOR:
                return [reader] if reader else ([extractor] if extractor else [])
            if agent == AgentName.SYNTHESIS:
                if calculator:
                    return [calculator]
                predecessor = reader or extractor or metadata
                return [predecessor] if predecessor else []
            return []

        for index, agent in enumerate(agents, start=1):
            node_id = node_ids[agent]
            operation, expected_outputs, schema_anchors, constraints = task_contracts[agent]
            tasks.append(
                TaskNode(
                    node_id=node_id,
                    agent=agent,
                    instruction=(
                        f"Apply plan {selected_plan_id} ({selected_candidate['strategy']}); execute {agent.value} "
                        "responsibilities for the original query without dropping its constraints. "
                        f"Governed contract operations: {', '.join(query_contract.get('operation_ids', []))}."
                    ),
                    operation=operation,
                    dependencies=task_dependencies(agent),
                    anchor_terms=list(
                        dict.fromkeys(
                            [
                                *schema_anchors,
                                *selected_candidate.get("tool_focus", {}).get(agent.value, []),
                                *query_contract.get("target_scopes", []),
                                *query_contract.get("target_stages", []),
                                *query_contract.get("resource_targets", []),
                                *query_contract.get("explicit_identifiers", []),
                                *query_contract.get("ontology_obligations", []),
                                *query_contract.get("advisory_semantic_anchors", []),
                            ]
                        )
                    ),
                    expected_outputs=expected_outputs,
                    execution_constraints=constraints,
                )
            )
        tdg = TaskDependencyGraph(
            mode=mode,
            tasks=tasks,
            rationale_summary=rationale_summary,
        )
        pending_agents = [agent.value for agent in tdg.execution_order()]
        task_control = TaskControlBlock(
            mode=mode,
            selected_plan_id=selected_plan_id,
            tdg=tdg.model_dump(mode="json"),
            pending_agents=pending_agents,
            current_agent=pending_agents[0] if pending_agents else None,
            completed_agents=[],
            plan_candidates=planning_candidates,
        )
        shared_space = SharedGlobalSpace(
            task_control_block=task_control,
            entity_pointer_registry={
                "status": "awaiting_metadata",
                "query": query,
                "compressed_schema_roles": ["S", "E", "P", "R", "C"],
            },
            baseline_factual_context={"status": "not_retrieved"},
            transient_delta_specifications={
                "status": "not_applicable"
                if mode == QueryMode.INFORMATION_RETRIEVAL
                else "awaiting_extraction"
            },
            analytical_metrics_and_provenance={
                "status": "planned",
                "selected_plan_id": selected_plan_id,
                "selection_policy": plan_result.get("selection_policy"),
                "query_contract": query_contract,
            },
        )
        return {
            "mode": mode.value,
            "query_contract": query_contract,
            "tdg": tdg.model_dump(mode="json"),
            "pending_agents": pending_agents,
            "shared_global_space": shared_space.model_dump(mode="json"),
            "analysis_required": analysis_required,
            "planning_candidates": planning_candidates,
            "selected_plan_id": selected_plan_id,
            "cq_matches": [],
            "entity_pointers": [],
            "ontology_context": {},
            "retrieved_context": {},
            "scenario_overlay": {},
            "calculation_results": {},
            "issues": [],
            "tool_calls": tool_calls,
            "provenance": _provenance(
                state,
                self.name,
                "classified intent and created an acyclic task dependency graph",
                details={
                    "mode": mode.value,
                    "task_count": len(tasks),
                    "route_source": route_source,
                    "structured_output_attempts": structured_output_attempts,
                    "transport_request_attempts": transport_request_attempts,
                    "transport_retry_count": transport_retry_count,
                    "deterministic_default": deterministic_mode.value,
                    "planning_failure": planning_details.get("planning_failure"),
                    "model_proposed_mode": (
                        model_proposed_mode.value if model_proposed_mode is not None else None
                    ),
                    "planning_arbitration": planning_arbitration,
                    "rule_signal_mode": rule_signal_mode.value,
                    "arbitration_confidence": arbitration_confidence,
                    "arbitration_evidence": arbitration_evidence,
                    "llm_api_usage": api_usage,
                    "llm_cumulative_api_usage": cumulative_api_usage,
                    "llm_token_budget": token_budget,
                    "analysis_required": analysis_required,
                    "planning_tool": "build_execution_plan",
                    "query_contract_tool": "parse_query_contract",
                    "query_task_family": query_contract.get("task_family"),
                    "query_operations": query_contract.get("operation_ids", []),
                    "advisory_task_family": advisory_task_family,
                    "advisory_operation_hints": advisory_operation_hints,
                    "advisory_semantic_anchors": advisory_semantic_anchors,
                    "selected_plan_id": selected_plan_id,
                    "candidate_count": len(planning_candidates),
                    "valid_candidate_count": sum(item["valid"] for item in planning_candidates),
                },
            ),
            "metrics": _metrics(
                state,
                self.name,
                started,
                mode=mode.value,
                task_count=len(tasks),
                tools_called=2,
            ),
        }


class MetadataAgent:
    name = AgentName.METADATA

    def __init__(self, services: AgentServices):
        self.services = services

    def _run_schema_only(self, state: SharedGlobalState) -> AgentUpdate:
        """Use a static schema without ontology retrieval or path discovery."""

        started = time.perf_counter()
        query = state["query"]
        registry = self.services.tool_registry()
        calls = list(state.get("tool_calls", []))
        issues = list(state.get("issues", []))
        pointers_payload, calls, mapping_call = append_tool_call(
            calls,
            registry,
            self.name,
            "map_ontology_entities",
            {"query": query},
        )
        if mapping_call.status == ToolCallStatus.ERROR:
            issues.append(
                _tool_failure_issue(
                    self.name,
                    mapping_call.tool_name,
                    mapping_call.error,
                    error_kind=mapping_call.error_kind,
                )
            )
            pointers_payload = []
        pointers = list(pointers_payload or [])
        candidates = [
            {
                "uri": item["uri"],
                "label": item["label"],
                "entity_kind": "concept",
                "semantic_roles": [],
                "source_graphs": [str(self.services.config.complete_schema)],
                "matched_phrases": [item["mention"]],
                "final_score": item["confidence"],
                "evidence": [item["evidence"]],
            }
            for item in pointers
        ]
        verification = {
            "sufficient": False,
            "structural_backbone_present": False,
            "required_roles": [],
            "present_roles": [],
            "missing_roles": [],
            "required_properties": [],
            "present_properties": [],
            "missing_properties": [],
            "granularity_satisfied": True,
            "boundary_satisfied": True,
            "reasons": [
                "Semantic backbone verification is unavailable in schema-only mode."
            ],
            "missing_concepts": [],
        }
        disabled = [
            "CQ-driven subontology selection",
            "hybrid ontology retrieval",
            "related-entity and relation expansion",
            "semantic-path resolution",
            "backbone verification",
            "bounded ontology retry",
            "query-specific subontology projection",
        ]
        metadata_context = {
            "ontology_mode": "schema_only",
            "schema_conformance": {
                # Without ontology types, require a lexical match or a contracted data axis.
                "passed": (
                    mapping_call.status != ToolCallStatus.ERROR
                    and bool(
                        candidates
                        or state.get("query_contract", {}).get("target_stages")
                        or state.get("query_contract", {}).get("target_scopes")
                        or state.get("query_contract", {}).get("resource_targets")
                    )
                ),
                "basis": "flat lexical evidence or contracted data axis present",
            },
            "backend": {
                "lexical_backend": "static_alias_dictionary",
                "dense_backend": "disabled",
                "reranker_backend": "disabled",
                "offline": True,
                "fallback_used": False,
            },
            "rounds": [],
            "final_candidates": candidates,
            "final_relations": [],
            "final_paths": [],
            "final_verification": verification,
            "retry_count": 0,
            "max_rounds": 1,
            "termination_reason": "schema_only_no_semantic_reasoning",
            "reference_profile_ids": [],
            "reference_signature_terms": [],
            "reference_signature_path_ids": [],
            "disabled_capabilities": disabled,
        }
        ontology_context = {
            "ontology_mode": "schema_only",
            "profile_ids": [],
            "selection_basis": "static schema typing only",
            "ontology_terms": sorted({item["label"] for item in candidates}),
            "relation_count": 0,
            "semantic_path_ids": [],
            "sufficient": False,
            "schema_conformance_passed": metadata_context["schema_conformance"]["passed"],
            "retry_count": 0,
            "termination_reason": metadata_context["termination_reason"],
            "disabled_capabilities": disabled,
        }
        shared_space = _update_shared_space(
            state,
            entity_registry={
                "status": "schema_conformant"
                if metadata_context["schema_conformance"]["passed"]
                else "schema_mapping_failed",
                "ontology_mode": "schema_only",
                "entities": candidates,
                "relations": [],
                "semantic_paths": [],
                "verification": verification,
                "retry_count": 0,
            },
            analytics_and_provenance={
                "metadata_rounds": 0,
                "metadata_termination": metadata_context["termination_reason"],
                "ontology_mode": "schema_only",
            },
        )
        return {
            "cq_matches": [],
            "entity_pointers": pointers,
            "ontology_context": ontology_context,
            "metadata_retrieval": metadata_context,
            "metadata_rounds": [],
            "shared_global_space": shared_space,
            "issues": _deduplicate_issues(issues),
            "tool_calls": calls,
            "provenance": _provenance(
                state,
                self.name,
                "typed direct query mentions against the static ontology schema without semantic ontology functions",
                sources=[str(self.services.config.complete_schema)],
                details={
                    "ontology_mode": "schema_only",
                    "direct_schema_pointers": len(pointers),
                    "relations": 0,
                    "semantic_paths": 0,
                    "retry_count": 0,
                    "disabled_capabilities": disabled,
                    "tools": [mapping_call.tool_name],
                },
            ),
            "metrics": _metrics(
                state,
                self.name,
                started,
                ontology_mode="schema_only",
                entity_pointers=len(pointers),
                relation_count=0,
                semantic_path_count=0,
                retry_count=0,
                schema_conformance_passed=metadata_context["schema_conformance"]["passed"],
                tools_called=1,
            ),
        }

    def run(self, state: SharedGlobalState) -> AgentUpdate:
        if self.services.ontology_mode == "schema_only":
            return self._run_schema_only(state)
        started = time.perf_counter()
        query = state["query"]
        registry = self.services.tool_registry()
        calls = list(state.get("tool_calls", []))
        issues = list(state.get("issues", []))
        called_tools: list[str] = []

        def invoke(tool_name: str, arguments: dict[str, Any]) -> Any:
            nonlocal calls
            result, calls, record = append_tool_call(
                calls,
                registry,
                self.name,
                tool_name,
                arguments,
            )
            called_tools.append(tool_name)
            if record.status == ToolCallStatus.ERROR:
                issues.append(_tool_failure_issue(self.name, tool_name, record.error, error_kind=record.error_kind))
                return None
            return result

        matches, calls, cq_call = append_tool_call(
            calls,
            registry,
            self.name,
            "search_competency_questions",
            {"query": query, "limit": 5},
        )
        called_tools.append(cq_call.tool_name)
        if cq_call.status == ToolCallStatus.ERROR:
            issues.append(_tool_failure_issue(self.name, cq_call.tool_name, cq_call.error, error_kind=cq_call.error_kind))
        matches = matches or []
        cq_ids = [item["cq_id"] for item in matches]

        tdg_tasks = state.get("tdg", {}).get("tasks", [])
        metadata_task = next(
            (item for item in tdg_tasks if item.get("agent") == self.name.value),
            {},
        )
        anchor_terms = list(metadata_task.get("anchor_terms", []))
        parsed = invoke(
            "parse_semantic_query",
            {
                "query": query,
                "mode": state["mode"],
                "anchor_terms": anchor_terms,
                "cq_ids": cq_ids,
            },
        )
        profile_projection = invoke(
            "select_ontology_profiles",
            {
                "query": query,
                "mode": state["mode"],
                "cq_ids": cq_ids,
                "preferred_profile_ids": state.get("query_contract", {}).get(
                    "planner_profile_ids", []
                ),
            },
        ) or {"profile_ids": [], "selection_basis": "unavailable", "governing_cq_ids": cq_ids}

        if parsed is None:
            issues.append(
                ReadinessIssue(
                    code="METADATA_PARSE_FAILED",
                    severity=Severity.BLOCKING,
                    message="The query could not be converted into the typed ontology retrieval contract.",
                    source="metadata",
                ).model_dump(mode="json")
            )
            return {
                "cq_matches": matches,
                "entity_pointers": [],
                "ontology_context": profile_projection,
                "metadata_retrieval": {},
                "metadata_rounds": [],
                "issues": _deduplicate_issues(issues),
                "tool_calls": calls,
                "shared_global_space": _update_shared_space(
                    state,
                    entity_registry={"status": "failed", "failure": "METADATA_PARSE_FAILED"},
                ),
                "provenance": _provenance(
                    state,
                    self.name,
                    "failed closed while parsing the query-specific ontology contract",
                ),
                "metrics": _metrics(state, self.name, started, tools_called=len(called_tools)),
            }

        interpretation = QueryInterpretation.model_validate(parsed["interpretation"])
        phrases = list(parsed["query_phrases"])
        rounds: list[MetadataRetrievalRound] = []
        accumulated_candidates: list[OntologyEntityCandidate] = []
        accumulated_relations: list[OntologyRelation] = []
        final_paths: list[ResolvedSemanticPath] = []
        previous_uris: set[str] = set()

        for round_no in range(1, self.services.config.metadata_max_rounds + 1):
            seed_payload = invoke(
                "hybrid_ontology_retrieval",
                {
                    "interpretation": interpretation.model_dump(mode="json"),
                    "phrases": phrases,
                },
            ) or []
            seeds = [OntologyEntityCandidate.model_validate(item) for item in seed_payload]
            expansion = invoke(
                "expand_related_ontology",
                {
                    "interpretation": interpretation.model_dump(mode="json"),
                    "seeds": [item.model_dump(mode="json") for item in seeds],
                },
            ) or {"candidates": [], "relations": []}
            expanded = [
                OntologyEntityCandidate.model_validate(item)
                for item in expansion.get("candidates", [])
            ]
            new_relations = [
                OntologyRelation.model_validate(item)
                for item in expansion.get("relations", [])
            ]
            candidates = _deduplicate_semantic_candidates(
                [*accumulated_candidates, *expanded]
            )
            relations = _deduplicate_semantic_relations(
                [*accumulated_relations, *new_relations]
            )
            path_payload = invoke(
                "resolve_ontology_paths",
                {
                    "interpretation": interpretation.model_dump(mode="json"),
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                },
            ) or []
            paths = [ResolvedSemanticPath.model_validate(item) for item in path_payload]
            verification_payload = invoke(
                "verify_query_backbone",
                {
                    "interpretation": interpretation.model_dump(mode="json"),
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                    "relations": [item.model_dump(mode="json") for item in relations],
                    "paths": [item.model_dump(mode="json") for item in paths],
                },
            )
            if verification_payload is None:
                verification = BackboneVerification(
                    sufficient=False,
                    structural_backbone_present=False,
                    required_roles=interpretation.required_roles,
                    present_roles=[],
                    missing_roles=interpretation.required_roles,
                    required_properties=interpretation.required_properties,
                    present_properties=[],
                    missing_properties=interpretation.required_properties,
                    reasons=["The schema-grounded sufficiency verifier failed."],
                    missing_concepts=[role.value for role in interpretation.required_roles],
                )
            else:
                verification = BackboneVerification.model_validate(verification_payload)

            current_uris = {item.uri for item in candidates}
            novel = len(current_uris - previous_uris)
            supplements: list[str] = []
            if interpretation.boundary_status == "outside_active_boundary":
                stop_reason = "outside_active_boundary"
            elif verification.sufficient:
                stop_reason = "sufficient"
            elif novel == 0 and round_no > 1:
                stop_reason = "no_novel_candidates"
            elif round_no >= self.services.config.metadata_max_rounds:
                stop_reason = "iteration_limit"
            else:
                supplements = invoke(
                    "supplement_missing_concepts",
                    {
                        "interpretation": interpretation.model_dump(mode="json"),
                        "verification": verification.model_dump(mode="json"),
                        "previous_phrases": phrases,
                    },
                ) or []
                stop_reason = "retry_required" if supplements else "no_novel_candidates"

            rounds.append(
                MetadataRetrievalRound(
                    round_no=round_no,
                    query_phrases=phrases,
                    seed_candidates=seeds,
                    expanded_candidates=expanded,
                    relations=relations,
                    semantic_paths=paths,
                    verification=verification,
                    supplementary_terms=supplements,
                    novel_candidate_count=novel,
                    stop_reason=stop_reason,
                )
            )
            accumulated_candidates = candidates
            accumulated_relations = relations
            final_paths = paths
            previous_uris = current_uris
            if stop_reason != "retry_required":
                break
            phrases = list(dict.fromkeys([*phrases, *supplements]))

        final_verification = rounds[-1].verification
        pipeline = self.services.metadata_pipeline or build_metadata_pipeline(
            self.services.config,
            self.services.cq_catalogue,
        )
        signature_profile_ids, signature_terms, signature_paths = pipeline.project_reference_signature(
            query,
            QueryMode(state["mode"]),
            cq_ids,
        )
        profile_ids = list(signature_profile_ids)
        planner_profile_hints = list(profile_projection.get("profile_ids", []))
        if bool(state.get("query_contract", {}).get("llm_contract_validated")):
            profile_ids = _coverage_preserving_profile_projection(
                planner_profile_hints,
                signature_profile_ids,
            )
        else:
            # Keep the full CQ signature when no validated planner contract is available.
            profile_ids = list(dict.fromkeys(profile_ids))
        # Keep profile obligations from both the validated plan and the CQ route.
        metadata_context = MetadataRetrievalContext(
            interpretation=interpretation,
            backend=parsed["backend"],
            rounds=rounds,
            final_candidates=accumulated_candidates,
            final_relations=accumulated_relations,
            final_paths=final_paths,
            final_verification=final_verification,
            retry_count=max(0, len(rounds) - 1),
            max_rounds=self.services.config.metadata_max_rounds,
            termination_reason=rounds[-1].stop_reason,
            reference_profile_ids=profile_ids,
            reference_signature_terms=signature_terms,
            reference_signature_path_ids=signature_paths,
        )
        pointers = [
            {
                "mention": "; ".join(item.matched_phrases) or item.label,
                "uri": item.uri,
                "label": item.label,
                "confidence": item.final_score,
                "evidence": "; ".join(item.evidence),
                "entity_kind": item.entity_kind.value,
                "semantic_roles": [role.value for role in item.semantic_roles],
                "source_graphs": item.source_graphs,
            }
            for item in accumulated_candidates
        ]
        ontology_context = {
            **profile_projection,
            "profile_ids": profile_ids,
            "planner_profile_hints": planner_profile_hints,
            "reference_signature_profile_ids": list(signature_profile_ids),
            "selection_basis": (
                "dynamic entity/relation retrieval with coverage-preserving semantic profile closure"
            ),
            "profile_projection_strategy": (
                "coverage_preserving_semantic_closure"
                if state.get("query_contract", {}).get("llm_contract_validated")
                else "deterministic_cq_fallback"
            ),
            "ontology_terms": sorted({item.label for item in accumulated_candidates}),
            "relation_count": len(accumulated_relations),
            "semantic_path_ids": [item.path_id for item in final_paths],
            "backend": parsed["backend"],
            "index_manifest": parsed["index_manifest"],
            "sufficient": final_verification.sufficient,
            "retry_count": metadata_context.retry_count,
            "termination_reason": metadata_context.termination_reason,
        }

        new_issues: list[dict[str, Any]] = []
        if not matches or float(matches[0]["score"]) < 0.03:
            new_issues.append(
                ReadinessIssue(
                    code="LOW_CQ_ALIGNMENT",
                    severity=Severity.WARNING,
                    message="The request has weak alignment with the governed CQ catalogue.",
                    source="metadata",
                ).model_dump(mode="json")
            )
        if any(not match.get("path_valid", True) for match in matches):
            new_issues.append(
                ReadinessIssue(
                    code="INVALID_SEMANTIC_PATH",
                    severity=Severity.BLOCKING,
                    message="A retrieved CQ path violates the S-E-P-R-C routing contract.",
                    source="metadata",
                ).model_dump(mode="json")
            )
        if (
            interpretation.boundary_status != "outside_active_boundary"
            and not final_verification.sufficient
        ):
            new_issues.append(
                ReadinessIssue(
                    code="METADATA_BACKBONE_INSUFFICIENT",
                    severity=Severity.BLOCKING,
                    message="Ontology retrieval stopped without a complete query-specific semantic backbone: "
                    + "; ".join(final_verification.reasons),
                    missing_fields=final_verification.missing_concepts,
                    source="metadata",
                ).model_dump(mode="json")
            )
        shared_space = _update_shared_space(
            state,
            entity_registry={
                "status": "sufficient" if final_verification.sufficient else "insufficient",
                "interpretation": interpretation.model_dump(mode="json"),
                "entities": [item.model_dump(mode="json") for item in accumulated_candidates],
                "relations": [item.model_dump(mode="json") for item in accumulated_relations],
                "semantic_paths": [item.model_dump(mode="json") for item in final_paths],
                "verification": final_verification.model_dump(mode="json"),
                "backend": parsed["backend"],
                "retry_count": metadata_context.retry_count,
            },
            analytics_and_provenance={
                "metadata_rounds": len(rounds),
                "metadata_termination": metadata_context.termination_reason,
            },
        )
        return {
            "cq_matches": matches,
            "entity_pointers": pointers,
            "ontology_context": ontology_context,
            "metadata_retrieval": metadata_context.model_dump(mode="json"),
            "metadata_rounds": [item.model_dump(mode="json") for item in rounds],
            "shared_global_space": shared_space,
            "issues": _deduplicate_issues(issues + new_issues),
            "tool_calls": calls,
            "provenance": _provenance(
                state,
                self.name,
                "resolved and verified a query-specific ontology context with bounded local retry",
                sources=[
                    str(self.services.config.cq_catalogue),
                    str(self.services.config.complete_schema),
                    str(self.services.config.cpd_reference),
                    str(self.services.config.efd_reference),
                ],
                details={
                    "cq_matches": len(matches),
                    "entity_pointers": len(pointers),
                    "relations": len(accumulated_relations),
                    "semantic_paths": len(final_paths),
                    "sufficient": final_verification.sufficient,
                    "retry_count": metadata_context.retry_count,
                    "termination_reason": metadata_context.termination_reason,
                    "backend": parsed["backend"],
                    "tools": called_tools,
                },
            ),
            "metrics": _metrics(
                state,
                self.name,
                started,
                cq_matches=len(matches),
                entity_pointers=len(pointers),
                relation_count=len(accumulated_relations),
                semantic_path_count=len(final_paths),
                retry_count=metadata_context.retry_count,
                sufficient=final_verification.sufficient,
                tools_called=len(called_tools),
            ),
        }


class ReaderAgent:
    name = AgentName.READER

    def __init__(self, services: AgentServices):
        self.services = services

    def run(self, state: SharedGlobalState) -> AgentUpdate:
        started = time.perf_counter()
        query = state["query"]
        lowered = query.lower()
        normalised = lowered.replace("-", " ")
        issues = list(state.get("issues", []))
        context: dict[str, Any] = {}
        registry = self.services.tool_registry()
        calls = list(state.get("tool_calls", []))
        called_tools: list[str] = []

        executed_templates: list[dict[str, Any]] = []
        reference_searches: list[dict[str, Any]] = []
        result_by_step: dict[str, Any] = {}
        updated_delta = dict(state.get("delta_graph_specification", {}))
        updated_overlay = dict(state.get("scenario_overlay", {}))

        def invoke(tool_name: str, arguments: dict[str, Any]) -> Any:
            nonlocal calls
            result, calls, record = append_tool_call(
                calls,
                registry,
                self.name,
                tool_name,
                arguments,
            )
            called_tools.append(tool_name)
            if record.status == ToolCallStatus.ERROR:
                issues.append(_tool_failure_issue(self.name, tool_name, record.error, error_kind=record.error_kind))
                return None
            return result

        readiness = invoke("check_knowledge_readiness", {}) or {"ready": False}
        plan_payload = invoke(
            "plan_reader_retrieval",
            {
                "query": query,
                "metadata_retrieval": state.get("metadata_retrieval", {}),
                "cq_matches": state.get("cq_matches", []),
                "delta_graph_specification": state.get("delta_graph_specification", {}),
                "query_contract": state.get("query_contract", {}),
            },
        )
        if plan_payload is None:
            issues.append(
                ReadinessIssue(
                    code="READER_PLAN_FAILED",
                    severity=Severity.BLOCKING,
                    message="The verified metadata context could not be converted into a dual-path retrieval plan.",
                    source="reader",
                ).model_dump(mode="json")
            )
            plan = ReaderRetrievalPlan(
                query=query,
                steps=[],
                target_databases=[],
                uses_ckg_template_path=False,
                uses_reference_hybrid_path=False,
            )
        else:
            plan = ReaderRetrievalPlan.model_validate(plan_payload)

        ontology_profiles = state.get("ontology_context", {}).get("profile_ids", [])
        if ontology_profiles:
            subontology = invoke(
                "read_subontology_signature",
                {
                    "profile_ids": ontology_profiles,
                    "query": query,
                    "mode": state["mode"],
                    "cq_ids": [
                        item.get("cq_id")
                        for item in state.get("cq_matches", [])
                        if item.get("cq_id")
                    ],
                },
            )
            if subontology:
                context["subontology"] = subontology

        if not readiness.get("ready"):
            issues.append(
                ReadinessIssue(
                    code="KNOWLEDGE_INDEX_NOT_READY",
                    severity=Severity.BLOCKING,
                    message="The governed CKG index is absent or stale; run the prepare command.",
                    source="reader",
                ).model_dump(mode="json")
            )
        else:
            for step in plan.steps:
                if step.protocol == "CKG_TEMPLATE":
                    bindings = dict(step.bindings)
                    source_step = bindings.pop("element_uri_from_step", None)
                    if source_step:
                        seed = result_by_step.get(str(source_step), [])
                        if isinstance(seed, list) and seed:
                            bindings["element_uri"] = seed[0].get("element_uri")
                            bindings.setdefault("limit", 250)
                        else:
                            issues.append(
                                ReadinessIssue(
                                    code="CKG_TEMPLATE_BINDING_MISSING",
                                    severity=Severity.BLOCKING,
                                    message=f"{step.template_id} requires an element URI from {source_step}.",
                                    missing_fields=["element_uri"],
                                    source="reader",
                                ).model_dump(mode="json")
                            )
                            continue
                    payload = invoke(
                        "execute_ckg_sparql_template",
                        {"template_id": step.template_id, "bindings": bindings},
                    )
                    if not payload:
                        continue
                    executed_templates.append(payload)
                    result = payload.get("result")
                    result_by_step[step.step_id] = result
                    if step.template_id == "CKG-PROJECT-OVERVIEW":
                        context["project_overview"] = result or {}
                    elif step.template_id == "CKG-TOP-ELEMENTS":
                        if step.step_id == "READ-CKG-TRACE-SEED":
                            context["trace_seed"] = result or []
                        else:
                            context["hotspots"] = result or []
                    elif step.template_id == "CKG-ELEMENT-TRACE":
                        context["representative_element_trace"] = result or {}
                    elif step.template_id == "CKG-RESOURCE-BREAKDOWN":
                        resource_result = result or {}
                        resource_name = str(
                            bindings.get("term")
                            or resource_result.get("term")
                            or step.target
                        )
                        context.setdefault("resource_breakdowns", {})[
                            resource_name.lower()
                        ] = resource_result
                        if "resource_breakdown" not in context:
                            context["resource_breakdown"] = resource_result
                    elif step.template_id == "CKG-WORK-PACKAGES":
                        context["work_packages"] = result or []
                elif step.protocol == "REFERENCE_HYBRID":
                    payload = invoke(
                        "hybrid_reference_lookup",
                        {
                            "query": query,
                            "categories": [step.category_filter] if step.category_filter else [],
                            "hard_filters": step.hard_filters,
                            "top_k": 8,
                        },
                    )
                    if payload:
                        reference_searches.append(
                            {"step_id": step.step_id, "target": step.target, **payload}
                        )
                        result_by_step[step.step_id] = payload
                elif (
                    step.protocol == "REFERENCE_EXACT"
                    and step.step_id == "READ-REFERENCE-IFC-INVENTORY"
                ):
                    context["ifc_inventory"] = invoke(
                        "read_reference_artifact",
                        {"dataset": "ifc_inventory", "filters": {}, "limit": 100},
                    ) or {}

            overview = context.get("project_overview", {})
            reference_ready = overview.get("assessment_status") == "COMPLETE_REFERENCE_CASE"

            query_contract = state.get("query_contract", {})
            contract_operations = set(query_contract.get("operation_ids", []))
            contract_scopes = set(query_contract.get("target_scopes", []))
            llm_contract = bool(query_contract.get("llm_contract_validated"))
            cpd_requested = "retrieve_cpd_efd" in contract_operations or (
                not llm_contract
                and any(
                    term in lowered
                    for term in (
                        "cpd",
                        "work-package code",
                        "work package code",
                        "工作包编码",
                        "cpd编码",
                    )
                )
            )
            if cpd_requested:
                if "ONT-FOUNDATION-EARTHWORK" in ontology_profiles or "foundation_pit" in contract_scopes or (
                    not llm_contract
                    and any(
                        term in normalised
                        for term in ("foundation pit", "earthwork", "基坑", "土方")
                    )
                ):
                    context["foundation_cpd_efd"] = invoke(
                        "read_reference_artifact",
                        {"dataset": "foundation_cpd_efd", "filters": {}, "limit": 500},
                    ) or {}
                elif "ONT-ENCLOSURE-CPD" in ontology_profiles or (
                    not llm_contract
                    and any(
                        term in lowered
                        for term in ("diaphragm", "地下连续墙", "地连墙")
                    )
                ):
                    coverage_request = llm_contract or any(
                        term in lowered
                        for term in ("coverage", "every", "all cpd", "complete cpd")
                    )
                    context["enclosure_cpd_efd"] = invoke(
                        "read_reference_artifact",
                        {
                            "dataset": "enclosure_cpd_efd",
                            "filters": (
                                {}
                                if coverage_request
                                else {"owner_type": "Type_DiaphragmWall"}
                            ),
                            "limit": 500,
                        },
                    ) or {}
                else:
                    context["enclosure_cpd_efd"] = invoke(
                        "read_reference_artifact",
                        {"dataset": "enclosure_cpd_efd", "filters": {}, "limit": 500},
                    ) or {}

            transport_requested = (
                "retrieve_transport_references" in contract_operations
                or "TRANSPORT" in set(query_contract.get("reference_categories", []))
                or "ONT-A4-TRANSPORT" in ontology_profiles
                or (
                    not llm_contract
                    and any(
                        term in lowered
                        for term in ("a4", "transport", "haul", "vehicle", "运输", "运距", "载具")
                    )
                )
            )
            if transport_requested:
                contract_material = next(
                    (
                        str(value)
                        for value in query_contract.get("explicit_identifiers", [])
                        if re.fullmatch(r"M-\d+-\d+", str(value), re.I)
                    ),
                    None,
                )
                material_match = re.search(r"\bm-\d+-\d+\b", lowered) if not contract_material else None
                direct_filters = (
                    {
                        "material_code": (
                            contract_material.upper()
                            if contract_material
                            else material_match.group(0).upper()
                        )
                    }
                    if contract_material or material_match
                    else {}
                )
                candidate_defaults = invoke(
                    "read_reference_artifact",
                    {
                        "dataset": "material_transport_defaults",
                        "filters": direct_filters,
                        "limit": 100,
                    },
                ) or {}
                requested_labels = transport_material_labels("", query_contract)
                matched_defaults = matching_transport_defaults(
                    candidate_defaults.get("rows", []),
                    requested_labels,
                    query=query,
                )
                if direct_filters:
                    matched_defaults = candidate_defaults.get("rows", [])
                resolved_material_code = (
                    str(matched_defaults[0].get("material_code", "")).upper()
                    if len(matched_defaults) == 1
                    else ""
                )
                resolved_filters = (
                    {"material_code": resolved_material_code}
                    if resolved_material_code
                    else direct_filters
                )
                if resolved_filters and resolved_filters != direct_filters:
                    context["material_transport_defaults"] = invoke(
                        "read_reference_artifact",
                        {
                            "dataset": "material_transport_defaults",
                            "filters": resolved_filters,
                            "limit": 100,
                        },
                    ) or {}
                else:
                    context["material_transport_defaults"] = candidate_defaults
                context["a4_transport_legs"] = invoke(
                    "read_reference_artifact",
                    {
                        "dataset": "a4_transport_legs",
                        "filters": resolved_filters,
                        "limit": 500,
                        "sort_by": "emission_kgco2e",
                        "descending": True,
                    },
                ) or {}
                default_rows = context["material_transport_defaults"].get("rows", [])
                vehicle_filters: dict[str, str] = {}
                if (
                    len(default_rows) == 1
                    and default_rows[0].get("vehicle_code")
                ):
                    vehicle_filters["vehicle_code"] = str(
                        default_rows[0]["vehicle_code"]
                    )
                context["vehicle_factors"] = invoke(
                    "read_reference_artifact",
                    {
                        "dataset": "vehicle_factors",
                        "filters": vehicle_filters,
                        "limit": 100,
                    },
                ) or {}
                material_resolution_required = (
                    bool(requested_labels or direct_filters)
                    and not bool(query_contract.get("meta_ontology_query"))
                    and (
                        str(query_contract.get("primary_operator"))
                        in {"trace", "transaction_preview"}
                        or str(query_contract.get("task_family"))
                        in {"correction_preview", "correction_impact_preview"}
                    )
                )
                if material_resolution_required and len(matched_defaults) != 1:
                    if not any(
                        issue.get("code") == "REFERENCE_DEMAND_UNRESOLVED"
                        for issue in issues
                    ):
                        issues.append(
                            ReadinessIssue(
                                code="REFERENCE_DEMAND_UNRESOLVED",
                                severity=Severity.BLOCKING,
                                message=(
                                    "The material or vehicle reference requested by the semantic contract "
                                    "did not resolve to one governed transport record."
                                ),
                                missing_fields=["one uniquely resolved transport reference"],
                                source="reader_entity_resolution",
                            ).model_dump(mode="json")
                        )
                context["transport_entity_resolution"] = {
                    "requested_labels": requested_labels,
                    "match_count": len(matched_defaults),
                    "resolved_material_code": resolved_material_code or None,
                    "method": "controlled_label_unique_match",
                }

            foundation_query = any(
                term in normalised
                for term in (
                    "foundation pit",
                    "earthwork",
                    "excavation volume",
                    "implicit excavation",
                    "excavation and backfill",
                    "基坑土方",
                    "土方量",
                )
            )
            if foundation_query and reference_ready:
                context["foundation_pit"] = overview.get("foundation_pit", {})
            elif foundation_query:
                issues.append(
                    ReadinessIssue(
                        code="FOUNDATION_PIT_GEOMETRY_MISSING",
                        severity=Severity.BLOCKING,
                        message="Build the declared reference-case profile before requesting foundation-pit results.",
                        source="reader",
                    ).model_dump(mode="json")
                )

            if not reference_ready:
                issues.append(
                    ReadinessIssue(
                        code="PARTIAL_PROJECT_BASELINE",
                        severity=Severity.WARNING,
                        message="The combined reference-case artifact is absent; only the governed indexed baseline is available.",
                        source="reader",
                    ).model_dump(mode="json")
                )
                issues.append(
                    ReadinessIssue(
                        code="ENCLOSURE_NOT_IN_CARBON_BASELINE",
                        severity=Severity.WARNING,
                        message="Build the reference case before reporting a combined main/enclosure/foundation-pit total.",
                        source="reader",
                    ).model_dump(mode="json")
                )

        if (
            state.get("mode") == QueryMode.STRATEGIC_ANALYSIS.value
            and updated_delta.get("reference_demands")
        ):
            resolution = invoke(
                "resolve_reference_demands",
                {
                    "delta_graph": updated_delta,
                    "overlay": updated_overlay,
                    "reference_searches": reference_searches,
                    "exact_artifacts": {
                        key: value
                        for key, value in context.items()
                        if key
                        in {
                            "foundation_cpd_efd",
                            "enclosure_cpd_efd",
                            "a4_transport_legs",
                            "material_transport_defaults",
                            "vehicle_factors",
                        }
                    },
                },
            )
            if resolution:
                updated_delta = resolution["delta_graph"]
                updated_overlay = resolution["overlay"]
                unresolved_demands = resolution.get("unresolved_demands", [])
                preserved_delta_issues = list(updated_delta.get("issues", []))
                if unresolved_demands:
                    # Report the original readiness issue when lookup cannot resolve the same fields.
                    issues.extend(preserved_delta_issues)
                context["reference_demand_resolution"] = {
                    "resolved_fields": resolution.get("resolved_fields", {}),
                    "unresolved_demands": unresolved_demands,
                }
                for demand in unresolved_demands:
                    required_fields = set(demand.get("required_fields", []))
                    already_explained = any(
                        item.get("severity") == Severity.BLOCKING.value
                        and bool(required_fields & set(item.get("missing_fields", [])))
                        for item in preserved_delta_issues
                    )
                    if already_explained:
                        continue
                    issues.append(
                        ReadinessIssue(
                            code="REFERENCE_DEMAND_UNRESOLVED",
                            severity=Severity.BLOCKING,
                            message=(
                                f"No unambiguous governed {demand.get('category')} record resolved "
                                f"{', '.join(demand.get('required_fields', []))}."
                            ),
                            missing_fields=demand.get("required_fields", []),
                            source="reader",
                        ).model_dump(mode="json")
                    )

        if not state.get("query_contract", {}).get("meta_boundary_query") and (re.search(
            r"(?<![a-z0-9])(?:b[1-7]|c[1-4]|d)(?![a-z0-9])",
            lowered,
        ) or any(
            term in lowered
            for term in (
                "operation stage",
                "operational-use",
                "end of life",
                "deconstruction",
                "demolition",
                "recycling benefit",
                "运营阶段",
                "运营使用",
                "拆除阶段",
                "回收收益",
                "边界外",
                "d模块",
            )
        )):
            issues.append(
                ReadinessIssue(
                    code="OUTSIDE_A1_A5_SCOPE",
                    severity=Severity.BLOCKING,
                    message="The requested life-cycle module is outside the declared A1-A5 assessment boundary.",
                    source="reader",
                ).model_dump(mode="json")
            )

        if reference_searches:
            context["reference_searches"] = reference_searches
        source_ids = sorted(
            {
                *(
                    str(item.get("sparql_file"))
                    for item in executed_templates
                    if item.get("sparql_file")
                ),
                *(
                    str(result.get("source"))
                    for search in reference_searches
                    for result in search.get("results", [])
                    if result.get("source")
                ),
                *(
                    str(value.get("source"))
                    for value in context.values()
                    if isinstance(value, dict) and value.get("source")
                ),
            }
        )
        metadata_context = state.get("metadata_retrieval", {})
        final_candidates = metadata_context.get("final_candidates", [])
        final_relations = metadata_context.get("final_relations", [])
        final_paths = metadata_context.get("final_paths", [])
        missing_context: list[str] = []
        if not readiness.get("ready"):
            missing_context.append("current CKG index")
        if "project_overview" not in context:
            missing_context.append("project assessment baseline")
        if plan.uses_reference_hybrid_path and not reference_searches:
            missing_context.append("CPD/EFD/transport reference candidates")
        composite = CompositeDataContext(
            baseline_facts=context,
            reference_parameters={
                "hybrid_searches": reference_searches,
                "exact_artifacts": {
                    key: value
                    for key, value in context.items()
                    if key
                    in {
                        "foundation_cpd_efd",
                        "enclosure_cpd_efd",
                        "a4_transport_legs",
                        "material_transport_defaults",
                        "vehicle_factors",
                    }
                },
            },
            retrieval_plan=plan,
            executed_templates=executed_templates,
            reference_searches=reference_searches,
            ontology_entities=[
                str(item.get("uri"))
                for item in final_candidates
                if item.get("uri")
            ],
            ontology_relations=[
                f"{item.get('subject_uri')}|{item.get('predicate_uri')}|{item.get('object_uri')}"
                for item in final_relations
            ],
            semantic_path_ids=[
                str(item.get("path_id"))
                for item in final_paths
                if item.get("path_id")
            ],
            source_ids=source_ids,
            complete=not missing_context,
            missing_context=missing_context,
        )
        shared_space = _update_shared_space(
            state,
            baseline_context={
                "status": "complete" if composite.complete else "incomplete",
                "composite_data_context": composite.model_dump(mode="json"),
                "retrieval_plan": plan.model_dump(mode="json"),
            },
            delta_specifications={
                "status": (
                    "ready"
                    if updated_delta.get("ready")
                    else "incomplete"
                ),
                "delta_graph": updated_delta,
            }
            if updated_delta
            else None,
            analytics_and_provenance={
                "reader_template_count": len(executed_templates),
                "reader_reference_search_count": len(reference_searches),
                "reader_source_ids": source_ids,
            },
        )
        return {
            "retrieved_context": context,
            "composite_data_context": composite.model_dump(mode="json"),
            "scenario_overlay": updated_overlay,
            "delta_graph_specification": updated_delta,
            "shared_global_space": shared_space,
            "issues": _deduplicate_issues(issues),
            "tool_calls": calls,
            "provenance": _provenance(
                state,
                self.name,
                "executed the dual-path CKG-template and reference-database retrieval plan",
                sources=source_ids,
                details={
                    "context_blocks": sorted(context),
                    "target_databases": plan.target_databases,
                    "executed_templates": [
                        item.get("template_id") for item in executed_templates
                    ],
                    "reference_search_count": len(reference_searches),
                    "hard_filters": [
                        item.get("hard_filters", {}) for item in reference_searches
                    ],
                    "composite_complete": composite.complete,
                    "tools": called_tools,
                },
            ),
            "metrics": _metrics(
                state,
                self.name,
                started,
                context_blocks=len(context),
                template_count=len(executed_templates),
                reference_search_count=len(reference_searches),
                composite_complete=composite.complete,
                tools_called=len(called_tools),
            ),
        }
class ExtractorAgent:
    name = AgentName.EXTRACTOR

    def __init__(self, services: AgentServices):
        self.services = services

    def run(self, state: SharedGlobalState) -> AgentUpdate:
        started = time.perf_counter()
        mode = QueryMode(state["mode"])
        persistent = mode == QueryMode.KNOWLEDGE_CORRECTION
        registry = self.services.tool_registry()
        calls = list(state.get("tool_calls", []))
        issues = list(state.get("issues", []))
        overlay, calls, overlay_call = append_tool_call(
            calls,
            registry,
            self.name,
            "extract_scenario_overlay",
            {
                "query": state["query"],
                "persistent": persistent,
                "semantic_contract": state.get("query_contract", {}),
            },
        )
        if overlay_call.status == ToolCallStatus.ERROR:
            issues.append(
                _tool_failure_issue(
                    self.name,
                    overlay_call.tool_name,
                    overlay_call.error,
                    error_kind=overlay_call.error_kind,
                )
            )
            overlay = {
                "overlay_id": "",
                "persistence": "persistent" if persistent else "transient",
                "changes": [],
                "ready": False,
                "issues": [],
            }
        delta_payload, calls, delta_call = append_tool_call(
            calls,
            registry,
            self.name,
            "construct_delta_graph_specification",
            {
                "query": state["query"],
                "mode": mode.value,
                "overlay": overlay,
            },
        )
        if delta_call.status == ToolCallStatus.ERROR or delta_payload is None:
            issues.append(
                _tool_failure_issue(
                    self.name,
                    delta_call.tool_name,
                    delta_call.error,
                    error_kind=delta_call.error_kind,
                )
            )
            delta = DeltaGraphSpecification(
                delta_id="delta-failed",
                mode=mode,
                persistence=(
                    PersistenceMode.PERSISTENT
                    if persistent
                    else PersistenceMode.TRANSIENT
                ),
                ready=False,
                feasibility_status="incomplete",
            )
        else:
            delta = DeltaGraphSpecification.model_validate(delta_payload)

        # Only a resolvable reference demand can defer a blocking issue to retrieval.
        resolvable_categories = {"CPD", "EFD", "TRANSPORT", "DENSITY"}
        demands_deferred = (
            mode == QueryMode.STRATEGIC_ANALYSIS
            and bool(delta.reference_demands)
            and all(
                demand.category in resolvable_categories
                for demand in delta.reference_demands
            )
            and delta.feasibility_status != "forbidden"
        )
        if not demands_deferred:
            issues.extend(
                item.model_dump(mode="json")
                for item in delta.issues
            )

        shared_space = _update_shared_space(
            state,
            delta_specifications={
                "status": (
                    "ready"
                    if delta.ready
                    else "awaiting_reference_resolution"
                    if demands_deferred
                    else delta.feasibility_status
                ),
                "delta_graph": delta.model_dump(mode="json"),
                "explicit_override_count": sum(
                    item.precedence == "explicit_user"
                    for item in delta.overrides
                ),
                "reference_demand_count": len(delta.reference_demands),
            },
            analytics_and_provenance={
                "extractor_overlay_id": overlay.get("overlay_id"),
                "extractor_delta_id": delta.delta_id,
            },
        )
        return {
            "scenario_overlay": overlay,
            "delta_graph_specification": delta.model_dump(mode="json"),
            "shared_global_space": shared_space,
            "issues": _deduplicate_issues(issues),
            "tool_calls": calls,
            "provenance": _provenance(
                state,
                self.name,
                "constructed a typed provenance-annotated delta graph and reference demands",
                details={
                    "overlay_id": overlay.get("overlay_id"),
                    "delta_id": delta.delta_id,
                    "persistence": delta.persistence.value,
                    "ready": delta.ready,
                    "feasibility_status": delta.feasibility_status,
                    "change_count": len(overlay.get("changes", [])),
                    "override_count": len(delta.overrides),
                    "reference_demand_count": len(delta.reference_demands),
                    "reference_resolution_deferred_to_reader": demands_deferred,
                    "tools": [
                        overlay_call.tool_name,
                        delta_call.tool_name,
                    ],
                },
            ),
            "metrics": _metrics(
                state,
                self.name,
                started,
                ready=delta.ready,
                overrides=len(delta.overrides),
                reference_demands=len(delta.reference_demands),
                tools_called=2,
            ),
        }
class CalculatorAgent:
    name = AgentName.CALCULATOR

    def __init__(self, services: AgentServices):
        self.services = services

    def run(self, state: SharedGlobalState) -> AgentUpdate:
        started = time.perf_counter()
        issues = list(state.get("issues", []))
        calls = list(state.get("tool_calls", []))
        context = state.get("retrieved_context", {})
        delta = state.get("delta_graph_specification", {})
        initial_digest = stable_digest(
            {
                "baseline": state.get("composite_data_context", context),
                "delta": delta,
            }
        )
        program = CalculationProgram(
            program_id=f"calc-{initial_digest[:16]}",
            initial_context_digest=initial_digest,
            steps=[],
        )
        blocking = any(
            issue.get("severity") == Severity.BLOCKING.value
            for issue in issues
        )
        if blocking:
            shared_space = _update_shared_space(
                state,
                analytics_and_provenance={
                    "calculation_status": "halted",
                    "calculation_program": program.model_dump(mode="json"),
                },
            )
            return {
                "calculation_results": {},
                "calculation_program": program.model_dump(mode="json"),
                "shared_global_space": shared_space,
                "issues": _deduplicate_issues(issues),
                "tool_calls": calls,
                "provenance": _provenance(
                    state,
                    self.name,
                    "halted before calculation because epistemic prerequisites failed",
                ),
                "metrics": _metrics(
                    state,
                    self.name,
                    started,
                    tools_called=0,
                    transition_count=0,
                ),
            }

        mode = QueryMode(state["mode"])
        compiled: dict[str, Any] | None = None
        compiler_record = None
        record = None
        result: dict[str, Any] = {}
        presentation_result: dict[str, Any] = {}
        step_results: dict[str, dict[str, Any]] = {}
        transitions: list[CalculationStateTransition] = []
        try:
            program_context = {
                key: context[key]
                for key in (
                    "project_overview",
                    "hotspots",
                    "work_packages",
                    "resource_breakdown",
                    "resource_breakdowns",
                    "material_transport_defaults",
                    "vehicle_factors",
                    "a4_transport_legs",
                )
                if key in context
            }
            compiled, calls, compiler_record = append_tool_call(
                calls,
                self.services.tool_registry(),
                self.name,
                "compile_calculation_program",
                {
                    "query": state["query"],
                    "mode": mode.value,
                    "query_contract": state.get("query_contract", {}),
                    "context": program_context,
                    "overlay": state.get("scenario_overlay") or {},
                },
            )
            if compiler_record.status == ToolCallStatus.ERROR or compiled is None:
                raise CalculationPlanningError(
                    "CALCULATION_PROGRAM_COMPILATION_FAILED",
                    compiler_record.error or "The typed calculation program could not be compiled.",
                )
            compiled_steps = [
                CalculationProgramStep.model_validate(item)
                for item in compiled.get("steps", [])
            ]
            if not compiled_steps:
                raise CalculationPlanningError(
                    "EMPTY_CALCULATION_PROGRAM",
                    "The typed compiler returned no executable calculation steps.",
                )
            executed_steps: list[CalculationProgramStep] = []
            for step in compiled_steps:
                missing_dependencies = [
                    dependency
                    for dependency in step.dependencies
                    if dependency not in step_results
                ]
                if missing_dependencies:
                    raise CalculationPlanningError(
                        "CALCULATION_DEPENDENCY_MISSING",
                        f"{step.step_id} is missing completed dependencies: {', '.join(missing_dependencies)}.",
                        missing_dependencies,
                    )
                resolved_arguments = resolve_program_bindings(step.arguments, step_results)
                executed_step = step.model_copy(update={"arguments": resolved_arguments})
                executed_steps.append(executed_step)
                input_state_digest = stable_digest(
                    {
                        "initial_context_digest": initial_digest,
                        "step": executed_step.model_dump(mode="json"),
                        "prior_transitions": [
                            item.model_dump(mode="json") for item in transitions
                        ],
                    }
                )
                step_result, calls, record = append_tool_call(
                    calls,
                    self.services.tool_registry(),
                    self.name,
                    executed_step.tool_name,
                    executed_step.arguments,
                )
                if record.status == ToolCallStatus.ERROR:
                    transitions.append(
                        CalculationStateTransition(
                            step_id=executed_step.step_id,
                            tool_name=executed_step.tool_name,
                            input_state_digest=input_state_digest,
                            status="error",
                            error=record.error,
                        )
                    )
                    raise CalculationPlanningError(
                        "CALCULATION_TOOL_FAILED",
                        record.error or f"{executed_step.tool_name} failed",
                    )
                step_result = step_result or {}
                step_results[executed_step.step_id] = step_result
                transitions.append(
                    CalculationStateTransition(
                        step_id=executed_step.step_id,
                        tool_name=executed_step.tool_name,
                        input_state_digest=input_state_digest,
                        output_state_digest=stable_digest(step_result),
                        result=step_result,
                        status="success",
                    )
                )
            result = resolve_program_bindings(
                compiled.get("final_result_template", {}),
                step_results,
            )
            if not isinstance(result, dict) or not result:
                raise CalculationPlanningError(
                    "CALCULATION_FINAL_RESULT_MISSING",
                    "The completed program did not produce a governed final result.",
                )
            presentation_result = {
                **result,
                **dict(compiled.get("presentation_extensions", {})),
            }
            program = program.model_copy(
                update={
                    "steps": executed_steps,
                    "transitions": transitions,
                    "final_result": result,
                    "complete": True,
                }
            )
        except (
            CalculationPlanningError,
            KeyError,
            TypeError,
            ValueError,
            IndexError,
        ) as exc:
            code = (
                exc.code
                if isinstance(exc, CalculationPlanningError)
                else "CALCULATION_PRECONDITION_FAILED"
            )
            missing_fields = (
                exc.missing_fields
                if isinstance(exc, CalculationPlanningError)
                else []
            )
            issues.append(
                ReadinessIssue(
                    code=code,
                    severity=Severity.BLOCKING,
                    message=str(exc),
                    missing_fields=missing_fields,
                    source="calculator",
                ).model_dump(mode="json")
            )
            result = {}
            presentation_result = {}
            program = program.model_copy(
                update={
                    "transitions": transitions,
                    "final_result": {},
                    "complete": False,
                }
            )

        shared_space = _update_shared_space(
            state,
            analytics_and_provenance={
                "calculation_status": (
                    "complete" if program.complete else "failed"
                ),
                "calculation_program": program.model_dump(mode="json"),
                "calculation_result_digest": (
                    stable_digest(presentation_result) if presentation_result else None
                ),
            },
        )
        return {
            "calculation_results": presentation_result,
            "calculation_program": program.model_dump(mode="json"),
            "shared_global_space": shared_space,
            "issues": _deduplicate_issues(issues),
            "tool_calls": calls,
            "provenance": _provenance(
                state,
                self.name,
                (
                    "executed the baseline-plus-delta calculation program"
                    if presentation_result
                    else "calculation program failed closed"
                ),
                details={
                    "program_id": program.program_id,
                    "complete": program.complete,
                    "tool": result.get("tool"),
                    "call_id": record.call_id if record else None,
                    "compiler_call_id": (
                        compiler_record.call_id if compiler_record else None
                    ),
                    "compiler": compiled.get("compiler") if compiled else None,
                    "step_tools": [step.tool_name for step in program.steps],
                    "initial_context_digest": initial_digest,
                    "transition_count": len(transitions),
                },
            ),
            "metrics": _metrics(
                state,
                self.name,
                started,
                tools_called=(1 if compiler_record else 0) + len(transitions),
                transition_count=len(transitions),
                complete=program.complete,
            ),
        }
class SynthesisAgent:
    name = AgentName.SYNTHESIS

    def __init__(self, services: AgentServices):
        self.services = services

    def run(self, state: SharedGlobalState) -> AgentUpdate:
        started = time.perf_counter()
        mode = QueryMode(state["mode"])
        query_contract = QueryContract.model_validate(state.get("query_contract", {}))
        safety_issues = [
            issue.model_dump(mode="json")
            for issue in assess_query_contract_safety(query_contract, mode)
        ]
        issues = _prioritise_root_cause_issues(
            _deduplicate_issues([*state.get("issues", []), *safety_issues])
        )
        blocking = any(issue.get("severity") == Severity.BLOCKING.value for issue in issues)
        transaction_preview: dict[str, Any] | None = None
        requires_confirmation = False
        registry = self.services.tool_registry()
        calls = list(state.get("tool_calls", []))
        called_tools: list[str] = []

        metadata_payload = state.get("metadata_retrieval", {})
        schema_only = metadata_payload.get("ontology_mode") == "schema_only"
        boundary_context_grounded = bool(
            query_contract.outside_boundary_referenced
            and metadata_payload.get("interpretation", {}).get("boundary_status")
            == "outside_active_boundary"
        )
        metadata_sufficient = (
            bool(metadata_payload.get("schema_conformance", {}).get("passed"))
            if schema_only
            else bool(metadata_payload.get("final_verification", {}).get("sufficient"))
            or boundary_context_grounded
        )
        composite_payload = state.get("composite_data_context", {})
        baseline_context_complete = bool(composite_payload.get("complete"))
        delta_payload = state.get("delta_graph_specification", {})
        delta_complete = (
            mode == QueryMode.INFORMATION_RETRIEVAL
            or bool(delta_payload.get("ready"))
        )
        calculation_payload = state.get("calculation_results", {})
        calculation_program_payload = state.get("calculation_program", {})
        calculation_complete = (
            not bool(state.get("analysis_required"))
            or bool(calculation_program_payload.get("complete"))
            or bool(calculation_payload)
        )
        query_answer_alignment = bool(
            calculation_payload
            if state.get("analysis_required")
            else state.get("retrieved_context")
        )
        gate_missing: list[str] = []
        if not metadata_sufficient:
            gate_missing.append(
                "static ontology-schema conformance"
                if schema_only
                else "verified query-specific ontology backbone"
            )
        if not baseline_context_complete:
            gate_missing.append("complete Reader composite context")
        if not delta_complete:
            gate_missing.append("feasible and complete delta graph")
        if not calculation_complete:
            gate_missing.append("successful deterministic calculation program")
        if not query_answer_alignment:
            gate_missing.append("query-aligned evidence or calculation result")
        if gate_missing and not blocking:
            issues = _deduplicate_issues(
                issues
                + [
                    ReadinessIssue(
                        code="EPISTEMIC_COMPLETENESS_FAILED",
                        severity=Severity.BLOCKING,
                        message="The final answer gate rejected an incomplete evidence state.",
                        missing_fields=gate_missing,
                        source="synthesis",
                    ).model_dump(mode="json")
                ]
            )
            blocking = True

        if query_contract.requires_human_confirmation and not blocking:
            preview_overlay = copy.deepcopy(state.get("scenario_overlay", {}))
            resolved_target = calculation_payload.get("resolved_target")
            preview_changes = preview_overlay.get("changes", [])
            if resolved_target and preview_changes:
                # Accept the calculated target only if it matches the correction target grammar.
                candidate = str(resolved_target)
                if re.fullmatch(r"material_transport_default:M-\d+-\d+", candidate):
                    preview_changes[0]["target"] = candidate
            preview, calls, preview_call = append_tool_call(
                calls,
                registry,
                self.name,
                "prepare_transaction_preview",
                {
                    "overlay": preview_overlay,
                    "query_contract": state.get("query_contract", {}),
                    "material_transport_defaults": state.get("retrieved_context", {})
                    .get("material_transport_defaults", {})
                    .get("rows", []),
                },
            )
            called_tools.append(preview_call.tool_name)
            if preview_call.status == ToolCallStatus.ERROR:
                issues = _deduplicate_issues(
                    issues + [_tool_failure_issue(self.name, preview_call.tool_name, preview_call.error, error_kind=preview_call.error_kind)]
                )
                blocking = True
            elif preview is None:
                issue = ReadinessIssue(
                    code="UNSUPPORTED_PERSISTENT_WRITE",
                    severity=Severity.BLOCKING,
                    message="This correction cannot be represented by the governed append-only transaction template.",
                    source="synthesis",
                ).model_dump(mode="json")
                issues = _deduplicate_issues(issues + [issue])
                blocking = True
            else:
                transaction_preview = preview
                requires_confirmation = True

        context = state.get("retrieved_context", {})
        calculation = state.get("calculation_results", {})
        retrieval_selection: Any = None
        partial = any(issue.get("code") == "PARTIAL_PROJECT_BASELINE" for issue in issues)
        status_result, calls, status_call = append_tool_call(
            calls,
            registry,
            self.name,
            "resolve_run_status",
            {
                "mode": mode.value,
                "blocking": blocking,
                "partial": partial,
                "transaction_preview_present": transaction_preview is not None,
                "requires_confirmation": query_contract.requires_human_confirmation,
            },
        )
        called_tools.append(status_call.tool_name)
        if status_call.status == ToolCallStatus.ERROR:
            issues = _deduplicate_issues(
                issues + [_tool_failure_issue(self.name, status_call.tool_name, status_call.error, error_kind=status_call.error_kind)]
            )
            status = RunStatus.BLOCKED
            blocking = True
        else:
            status = RunStatus(status_result["status"])
            requires_confirmation = bool(status_result["requires_human_confirmation"])

        if status == RunStatus.BLOCKED:
            missing = sorted({field for issue in issues for field in issue.get("missing_fields", [])})
            answer = "Execution stopped safely. " + "; ".join(
                issue["message"] for issue in issues if issue.get("severity") == Severity.BLOCKING.value
            )
            if missing:
                answer += f" Required clarification: {', '.join(missing)}."
        elif status == RunStatus.PENDING_CONFIRMATION:
            if calculation.get("tool") == "correction_impact_then_preview":
                answer = (
                    f"The current {calculation.get('material_name', calculation['material_code'])} "
                    f"({calculation['material_code']}) distance is "
                    f"{calculation['old_distance_km']:g} km; the proposed "
                    f"{calculation['new_distance_km']:g} km value gives an A4 "
                    f"distance multiplier of {calculation['distance_multiplier']:.6f}. "
                    "A persistent correction delta has been prepared but not written; "
                    "review and explicitly confirm its digest before commit."
                )
            else:
                answer = "A persistent correction delta has been prepared but not written. Review and explicitly confirm its digest before commit."
        elif calculation:
            language = "zh" if re.search(r"[\u3400-\u9fff]", state["query"]) else "en"
            rendered, calls, render_call = append_tool_call(
                calls,
                registry,
                self.name,
                "render_calculation_result",
                {"result": calculation, "language": language},
            )
            called_tools.append(render_call.tool_name)
            if render_call.status == ToolCallStatus.ERROR:
                issues = _deduplicate_issues(
                    issues + [_tool_failure_issue(self.name, render_call.tool_name, render_call.error, error_kind=render_call.error_kind)]
                )
                status = RunStatus.BLOCKED
                answer = f"Execution stopped safely. {render_call.error}"
            else:
                answer = rendered["answer"]
        else:
            language = "zh" if re.search(r"[\u3400-\u9fff]", state["query"]) else "en"
            rendered, calls, render_call = append_tool_call(
                calls,
                registry,
                self.name,
                "render_retrieval_result",
                {"query": state["query"], "context": context, "language": language},
            )
            called_tools.append(render_call.tool_name)
            if render_call.status == ToolCallStatus.ERROR:
                issues = _deduplicate_issues(
                    issues + [_tool_failure_issue(self.name, render_call.tool_name, render_call.error, error_kind=render_call.error_kind)]
                )
                status = RunStatus.BLOCKED
                answer = f"Execution stopped safely. {render_call.error}"
                retrieval_selection = None
            else:
                answer = rendered["answer"]
                retrieval_selection = rendered.get("selected_data")

        blocking_codes = sorted(
            {
                str(issue.get("code"))
                for issue in issues
                if issue.get("severity") == Severity.BLOCKING.value
            }
        )
        missing_variables = sorted(
            {
                str(field)
                for issue in issues
                for field in issue.get("missing_fields", [])
            }
        )
        clarification_request = (
            "Please provide or resolve: " + ", ".join(missing_variables) + "."
            if missing_variables
            else None
        )
        epistemic = EpistemicValidation(
            passed=not blocking_codes and status != RunStatus.BLOCKED,
            metadata_sufficient=metadata_sufficient,
            baseline_context_complete=baseline_context_complete,
            delta_complete=delta_complete,
            calculation_complete=calculation_complete,
            query_answer_alignment=query_answer_alignment,
            blocking_issue_codes=blocking_codes,
            missing_variables=missing_variables,
            clarification_request=clarification_request,
            write_confirmation_required=(
                mode == QueryMode.KNOWLEDGE_CORRECTION
                and status == RunStatus.PENDING_CONFIRMATION
            ),
        )
        shared_space = _update_shared_space(
            state,
            analytics_and_provenance={
                "epistemic_validation": epistemic.model_dump(mode="json"),
                "final_status": status.value,
                "final_tool_call_count": len(calls),
                "ontology_mode": metadata_payload.get("ontology_mode", "full"),
            },
        )
        event_provenance = _provenance(
            state,
            self.name,
            "applied epistemic checks and assembled the final report",
            details={
                "status": status.value,
                "requires_human_confirmation": requires_confirmation,
                "epistemic_passed": epistemic.passed,
                "ontology_mode": metadata_payload.get("ontology_mode", "full"),
                "blocking_issue_codes": blocking_codes,
                "tools": called_tools,
                "tool_call_count": len(calls),
            },
        )
        report = FinalReport(
            status=status,
            mode=mode,
            answer=answer,
            data={
                "retrieved_context": context,
                "metadata_retrieval": metadata_payload,
                "composite_data_context": composite_payload,
                "scenario_overlay": state.get("scenario_overlay", {}),
                "delta_graph_specification": delta_payload,
                "calculation_results": calculation,
                "calculation_program": calculation_program_payload,
                "epistemic_validation": epistemic.model_dump(mode="json"),
                "shared_global_space": shared_space,
                "cq_matches": state.get("cq_matches", []),
                "entity_pointers": state.get("entity_pointers", []),
                "ontology_context": state.get("ontology_context", {}),
                "planning_candidates": state.get("planning_candidates", []),
                "selected_plan_id": state.get("selected_plan_id"),
                "subontology": context.get("subontology", {}),
                "retrieval_selection": retrieval_selection,
                "tool_calls": calls,
            },
            issues=[ReadinessIssue.model_validate(issue) for issue in issues],
            execution_process=[item["action"] for item in event_provenance],
            provenance=[ProvenanceEvent.model_validate(item) for item in event_provenance],
            recommendations=self._recommendations(status, issues),
            requires_human_confirmation=requires_confirmation,
            transaction_preview=transaction_preview,
        )
        return {
            "issues": issues,
            "provenance": event_provenance,
            "tool_calls": calls,
            "epistemic_validation": epistemic.model_dump(mode="json"),
            "shared_global_space": shared_space,
            "final_report": report.model_dump(mode="json"),
            "metrics": _metrics(
                state,
                self.name,
                started,
                status=status.value,
                tools_called=len(called_tools),
            ),
        }

    @staticmethod
    def _recommendations(status: RunStatus, issues: list[dict[str, Any]]) -> list[str]:
        recommendations: list[str] = []
        codes = {issue.get("code") for issue in issues}
        if "PARTIAL_PROJECT_BASELINE" in codes:
            recommendations.append("Report RC4 values as partial design-stage results, not as the manuscript's final station total.")
        if "ENCLOSURE_NOT_IN_CARBON_BASELINE" in codes:
            recommendations.append(
                "Complete and validate enclosure CPD/EFD mappings before reporting a combined two-model carbon total."
            )
        if "MISSING_A4_PARAMETERS" in codes:
            recommendations.append("Provide transported mass, distance, mode-specific factor and compatible units for A4.")
        if "MISSING_SUBSTITUTION_PARAMETERS" in codes:
            recommendations.append("Provide provenance-backed, unit-compatible baseline and alternative material factors.")
        if status == RunStatus.PENDING_CONFIRMATION:
            recommendations.append("Review the target, predicate, value, unit and digest before committing the delta graph.")
        return recommendations


def build_agents(services: AgentServices) -> dict[str, Any]:
    registry = services.tool_registry()
    return {
        AgentName.ORCHESTRATOR.value: OrchestratorAgent(services.planning_reasoner, registry),
        AgentName.METADATA.value: MetadataAgent(services),
        AgentName.EXTRACTOR.value: ExtractorAgent(services),
        AgentName.READER.value: ReaderAgent(services),
        AgentName.CALCULATOR.value: CalculatorAgent(services),
        AgentName.SYNTHESIS.value: SynthesisAgent(services),
    }
