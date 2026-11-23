from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .calculations import (
    aggregate_named_values,
    amortise_reusable_resource,
    apply_named_deltas,
    calculate_shares,
    calculate_ratio,
    carbon_intensity,
    compare_named_values,
    compare_to_reference,
    equipment_emission,
    evaluate_threshold,
    factor_reduction,
    material_emission,
    material_substitution,
    one_at_a_time_sensitivity,
    project_foundation_geometry,
    rank_named_values,
    transport_emission,
)
from .config import ProjectConfig
from .cq import CQCatalogue, map_entity_mentions
from .entity_resolution import entity_label_tokens
from .ontology_profiles import ontology_profiles
from .metadata_pipeline import MetadataPipeline, build_metadata_pipeline
from .mas_program import compile_mas_calculation_program
from .models import (
    AgentName,
    PlanCandidate,
    QueryMode,
    RunStatus,
    ScenarioOverlay,
    ToolCallRecord,
    ToolCallStatus,
)
from .semantic_models import (
    BackboneVerification,
    DeltaGraphSpecification,
    DeltaOverride,
    OntologyEntityCandidate,
    OntologyRelation,
    QueryInterpretation,
    QueryContract,
    ReferenceDemand,
    ResolvedSemanticPath,
)
from .query_contract import parse_query_contract
from .scenarios import extract_overlay
from .reference_retrieval import ReferenceDatabaseRetriever
from .semantic_models import ReaderRetrievalPlan, RetrievalStep
from .transactions import TransactionService
from .util import sha256_file, stable_digest


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Distinguish tool implementation errors from rejected domain inputs.
_CRASH_EXCEPTIONS = (
    NameError,
    UnboundLocalError,
    AttributeError,
    TypeError,
    RecursionError,
    NotImplementedError,
)


class ExecutionPlanArguments(ToolArguments):
    mode: QueryMode
    analysis_required: bool = False
    query: str = Field(min_length=1)
    query_contract: dict[str, Any] = Field(default_factory=dict)


class QueryContractArguments(ToolArguments):
    query: str = Field(min_length=1)
    mode: QueryMode
    advisory_task_family: str | None = None
    advisory_operation_hints: list[str] = Field(default_factory=list)
    advisory_semantic_anchors: list[str] = Field(default_factory=list)
    advisory_contract: dict[str, Any] = Field(default_factory=dict)


class QueryArguments(ToolArguments):
    query: str = Field(min_length=1)


class CQSearchArguments(QueryArguments):
    limit: int = Field(default=5, ge=1, le=20)


class OntologySelectionArguments(QueryArguments):
    mode: QueryMode
    cq_ids: list[str] = Field(default_factory=list)
    preferred_profile_ids: list[str] = Field(default_factory=list)


class SemanticParseArguments(QueryArguments):
    mode: QueryMode
    anchor_terms: list[str] = Field(default_factory=list)
    cq_ids: list[str] = Field(default_factory=list)


class HybridOntologyRetrievalArguments(ToolArguments):
    interpretation: dict[str, Any]
    phrases: list[str] = Field(min_length=1)


class OntologyExpansionArguments(ToolArguments):
    interpretation: dict[str, Any]
    seeds: list[dict[str, Any]] = Field(default_factory=list)


class OntologyPathArguments(ToolArguments):
    interpretation: dict[str, Any]
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class BackboneVerificationArguments(OntologyPathArguments):
    relations: list[dict[str, Any]] = Field(default_factory=list)
    paths: list[dict[str, Any]] = Field(default_factory=list)


class MetadataSupplementArguments(ToolArguments):
    interpretation: dict[str, Any]
    verification: dict[str, Any]
    previous_phrases: list[str] = Field(default_factory=list)


class EmptyArguments(ToolArguments):
    pass


class TopElementsArguments(ToolArguments):
    limit: int = Field(default=10, ge=1, le=100)
    stage: str | None = None


class ResourceBreakdownArguments(ToolArguments):
    term: str = Field(min_length=1)
    stage: str | None = None
    limit: int = Field(default=30, ge=1, le=250)


class WorkPackageBreakdownArguments(ToolArguments):
    term: str = ""
    limit: int = Field(default=30, ge=1, le=250)


class ElementTraceArguments(ToolArguments):
    element_uri: str = Field(min_length=1)
    limit: int = Field(default=250, ge=1, le=1000)


class SubontologyArguments(ToolArguments):
    profile_ids: list[str] = Field(min_length=1)
    query: str = Field(min_length=1)
    mode: QueryMode
    cq_ids: list[str] = Field(default_factory=list)


class ReferenceArtifactArguments(ToolArguments):
    dataset: Literal[
        "enclosure_assignments",
        "enclosure_cpd_efd",
        "foundation_cpd_efd",
        "a4_transport_legs",
        "material_transport_defaults",
        "vehicle_factors",
        "ifc_inventory",
    ]
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=100, ge=1, le=10000)
    sort_by: str | None = None
    descending: bool = False


class ReaderPlanArguments(QueryArguments):
    metadata_retrieval: dict[str, Any] = Field(default_factory=dict)
    cq_matches: list[dict[str, Any]] = Field(default_factory=list)
    delta_graph_specification: dict[str, Any] = Field(default_factory=dict)
    query_contract: dict[str, Any] = Field(default_factory=dict)


class CKGTemplateArguments(ToolArguments):
    template_id: str = Field(min_length=1)
    bindings: dict[str, Any] = Field(default_factory=dict)


class ReferenceLookupArguments(QueryArguments):
    categories: list[Literal["CPD", "EFD", "TRANSPORT"]] = Field(default_factory=list)
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=8, ge=1, le=50)
    lexical_only: bool = False


class ReferenceDemandResolutionArguments(ToolArguments):
    delta_graph: dict[str, Any]
    overlay: dict[str, Any]
    reference_searches: list[dict[str, Any]] = Field(default_factory=list)
    exact_artifacts: dict[str, Any] = Field(default_factory=dict)


class OverlayArguments(QueryArguments):
    persistent: bool = False
    semantic_contract: dict[str, Any] = Field(default_factory=dict)


class DeltaGraphArguments(QueryArguments):
    mode: QueryMode
    overlay: dict[str, Any]


class CalculationProgramCompileArguments(QueryArguments):
    mode: QueryMode
    query_contract: dict[str, Any]
    context: dict[str, Any]
    overlay: dict[str, Any] = Field(default_factory=dict)


class NamedValuesArguments(ToolArguments):
    values: dict[str, float]
    unit: str = "kgCO2e"


class ComparisonArguments(NamedValuesArguments):
    baseline_name: str | None = None


class RankingArguments(NamedValuesArguments):
    descending: bool = True
    limit: int | None = Field(default=None, ge=1)


class SensitivityArguments(ToolArguments):
    baseline_value: float
    baseline_parameter: float
    parameter_values: list[float] = Field(min_length=1)
    parameter_name: str = Field(min_length=1)
    result_unit: str = "kgCO2e"


class IntensityArguments(ToolArguments):
    emission_kgco2e: float
    denominator: float
    denominator_unit: str = Field(min_length=1)


class MaterialEmissionArguments(ToolArguments):
    work_quantity: float
    consumption_rate: float
    emission_factor: float
    result_unit: str = Field(default="kgCO2e", min_length=1)


class EquipmentEmissionArguments(ToolArguments):
    work_volume: float
    productivity_per_hour: float
    energy_per_hour: float
    energy_factor: float


class FactorReductionArguments(ToolArguments):
    baseline_emission: float
    reduction_fraction: float


class SubstitutionArguments(ToolArguments):
    baseline_emission: float
    substitution_fraction: float
    baseline_factor: float
    alternative_factor: float


class TransportArguments(ToolArguments):
    mass: float
    mass_unit: str
    distance_km: float
    factor_kgco2e_per_tkm: float


class ReuseArguments(ToolArguments):
    baseline_emission: float
    reuse_cycles: float


class NamedDeltaArguments(ToolArguments):
    baseline_value: float
    deltas: dict[str, float]
    unit: str = "kgCO2e"


class RatioArguments(ToolArguments):
    numerator: float
    denominator: float
    numerator_name: str = "numerator"
    denominator_name: str = "denominator"


class ThresholdArguments(ToolArguments):
    value: float
    threshold: float
    comparison: Literal[
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
    ] = "greater_than"


class FoundationGeometryArguments(ToolArguments):
    gross_envelope_m3: float
    plan_utilisation_factor: float
    backfill_ratio: float
    loose_volume_factor: float


class ReferenceComparisonArguments(ToolArguments):
    values: dict[str, float]
    reference_value: float
    reference_name: str = "governed_reference"
    tolerance: float = 0.0


class StatusArguments(ToolArguments):
    mode: QueryMode
    blocking: bool
    partial: bool = False
    transaction_preview_present: bool = False
    requires_confirmation: bool = False


class RenderArguments(ToolArguments):
    result: dict[str, Any]
    language: str = "en"


class RetrievalRenderArguments(QueryArguments):
    context: dict[str, Any]
    language: str = "en"


class TransactionPreviewArguments(ToolArguments):
    overlay: dict[str, Any]
    query_contract: dict[str, Any] = Field(default_factory=dict)
    material_transport_defaults: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    owner: AgentName
    description: str
    input_model: type[ToolArguments]
    handler: Callable[[ToolArguments], Any]
    output_contract: str


ONTOLOGY_REASONING_TOOL_NAMES = frozenset(
    {
        "search_competency_questions",
        "parse_semantic_query",
        "select_ontology_profiles",
        "hybrid_ontology_retrieval",
        "expand_related_ontology",
        "resolve_ontology_paths",
        "verify_query_backbone",
        "supplement_missing_concepts",
        "read_subontology_signature",
    }
)
"""Dedicated ontology reasoning tools unavailable in schema-only mode."""


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("tool names must be unique")

    def manifest(self, owner: AgentName | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "owner": spec.owner.value,
                "description": spec.description,
                "input_schema": spec.input_model.model_json_schema(),
                "output_contract": spec.output_contract,
            }
            for spec in sorted(self._specs.values(), key=lambda item: (item.owner.value, item.name))
            if owner is None or spec.owner == owner
        ]

    def names(self) -> set[str]:
        """Return the registered tool names."""

        return set(self._specs)

    def restricted(
        self,
        *,
        allowed_names: set[str] | frozenset[str] | None = None,
        excluded_names: set[str] | frozenset[str] = frozenset(),
    ) -> "ToolRegistry":
        """Return a registry with the specified tools enabled or excluded."""

        selected = [
            spec
            for name, spec in self._specs.items()
            if (allowed_names is None or name in allowed_names)
            and name not in excluded_names
        ]
        return ToolRegistry(selected)

    def with_wrapped(
        self,
        name: str,
        wrapper: Callable[[Callable[[Any], Any]], Callable[[Any], Any]],
    ) -> "ToolRegistry":
        """Wrap a tool's handler without changing its name, owner, or schema."""

        import dataclasses

        if name not in self._specs:
            raise KeyError(f"unknown tool to wrap: {name}")
        selected = [
            dataclasses.replace(spec, handler=wrapper(spec.handler))
            if spec.name == name
            else spec
            for spec in self._specs.values()
        ]
        return ToolRegistry(selected)

    def invoke(
        self,
        agent: AgentName,
        tool_name: str,
        arguments: dict[str, Any],
        sequence: int,
    ) -> ToolCallRecord:
        started = time.perf_counter()
        spec = self._specs.get(tool_name)
        result: Any = None
        error: str | None = None
        error_kind: Literal["domain", "crash"] | None = None
        if spec is None:
            error = f"unknown tool: {tool_name}"
        elif spec.owner != agent:
            error = f"{agent.value} is not authorised to call {tool_name}; owner is {spec.owner.value}"
        else:
            try:
                validated = spec.input_model.model_validate(arguments)
                result = spec.handler(validated)
                if isinstance(result, BaseModel):
                    result = result.model_dump(mode="json")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                error_kind = "crash" if isinstance(exc, _CRASH_EXCEPTIONS) else "domain"
        duration = round(time.perf_counter() - started, 6)
        return ToolCallRecord(
            call_id=f"call-{sequence:03d}-{agent.value}-{tool_name}",
            sequence=sequence,
            agent=agent,
            tool_name=tool_name,
            arguments=arguments,
            status=ToolCallStatus.ERROR if error else ToolCallStatus.SUCCESS,
            result=result,
            result_digest=stable_digest(result) if error is None else None,
            error=error,
            error_kind=error_kind,
            duration_seconds=duration,
        )


def _execution_plan(arguments: ExecutionPlanArguments) -> dict[str, Any]:
    mode = arguments.mode
    contract = QueryContract.model_validate(
        arguments.query_contract
        or parse_query_contract(arguments.query, mode).model_dump(mode="json")
    )
    if mode == QueryMode.INFORMATION_RETRIEVAL:
        required = [AgentName.METADATA, AgentName.READER]
        if arguments.analysis_required:
            required.append(AgentName.CALCULATOR)
        required.append(AgentName.SYNTHESIS)
        if arguments.analysis_required:
            raw_candidates = [
                ("PLAN-IR-CALC-ONTOLOGY-FIRST", "ontology-guided read then deterministic calculation", required),
                ("PLAN-IR-CALC-PREMATURE", "calculate before governed facts are available", [AgentName.METADATA, AgentName.CALCULATOR, AgentName.READER, AgentName.SYNTHESIS]),
                ("PLAN-IR-CALC-OMITTED", "retrieval-only shortcut", [AgentName.METADATA, AgentName.READER, AgentName.SYNTHESIS]),
            ]
        else:
            raw_candidates = [
                ("PLAN-IR-ONTOLOGY-FIRST", "ontology-guided governed retrieval", required),
                ("PLAN-IR-DIRECT-READ", "direct fact lookup without metadata grounding", [AgentName.READER, AgentName.SYNTHESIS]),
                ("PLAN-IR-METADATA-ONLY", "metadata-only answer without governed facts", [AgentName.METADATA, AgentName.SYNTHESIS]),
            ]
    elif mode == QueryMode.KNOWLEDGE_CORRECTION:
        if contract.correction_impact_preview:
            required = [
                AgentName.METADATA,
                AgentName.EXTRACTOR,
                AgentName.READER,
                AgentName.CALCULATOR,
                AgentName.SYNTHESIS,
            ]
            raw_candidates = [
                (
                    "PLAN-KC-IMPACT-DAG",
                    "validate correction, retrieve current evidence, calculate impact, then prepare preview",
                    required,
                ),
                (
                    "PLAN-KC-IMPACT-EVIDENCE-FIRST",
                    "retrieve evidence before extracting and calculating the correction",
                    [
                        AgentName.METADATA,
                        AgentName.READER,
                        AgentName.EXTRACTOR,
                        AgentName.CALCULATOR,
                        AgentName.SYNTHESIS,
                    ],
                ),
                (
                    "PLAN-KC-IMPACT-INCOMPLETE",
                    "prepare a correction without an impact calculation",
                    [AgentName.METADATA, AgentName.EXTRACTOR, AgentName.READER, AgentName.SYNTHESIS],
                ),
            ]
        else:
            required = [AgentName.METADATA, AgentName.EXTRACTOR, AgentName.READER, AgentName.SYNTHESIS]
            raw_candidates = [
                ("PLAN-KC-OVERLAY-FIRST", "validate correction parameters before evidence read", required),
                ("PLAN-KC-EVIDENCE-FIRST", "read evidence before extracting the correction", [AgentName.METADATA, AgentName.READER, AgentName.EXTRACTOR, AgentName.SYNTHESIS]),
                ("PLAN-KC-DIRECT-WRITE", "direct write shortcut without metadata or evidence", [AgentName.EXTRACTOR, AgentName.SYNTHESIS]),
            ]
    else:
        required = [AgentName.METADATA, AgentName.EXTRACTOR, AgentName.READER, AgentName.CALCULATOR, AgentName.SYNTHESIS]
        if contract.compound:
            raw_candidates = [
                (
                    "PLAN-SA-COMPOUND-DAG",
                    "validate every overlay branch, join governed baselines, execute dependencies, then verify",
                    required,
                ),
                (
                    "PLAN-SA-COMPOUND-BASELINE-FIRST",
                    "retrieve all baselines before extracting the compound overlay",
                    [AgentName.METADATA, AgentName.READER, AgentName.EXTRACTOR, AgentName.CALCULATOR, AgentName.SYNTHESIS],
                ),
                (
                    "PLAN-SA-COMPOUND-INCOMPLETE",
                    "collapse the compound request into one calculation",
                    [AgentName.METADATA, AgentName.EXTRACTOR, AgentName.READER, AgentName.SYNTHESIS],
                ),
            ]
        else:
            raw_candidates = [
                ("PLAN-SA-OVERLAY-FIRST", "validate scenario parameters before baseline retrieval", required),
                ("PLAN-SA-BASELINE-FIRST", "retrieve the baseline before parsing scenario parameters", [AgentName.METADATA, AgentName.READER, AgentName.EXTRACTOR, AgentName.CALCULATOR, AgentName.SYNTHESIS]),
                ("PLAN-SA-PREMATURE-CALC", "calculate before the scenario overlay is validated", [AgentName.METADATA, AgentName.READER, AgentName.CALCULATOR, AgentName.EXTRACTOR, AgentName.SYNTHESIS]),
            ]

    required_set = set(required)
    candidates: list[PlanCandidate] = []
    for plan_id, strategy, agents in raw_candidates:
        violations: list[str] = []
        positions = {agent: index for index, agent in enumerate(agents)}
        missing = required_set - set(agents)
        if missing:
            violations.append("missing required roles: " + ", ".join(sorted(agent.value for agent in missing)))
        if agents[-1:] != [AgentName.SYNTHESIS]:
            violations.append("synthesis must be the terminal role")
        if AgentName.METADATA in positions and AgentName.READER in positions and positions[AgentName.METADATA] > positions[AgentName.READER]:
            violations.append("metadata must precede reader for query-specific ontology selection")
        if AgentName.CALCULATOR in positions and AgentName.READER in positions and positions[AgentName.CALCULATOR] < positions[AgentName.READER]:
            violations.append("calculator must follow governed fact retrieval")
        if (
            (mode == QueryMode.STRATEGIC_ANALYSIS or contract.correction_impact_preview)
            and AgentName.CALCULATOR in positions
            and AgentName.EXTRACTOR in positions
            and positions[AgentName.CALCULATOR] < positions[AgentName.EXTRACTOR]
        ):
            violations.append("scenario overlay must be validated before calculation")

        coverage = len(required_set & set(agents)) / len(required_set)
        dependency = 1.0 if not [item for item in violations if "must" in item] else 0.0
        ontology_grounding = 1.0 if (
            AgentName.METADATA in positions
            and AgentName.READER in positions
            and positions[AgentName.METADATA] < positions[AgentName.READER]
        ) else 0.0
        efficiency = min(1.0, len(required) / len(agents))
        early_validation = 1.0
        if mode in {QueryMode.STRATEGIC_ANALYSIS, QueryMode.KNOWLEDGE_CORRECTION}:
            early_validation = (
                1.0
                if AgentName.EXTRACTOR in positions
                and AgentName.READER in positions
                and positions[AgentName.EXTRACTOR] < positions[AgentName.READER]
                else 0.8
            )
        total = (
            0.38 * coverage
            + 0.27 * dependency
            + 0.20 * ontology_grounding
            + 0.05 * efficiency
            + 0.10 * early_validation
        )
        focus: dict[str, list[str]] = {
            AgentName.METADATA.value: ["CQ", "ontology profile", "semantic path"],
            AgentName.READER.value: ["CKG", "reference artifact", "subontology signature"],
            AgentName.EXTRACTOR.value: ["typed overlay", "parameter completeness"],
            AgentName.CALCULATOR.value: ["schema-validated deterministic function"],
            AgentName.SYNTHESIS.value: ["status", "grounded answer", "tool trace"],
        }
        focus[AgentName.ORCHESTRATOR.value] = list(contract.operation_ids)
        candidates.append(
            PlanCandidate(
                plan_id=plan_id,
                strategy=strategy,
                agents=agents,
                tool_focus={agent.value: focus[agent.value] for agent in agents},
                valid=not violations,
                constraint_violations=violations,
                score_components={
                    "required_role_coverage": coverage,
                    "dependency_validity": dependency,
                    "ontology_grounding": ontology_grounding,
                    "execution_efficiency": efficiency,
                    "early_parameter_validation": early_validation,
                },
                total_score=round(total, 6),
            )
        )
    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    if not valid_candidates:
        raise ValueError("no execution-plan candidate satisfies the hard constraints")
    selected = sorted(valid_candidates, key=lambda item: (-item.total_score, item.plan_id))[0]
    candidates = [candidate.model_copy(update={"selected": candidate.plan_id == selected.plan_id}) for candidate in candidates]
    return {
        "selected_plan_id": selected.plan_id,
        "agents": [agent.value for agent in selected.agents],
        "analysis_required": arguments.analysis_required,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "selection_policy": "hard constraints first; weighted deterministic score second",
        "query_contract": contract.model_dump(mode="json"),
    }


def _select_ontology_profiles(arguments: OntologySelectionArguments) -> dict[str, Any]:
    query = arguments.query.lower().replace("–", "-").replace("—", "-")
    cq_ids = set(arguments.cq_ids)
    profiles: list[str] = []

    def add(profile_id: str) -> None:
        if profile_id not in profiles:
            profiles.append(profile_id)

    if arguments.preferred_profile_ids:
        available = set(ontology_profiles())
        unknown = sorted(set(arguments.preferred_profile_ids) - available)
        if unknown:
            raise ValueError(f"unknown planner ontology profiles: {unknown}")
        for profile_id in arguments.preferred_profile_ids:
            add(profile_id)
        if arguments.mode in {QueryMode.STRATEGIC_ANALYSIS, QueryMode.KNOWLEDGE_CORRECTION}:
            add("ONT-SCENARIO")
        if arguments.mode == QueryMode.KNOWLEDGE_CORRECTION:
            add("ONT-SCOPE-QUALITY")
        return {
            "profile_ids": profiles,
            "selection_basis": "validated LLM semantic contract plus invariant safety closure",
            "governing_cq_ids": arguments.cq_ids,
        }

    def has(*terms: str) -> bool:
        return any(term in query for term in terms)

    out_of_scope = bool(re.search(r"\b(?:b1|c1|module\s*d)\b", query)) or has(
        "operational-use", "deconstruction", "beyond the system boundary", "运营使用", "拆除阶段", "边界外", "d模块"
    )
    included_stages = has(
        "lifecycle stages are included", "included in the active assessment", "核算包含", "包含哪些生命周期"
    )
    ontology_query = has(
        "ontology term",
        "ontology content",
        "subontology",
        "semantic path",
        "query-specific ontology",
        "ontology术语",
        "ontology 术语",
        "本体术语",
    )
    transport = has("a4", "transport", "haul", "vehicle", "t-km", "运输", "运距", "载具", "吨公里")
    foundation = has("foundation pit", "foundation-pit", "foundation haul", "foundation-haul", "earthwork", "excavation", "backfill", "基坑", "土方", "开挖", "回填")
    enclosure = has("enclosure", "diaphragm", "sectionsteelstrut", "围护", "地下连续墙", "地连墙", "适用围护构件", "映射率")
    main_structure = has("main structure", "main-structure", "主体结构")
    trace = has("trace", "provenance", "audit chain", "resource breakdown", "追溯", "溯源", "审计链", "资源明细", "完整链路")
    # A provenance request alone does not require the full trace.
    if "source provenance" in query and not has(
        "trace", "audit chain", "resource breakdown", "emission provenance", "factor provenance"
    ):
        trace = False
    public_context = has(
        "public geometry",
        "building area",
        "construction zones",
        "ground conditions",
        "production peaks",
        "station form",
    )
    process_resource_detail = has(
        "equipment emission",
        "material emission",
        "resource breakdown",
        "work package",
        "formwork",
        "reuse",
        "concrete strength",
        "supplier",
        "construction-process",
    )
    multi_scope_query = all((main_structure, enclosure, foundation))
    multi_scope_transport = transport and multi_scope_query
    project_stage_aggregate = has("project", "项目") and has("carbon", "emission", "碳", "排放") and has(
        "total", "a1-a3", "a4", "a5", "lifecycle stage", "合计", "总量", "阶段"
    )
    system_carbon_aggregate = (
        main_structure or enclosure or foundation
    ) and (
        has("total carbon", "total emission")
        or all(has(stage) for stage in ("a1-a3", "a4", "a5"))
    )
    ifc_multi_scope = has("ifc") and has("product", "products") and main_structure and enclosure
    foundation_detail = foundation and has(
        "excavation",
        "backfill",
        "earthwork",
        "volume",
        "haul",
        "intensity",
        "cpd",
        "work package",
        "resource flow",
        "sensitivity result",
        "declared parameters",
        "体积",
        "方量",
        "外运",
        "运距",
    )
    enclosure_detail = enclosure and has(
        "mapping",
        "applicable",
        "diaphragm",
        "sectionsteelstrut",
        "variant",
        "cpd",
        "work package",
        "resource flow",
        "element type",
        "quantity",
    )
    material_transport_detail = transport and (
        has("a4", "vehicle", "t-km", "material transport", "transport leg")
        or bool(re.search(r"(?<![A-Za-z0-9])M-\d+-\d+(?![A-Za-z0-9])", arguments.query, re.I))
    )

    ontology_signature_query = has(
        "which ontology terms are required",
        "which ontology terms are needed",
    ) or (
        "ontology" in query
        and "subontology" not in query
        and "query-specific" not in query
        and bool(cq_ids & {"CQ-C03", "CQ-C08", "CQ-C06", "CQ-C02"})
    )
    if ontology_signature_query:
        # Use one profile for signature queries; compose profiles for explicit subontology requests.
        if transport:
            add("ONT-A4-TRANSPORT")
        elif trace:
            add("ONT-SEPRC-PROVENANCE")
        elif (
            "CQ-C08" in cq_ids
            or has("main, enclosure", "by main, enclosure", "engineering system")
        ):
            add("ONT-SYSTEM-CARBON")
        else:
            add("ONT-PROJECT-CARBON")
    elif ontology_query:
        scenario_ontology = has(
            "scenario",
            "transient",
            "simulate",
            "simulation",
            "sensitivity",
            "override",
            "without writing",
        )
        provenance_ontology = trace or has(
            "emission provenance",
            "factor provenance",
            "auditable",
            "system-element-process-resource-carbon",
            "through cpd",
            "resource flow",
            "resource flows",
        )
        detailed_enclosure = enclosure and has(
            "ifc", "cpd", "efd", "work package", "resource flow", "lifecycle emission"
        )
        detailed_foundation = foundation and has(
            "geometry", "earthwork", "excavation", "backfill", "resource flow", "haul"
        )
        project_context = has(
            "project lifecycle",
            "project total",
            "lifecycle total",
            "project-stage",
            "project stage",
            "project carbon",
            "project baseline",
            "active a1-a5",
        )
        if out_of_scope or has("boundary", "boundary-checked", "assessment boundary"):
            add("ONT-SCOPE-QUALITY")
        if scenario_ontology:
            add("ONT-SCENARIO")
        if detailed_enclosure:
            add("ONT-ENCLOSURE-CPD")
        if detailed_foundation:
            add("ONT-FOUNDATION-EARTHWORK")
        if transport:
            add("ONT-A4-TRANSPORT")
        if provenance_ontology:
            add("ONT-SEPRC-PROVENANCE")
        if project_context or out_of_scope or has("boundary-checked", "boundary checked"):
            add("ONT-PROJECT-CARBON")
        if multi_scope_query or has("non-overlapping main", "scope totals"):
            add("ONT-SYSTEM-CARBON")
        if not profiles:
            add("ONT-PROJECT-CARBON")
    elif out_of_scope:
        add("ONT-SCOPE-QUALITY")
    elif included_stages:
        add("ONT-PROJECT-CARBON")
        add("ONT-SCOPE-QUALITY")
    elif has("applicability scope", "适用范围"):
        add("ONT-PROJECT-CARBON")
    elif has("formula", "公式"):
        add("ONT-PROJECT-CARBON")
        add("ONT-FOUNDATION-EARTHWORK")
        add("ONT-A4-TRANSPORT")
    elif arguments.mode in {QueryMode.STRATEGIC_ANALYSIS, QueryMode.KNOWLEDGE_CORRECTION}:
        add("ONT-SCENARIO")
        if transport or "CQ-X02" in cq_ids:
            add("ONT-A4-TRANSPORT")
        if cq_ids & {"CQ-X01", "CQ-X03", "CQ-X05"} or process_resource_detail:
            add("ONT-SEPRC-PROVENANCE")
        if cq_ids & {"CQ-X08", "CQ-Q04"}:
            add("ONT-SCOPE-QUALITY")
    elif public_context:
        add("ONT-PROJECT-CARBON")
    elif project_stage_aggregate:
        add("ONT-PROJECT-CARBON")
    elif multi_scope_transport:
        add("ONT-A4-TRANSPORT")
    elif multi_scope_query:
        add("ONT-SYSTEM-CARBON")
    elif ifc_multi_scope:
        add("ONT-SYSTEM-CARBON")
    elif main_structure and process_resource_detail:
        add("ONT-SEPRC-PROVENANCE")
    elif system_carbon_aggregate:
        add("ONT-SYSTEM-CARBON")
    elif material_transport_detail:
        add("ONT-A4-TRANSPORT")
    elif foundation_detail:
        add("ONT-FOUNDATION-EARTHWORK")
        if "CQ-C06" in cq_ids or has("haul distance", "运距"):
            add("ONT-A4-TRANSPORT")
    elif enclosure_detail:
        add("ONT-ENCLOSURE-CPD")
    elif transport:
        add("ONT-A4-TRANSPORT")
    elif cq_ids & {"CQ-C09", "CQ-C02"} or trace or (
        main_structure and cq_ids >= {"CQ-P04", "CQ-C03"}
    ) or (main_structure and cq_ids >= {"CQ-I02", "CQ-P03"}):
        add("ONT-SEPRC-PROVENANCE")
    elif cq_ids & {"CQ-S01", "CQ-S03"} or ("CQ-C08" in cq_ids and "CQ-I03" not in cq_ids):
        add("ONT-SYSTEM-CARBON")
    elif "CQ-C06" in cq_ids and "CQ-C05" not in cq_ids:
        add("ONT-A4-TRANSPORT")
    elif "CQ-C05" in cq_ids:
        add("ONT-FOUNDATION-EARTHWORK")
        if "CQ-C06" in cq_ids:
            add("ONT-A4-TRANSPORT")
    elif cq_ids & {"CQ-I02", "CQ-I03"}:
        add("ONT-ENCLOSURE-CPD")
    else:
        add("ONT-PROJECT-CARBON")
    return {
        "profile_ids": profiles,
        "selection_basis": "governed query/CQ profile router",
        "governing_cq_ids": arguments.cq_ids,
    }


def _subontology_signature(arguments: SubontologyArguments) -> dict[str, Any]:
    available = ontology_profiles()
    unknown = sorted(set(arguments.profile_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown ontology profiles: {unknown}")
    selected = [available[profile_id] for profile_id in arguments.profile_ids]
    query = arguments.query.lower().replace("–", "-").replace("—", "-")
    profile_ids = set(arguments.profile_ids)

    def has(*terms: str) -> bool:
        return any(term in query for term in terms)

    ontology_query = has(
        "ontology term",
        "ontology content",
        "subontology",
        "semantic path",
        "query-specific ontology",
        "ontology术语",
        "ontology 术语",
        "本体术语",
    )
    out_of_scope = bool(re.search(r"(?<![a-z0-9])(?:b1|c1|d)(?![a-z0-9])", query)) or has(
        "operational-use", "deconstruction", "beyond the system boundary", "运营使用", "拆除阶段", "边界外", "d模块"
    )
    missing = has("without", "missing", "omit", "不提供", "缺少", "省略")
    query_specific_design = has(
        "query-specific subontology",
        "query-specific ontology",
        "subontology",
        "semantic paths",
    )
    if query_specific_design:
        # For a schema question, cite the schema and CQ evidence.
        data_sources = ["ONTOLOGY_SCHEMA", "CQ_CATALOGUE", "REFERENCE_ABOX"]
    elif ontology_query:
        if "ONT-SEPRC-PROVENANCE" in profile_ids:
            data_sources = ["CKG_MANIFEST", "ONTOLOGY_SCHEMA", "REFERENCE_ABOX"]
        elif "ONT-A4-TRANSPORT" in profile_ids:
            data_sources = ["A4_TRANSPORT_LEGS", "ONTOLOGY_SCHEMA", "REFERENCE_ABOX", "TRANSPORT_DEFAULTS"]
        elif "ONT-SYSTEM-CARBON" in profile_ids:
            data_sources = ["CKG_MANIFEST", "ONTOLOGY_SCHEMA", "REFERENCE_ABOX", "REFERENCE_SUMMARY"]
        else:
            data_sources = ["ONTOLOGY_SCHEMA", "REFERENCE_ABOX", "REFERENCE_SUMMARY"]
    elif has("formula", "公式"):
        data_sources = ["REFERENCE_SUMMARY"]
    elif out_of_scope or (missing and arguments.mode != QueryMode.INFORMATION_RETRIEVAL) or arguments.mode == QueryMode.KNOWLEDGE_CORRECTION:
        data_sources = ["ONTOLOGY_SCHEMA", "PROJECT_CONFIG"]
    elif arguments.mode == QueryMode.STRATEGIC_ANALYSIS:
        data_sources = (
            ["ONTOLOGY_SCHEMA"]
            if "ONT-A4-TRANSPORT" in profile_ids and "ONT-SEPRC-PROVENANCE" not in profile_ids
            else ["CKG_INDEX", "ONTOLOGY_SCHEMA"]
        )
    elif "ONT-SEPRC-PROVENANCE" in profile_ids:
        data_sources = ["CKG_INDEX", "CKG_MANIFEST", "ONTOLOGY_SCHEMA"]
    elif has("ifc") and has("product", "products", "产品"):
        data_sources = ["IFC_INVENTORY"]
    elif "ONT-FOUNDATION-EARTHWORK" in profile_ids:
        if "ONT-A4-TRANSPORT" in profile_ids and has("haul", "运距", "外运"):
            data_sources = ["REFERENCE_SUMMARY", "REFERENCE_PROFILE", "A4_TRANSPORT_LEGS"]
        else:
            data_sources = ["REFERENCE_SUMMARY", "FOUNDATION_CPD_EFD", "REFERENCE_PROFILE"]
    elif "ONT-ENCLOSURE-CPD" in profile_ids:
        data_sources = ["REFERENCE_SUMMARY", "ENCLOSURE_ASSIGNMENTS", "ENCLOSURE_CPD_EFD"]
    elif "ONT-A4-TRANSPORT" in profile_ids:
        data_sources = ["REFERENCE_SUMMARY", "A4_TRANSPORT_LEGS", "TRANSPORT_DEFAULTS", "VEHICLE_FACTORS"]
    elif "ONT-SCOPE-QUALITY" in profile_ids:
        data_sources = (
            ["REFERENCE_SUMMARY", "PROJECT_CONFIG"]
            if has("included", "包含", "核算")
            else ["PROJECT_CONFIG", "ONTOLOGY_SCHEMA"]
        )
    elif "ONT-SYSTEM-CARBON" in profile_ids:
        data_sources = ["REFERENCE_SUMMARY", "REFERENCE_ABOX"]
    elif has("applicability scope", "适用范围", "formula", "公式"):
        data_sources = ["REFERENCE_SUMMARY"]
    else:
        data_sources = ["REFERENCE_SUMMARY", "REFERENCE_ABOX"]
    return {
        "profile_ids": arguments.profile_ids,
        "ontology_terms": sorted({term for profile in selected for term in profile.required_terms}),
        "semantic_path_ids": sorted(
            {path["path_id"] for profile in selected for path in profile.semantic_paths}
        ),
        "semantic_paths": [
            path
            for profile in selected
            for path in profile.semantic_paths
        ],
        "data_source_ids": sorted(data_sources),
        "profile_source_modules": sorted({source for profile in selected for source in profile.source_modules}),
        "subontology_files": [
            f"benchmark/ground_truth/subontology/{profile.profile_id.lower()}.ttl"
            for profile in selected
        ],
    }


def _resolve_status(arguments: StatusArguments) -> dict[str, Any]:
    if arguments.blocking:
        status = RunStatus.BLOCKED
    elif arguments.requires_confirmation and arguments.transaction_preview_present:
        status = RunStatus.PENDING_CONFIRMATION
    elif arguments.requires_confirmation:
        status = RunStatus.BLOCKED
    elif arguments.partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.COMPLETE
    return {
        "status": status.value,
        "requires_human_confirmation": status == RunStatus.PENDING_CONFIRMATION,
    }


def _render_result(arguments: RenderArguments) -> dict[str, Any]:
    result = arguments.result
    tool = result.get("tool")
    zh = arguments.language == "zh"
    if tool == "aggregate_named_values":
        items = ", ".join(f"{name}: {value:,.3f}" for name, value in result["values"].items())
        answer = (
            f"分项为 {items} {result['unit']}；聚合合计为 {result['total']:,.3f} {result['unit']}。"
            if zh
            else f"Items are {items} {result['unit']}; the aggregated total is {result['total']:,.3f} {result['unit']}."
        )
    elif tool == "compare_named_values":
        values = ", ".join(f"{name}: {value:,.3f}" for name, value in result["values"].items())
        maximum = result["maximum"]
        minimum = result["minimum"]
        answer = (
            f"比较结果：{values} {result['unit']}；最高为{maximum['name']}（{maximum['value']:,.3f}），最低为{minimum['name']}（{minimum['value']:,.3f}），极差为 {result['range']:,.3f} {result['unit']}。"
            if zh
            else f"Comparison: {values} {result['unit']}. Highest is {maximum['name']} ({maximum['value']:,.3f}); lowest is {minimum['name']} ({minimum['value']:,.3f}); range is {result['range']:,.3f} {result['unit']}."
        )
    elif tool == "calculate_shares":
        shares = ", ".join(f"{item['name']}: {item['percentage']:.3f}%" for item in result["shares"])
        answer = f"占比为：{shares}。" if zh else f"Shares of the total are: {shares}."
    elif tool == "rank_named_values":
        ranked = ", ".join(f"{item['rank']}. {item['name']} ({item['value']:,.3f})" for item in result["ranked"])
        answer = f"排序结果：{ranked} {result['unit']}。" if zh else f"Ranking: {ranked} {result['unit']}."
    elif tool == "one_at_a_time_sensitivity":
        points = ", ".join(
            f"{item['parameter_value']}: {item['result']:,.3f} ({item['percentage_change']:+.3f}%)"
            for item in result["points"]
        )
        answer = (
            f"{result['parameter_name']}单因素敏感性结果：{points} {result['unit']}；结果极差为 {result['result_range']:,.3f} {result['unit']}。"
            if zh
            else f"One-at-a-time sensitivity for {result['parameter_name']}: {points} {result['unit']}; result range is {result['result_range']:,.3f} {result['unit']}."
        )
    elif tool == "carbon_intensity":
        answer = (
            f"碳强度为 {result['result']:,.6f} {result['unit']}。"
            if zh
            else f"Carbon intensity is {result['result']:,.6f} {result['unit']}."
        )
    elif tool == "transport_emission":
        answer = (
            f"A4运输排放为 {result['result']:,.3f} {result['unit']}。"
            if zh
            else f"A4 transport emission is {result['result']:,.3f} {result['unit']}."
        )
    elif tool in {"factor_reduction", "material_substitution", "amortise_reusable_resource"}:
        answer = (
            f"情景计算完成：基准 {result['baseline']:,.3f} kgCO2e，替代 {result['alternative']:,.3f} kgCO2e，变化 {result['delta']:,.3f} kgCO2e（{result['percentage_change']:+.3f}%）。"
            if zh
            else f"Scenario completed: baseline {result['baseline']:,.3f} kgCO2e, alternative {result['alternative']:,.3f} kgCO2e, delta {result['delta']:,.3f} kgCO2e ({result['percentage_change']:+.3f}%)."
        )
    elif tool == "dependency_ordered_calculation_program":
        step_names = result.get("program_specification", {}).get("steps", [])
        answer = (
            f"依赖有序的复合计算程序已完成：{'; '.join(step_names)}。"
            if zh
            else "The dependency-ordered compound calculation completed: "
            + "; ".join(step_names)
            + "."
        )
    elif tool == "correction_impact_then_preview":
        answer = (
            f"材料 {result['material_code']} 运距由 {result['old_distance_km']:g} km 变为 {result['new_distance_km']:g} km，A4运距倍数为 {result['distance_multiplier']:.6f}；仅生成待确认预览。"
            if zh
            else f"For {result['material_code']}, changing distance from {result['old_distance_km']:g} km to {result['new_distance_km']:g} km gives an A4 distance multiplier of {result['distance_multiplier']:.6f}; only a confirmation preview was prepared."
        )
    else:
        answer = "计算结果已完成。" if zh else "The governed calculation completed."
    return {"answer": answer, "tool": tool}


def _render_retrieval(arguments: RetrievalRenderArguments) -> dict[str, Any]:
    query = arguments.query.lower()
    context = arguments.context
    overview = context.get("project_overview", {})
    profile_ids = set(context.get("subontology", {}).get("profile_ids", []))
    zh = arguments.language == "zh"

    def answer(en: str, cn: str, selected: Any) -> dict[str, Any]:
        return {"answer": cn if zh else en, "selected_data": selected}

    if any(term in query for term in ("ontology term", "ontology content", "subontology", "ontology术语", "ontology内容", "子ontology")):
        signature = context.get("subontology", {})
        selected = {
            "profile_ids": signature.get("profile_ids", []),
            "ontology_terms": signature.get("ontology_terms", []),
            "semantic_path_ids": signature.get("semantic_path_ids", []),
            "data_source_ids": signature.get("data_source_ids", []),
        }
        return answer(
            f"Required profiles: {', '.join(selected['profile_ids'])}; {len(selected['ontology_terms'])} ontology URIs and semantic paths {', '.join(selected['semantic_path_ids'])}.",
            f"所需profile为：{', '.join(selected['profile_ids'])}；共{len(selected['ontology_terms'])}个ontology URI，语义路径为{', '.join(selected['semantic_path_ids'])}。",
            selected,
        )
    if any(term in query for term in ("formula", "公式")):
        selected = overview.get("formulae", {})
        return answer(
            "Governed formulae: " + "; ".join(f"{name}: {value}" for name, value in selected.items()),
            "受控公式：" + "；".join(f"{name}: {value}" for name, value in selected.items()),
            selected,
        )
    if any(term in query for term in ("applicability", "适用范围")):
        selected = overview.get("profile", {}).get("applicability_scope") or overview.get("applicability_scope")
        return answer(f"Applicability scope: {selected}.", f"适用范围为：{selected}。", selected)
    if any(term in query for term in ("included stage", "lifecycle stages", "包含哪些", "核算阶段")):
        selected = [item.get("stage") for item in overview.get("totals", [])]
        return answer(
            "Included stages: " + ", ".join(selected) + ".",
            "包含阶段：" + "、".join(selected) + "。",
            selected,
        )
    if context.get("ifc_inventory"):
        inventory = context["ifc_inventory"].get("data", {})
        selected = {
            **{row["role"]: int(row["scoped_product_count"]) for row in inventory.get("sources", [])},
            "combined": int(inventory.get("combined_product_count", 0)),
        }
        return answer(
            "IFC product counts: " + ", ".join(f"{name}: {value}" for name, value in selected.items()) + ".",
            "IFC产品数量：" + "，".join(f"{name}: {value}" for name, value in selected.items()) + "。",
            selected,
        )
    stages = {
        item["stage"]: float(item["value"])
        for item in overview.get("totals", [])
        if item.get("value") is not None
    }
    if "ONT-PROJECT-CARBON" in profile_ids:
        stage_markers = (
            ("A1-A3_ProductStage", ("a1-a3", "a1–a3", "产品阶段")),
            ("A4_TransportationStage", ("project a4", "项目a4", "运输阶段")),
            ("A5_ConstructionStage", ("a5", "施工阶段")),
        )
        selected_stages = {
            stage: stages[stage]
            for stage, markers in stage_markers
            if stage in stages and any(marker in query for marker in markers)
        }
        if selected_stages and not any(term in query for term in ("formula", "公式", "percentage", "占比", "比例", "largest", "最大")):
            selected: Any = selected_stages if len(selected_stages) > 1 else next(iter(selected_stages.items()))
            return answer(
                "Project stage carbon: " + ", ".join(f"{name}: {value:,.3f} kgCO2e" for name, value in selected_stages.items()) + ".",
                "项目阶段碳排放：" + "，".join(f"{name}: {value:,.3f} kgCO2e" for name, value in selected_stages.items()) + "。",
                selected,
            )

    if "ONT-SYSTEM-CARBON" in profile_ids and not context.get("ifc_inventory"):
        scope_markers = (
            ("main_structure", ("main structure", "主体结构")),
            ("enclosure_structure", ("enclosure structure", "围护结构")),
            ("foundation_pit", ("foundation pit", "foundation-pit", "基坑工程")),
        )
        named_scopes = [
            scope
            for scope, markers in scope_markers
            if any(marker in query for marker in markers)
        ]
        if len(named_scopes) == 1 and any(term in query for term in ("a1-a3", "a1–a3", "a4", "a5", "各阶段")):
            scope = named_scopes[0]
            selected = overview.get("scope_stage_totals_kgco2e", {}).get(scope, {})
            return answer(
                f"{scope} stage carbon: " + ", ".join(f"{name}: {float(value):,.3f} kgCO2e" for name, value in selected.items()) + ".",
                f"{scope}各阶段碳排放：" + "，".join(f"{name}: {float(value):,.3f} kgCO2e" for name, value in selected.items()) + "。",
                selected,
            )

    transport_query = any(term in query for term in ("a4", "transport leg", "material transport", "vehicle", "t-km", "运输leg", "材料运输", "载具", "吨公里"))
    foundation_haul_query = any(term in query for term in ("foundation pit", "foundation-pit", "earthwork", "基坑", "土方")) and any(
        term in query for term in ("haul distance", "off-site haul", "外运采用的运距", "土方外运")
    )
    foundation_query = any(term in query for term in ("foundation pit", "foundation-pit", "earthwork", "excavation", "backfill", "基坑", "土方", "开挖", "回填")) and (not transport_query or foundation_haul_query)
    if foundation_query:
        foundation = overview.get("foundation_pit", {})
        geometry = foundation.get("geometry", {})
        if context.get("foundation_cpd_efd"):
            rows = context["foundation_cpd_efd"].get("rows", [])
            selected = sorted({row.get("cpd_code") for row in rows if row.get("cpd_code")})
            return answer(
                "Foundation-pit CPD codes: " + ", ".join(selected) + ".",
                "基坑CPD编码：" + "、".join(selected) + "。",
                selected,
            )
        if any(term in query for term in ("sensitivity result", "sensitivity results", "敏感性结果")):
            selected = foundation.get("sensitivity", {})
            return answer(
                "Foundation-pit low/base/high sensitivity results: " + json.dumps(selected, ensure_ascii=False, sort_keys=True) + ".",
                "基坑低/基准/高敏感性结果：" + json.dumps(selected, ensure_ascii=False, sort_keys=True) + "。",
                selected,
            )
        geometry_fields = (
            ("gross", "gross_envelope_m3", "gross envelope volume", "外包络总体积", "m3"),
            ("backfill", "backfill_m3", "backfill volume", "回填体积", "m3"),
            ("disposal", "offsite_disposal_in_situ_m3", "off-site disposal in-situ volume", "外运弃土原状体积", "m3"),
            ("loose", "offsite_transport_loose_m3", "loose transport volume", "外运松方体积", "m3"),
            ("haul distance", "offsite_haul_distance_km", "off-site haul distance", "外运运距", "km"),
            ("excavation", "excavation_m3", "excavation volume", "开挖体积", "m3"),
        )
        for marker, key, label_en, label_zh, unit in geometry_fields:
            if marker in query or label_zh in query:
                value = float(geometry.get(key, 0.0))
                return answer(
                    f"Foundation-pit {label_en}: {value:,.6f} {unit}.",
                    f"基坑{label_zh}为 {value:,.6f} {unit}。",
                    {key: value, "unit": unit},
                )
        selected = {
            "geometry": geometry,
            "stage_totals_kgco2e": foundation.get("stage_totals_kgco2e", {}),
            "work_package_count": foundation.get("work_package_count"),
            "resource_flow_count": foundation.get("resource_flow_count"),
            "parameters": overview.get("parameters", {}).get("foundation_pit", {}),
        }
        return answer(
            f"Foundation-pit reference facts: excavation {geometry.get('excavation_m3', 0):,.3f} m3, backfill {geometry.get('backfill_m3', 0):,.3f} m3; stage emissions {selected['stage_totals_kgco2e']}.",
            f"基坑参考结果：开挖 {geometry.get('excavation_m3', 0):,.3f} m3，回填 {geometry.get('backfill_m3', 0):,.3f} m3；阶段排放为 {selected['stage_totals_kgco2e']}。",
            selected,
        )
    enclosure_query = (
        any(
            term in query
            for term in ("enclosure", "diaphragm", "retaining", "strut", "围护", "地下连续墙", "地连墙", "支撑")
        )
        # Read enclosure mapping coverage for requests about mapped elements.
        or any(
            term in query
            for term in ("mapping rate", "mapping coverage", "applicable element", "applicable-element", "unmapped", "映射率", "映射覆盖率")
        )
    ) and not transport_query
    if enclosure_query:
        enclosure = overview.get("enclosure_structure", {})
        coverage = enclosure.get("mapping_coverage", {})
        if context.get("enclosure_cpd_efd"):
            rows = context["enclosure_cpd_efd"].get("rows", [])
            selected = sorted({row.get("cpd_code") for row in rows if row.get("cpd_code")})
            return answer(
                "Enclosure CPD codes: " + ", ".join(selected) + ".",
                "围护CPD编码：" + "、".join(selected) + "。",
                selected,
            )
        if any(term in query for term in ("most numerous", "largest type", "数量最多")):
            counts = coverage.get("element_type_counts", {})
            name, value = max(counts.items(), key=lambda item: (int(item[1]), item[0]))
            selected = {"element_type": name, "count": int(value)}
            return answer(
                f"The most numerous enclosure type is {name}, with {int(value)} elements.",
                f"数量最多的围护构件类型为{name}，共{int(value)}个。",
                selected,
            )
        if any(term in query for term in ("mapping rate", "mapping coverage", "coverage", "映射率", "映射覆盖", "覆盖率")):
            selected = {
                "total_elements": coverage.get("total_elements"),
                "applicable_elements": coverage.get("applicable_elements"),
                "mapped_applicable_elements": coverage.get("mapped_applicable_elements"),
                "not_applicable_elements": coverage.get("not_applicable_elements"),
                "missing_quantity_elements": coverage.get("missing_quantity_elements"),
                "mapping_rate": coverage.get("mapping_rate"),
            }
            rate = float(selected.get("mapping_rate") or 0.0) * 100.0
            return answer(
                f"Enclosure mapping coverage is {rate:.3f}%: {selected['mapped_applicable_elements']} of {selected['applicable_elements']} applicable elements are mapped; {selected['not_applicable_elements']} element is marked not applicable and {selected['missing_quantity_elements']} mapped elements lack quantity data.",
                f"围护映射覆盖率为 {rate:.3f}%：{selected['applicable_elements']} 个适用构件中 {selected['mapped_applicable_elements']} 个已映射；{selected['not_applicable_elements']} 个构件标记为不适用，{selected['missing_quantity_elements']} 个已映射构件缺少工程量。",
                selected,
            )
        if any(term in query for term in ("type count", "element count", "各类型", "类型数量")):
            selected = coverage.get("element_type_counts", {})
        elif any(term in query for term in ("variant", "变体")):
            selected = coverage.get("variant_counts", {})
        elif any(term in query for term in ("volume", "体积")) and any(term in query for term in ("diaphragm", "地下连续墙", "地连墙")):
            selected = {"diaphragm_wall_volume_m3": coverage.get("diaphragm_wall_volume_m3")}
        elif any(term in query for term in ("length", "长度")) and any(term in query for term in ("diaphragm", "地下连续墙", "地连墙")):
            selected = {"diaphragm_wall_length_m": coverage.get("diaphragm_wall_length_m")}
        else:
            selected = {
                "mapping_coverage": coverage,
                "work_package_count": enclosure.get("work_package_count"),
                "resource_flow_count": enclosure.get("resource_flow_count"),
                "stage_totals_kgco2e": enclosure.get("stage_totals_kgco2e", {}),
            }
        return answer(
            "Enclosure reference result: " + json.dumps(selected, ensure_ascii=False, sort_keys=True) + ".",
            "围护参考结果：" + json.dumps(selected, ensure_ascii=False, sort_keys=True) + "。",
            selected,
        )
    if transport_query:
        transport = overview.get("transport", {})
        legs = context.get("a4_transport_legs", {}).get("rows", [])
        defaults = context.get("material_transport_defaults", {}).get("rows", [])
        vehicles = context.get("vehicle_factors", {}).get("rows", [])
        if defaults and any(term in query for term in ("default", "assigned", "默认", "配置")):
            selected = {"material_defaults": defaults, "vehicle_factors": vehicles}
            if len(defaults) == 1:
                default = defaults[0]
                vehicle = vehicles[0] if len(vehicles) == 1 else {}
                return answer(
                    f"Material {default.get('material_code')} defaults to {default.get('vehicle_code')} ({vehicle.get('vehicle_name', 'governed vehicle')}) over {default.get('default_distance_km')} km; the vehicle factor is {vehicle.get('factor_kgco2e_per_tkm', 'not available')} kgCO2e/tkm.",
                    f"材料 {default.get('material_code')} 默认配置 {default.get('vehicle_code')}（{vehicle.get('vehicle_name', '受控载具')}），运距 {default.get('default_distance_km')} km；载具因子为 {vehicle.get('factor_kgco2e_per_tkm', '未提供')} kgCO2e/tkm。",
                    selected,
                )
        elif legs and any(term in query for term in ("largest", "top", "most", "最大", "最高", "前五")):
            selected = legs[:5 if any(term in query for term in ("five", "前五")) else 1]
        elif legs and any(term in query for term in ("vehicle code", "each vehicle", "载具编码", "每个载具")):
            counts: dict[str, int] = {}
            for row in legs:
                counts[row["vehicle_code"]] = counts.get(row["vehicle_code"], 0) + 1
            selected = counts
        elif legs and any(term in query for term in ("distance", "运距")):
            counts = {}
            for row in legs:
                counts[row["distance_km"]] = counts.get(row["distance_km"], 0) + 1
            selected = counts
        elif legs and len(legs) < int(transport.get("leg_count", 0)):
            selected = legs
        else:
            selected = transport
        return answer(
            "A4 reference result: " + json.dumps(selected, ensure_ascii=False, sort_keys=True) + ".",
            "A4参考结果：" + json.dumps(selected, ensure_ascii=False, sort_keys=True) + "。",
            selected,
        )
    if context.get("representative_element_trace"):
        trace = context["representative_element_trace"]
        selected = {
            "semantic_route": trace.get("semantic_route"),
            "system": trace.get("system"),
            "element": trace.get("element"),
            "distinct_work_packages": trace.get("distinct_work_packages"),
            "distinct_resources": trace.get("distinct_resources"),
            "distinct_emissions": trace.get("distinct_emissions"),
            "emission_sum_in_returned_rows": trace.get("emission_sum_in_returned_rows"),
        }
        return answer(
            f"Audited chain: {selected['system']} -> {selected['element']} -> {selected['distinct_work_packages']} work packages -> {selected['distinct_resources']} resources -> {selected['distinct_emissions']} emissions; sum {selected['emission_sum_in_returned_rows']:,.3f} kgCO2e.",
            f"审计链：{selected['system']} -> {selected['element']} -> {selected['distinct_work_packages']}个工作包 -> {selected['distinct_resources']}个资源 -> {selected['distinct_emissions']}个排放记录；合计 {selected['emission_sum_in_returned_rows']:,.3f} kgCO2e。",
            selected,
        )
    if context.get("resource_breakdown"):
        selected = context["resource_breakdown"]
        return answer(
            f"Matched resource emission is {selected.get('total_emission', 0):,.3f} {selected.get('unit') or 'kgCO2e'}.",
            f"匹配资源排放为 {selected.get('total_emission', 0):,.3f} {selected.get('unit') or 'kgCO2e'}。",
            selected,
        )
    if context.get("work_packages"):
        selected = context["work_packages"][:10]
        return answer(
            "Top work-package templates: " + json.dumps(selected, ensure_ascii=False, sort_keys=True) + ".",
            "工作包模板Top-10：" + json.dumps(selected, ensure_ascii=False, sort_keys=True) + "。",
            selected,
        )
    # Return project totals only for a project overview request.
    overview_terms = (
        "total",
        "stage",
        "overview",
        "baseline",
        "a1-a5",
        "a1–a5",
        "总量",
        "阶段",
        "合计",
        "基准",
    )
    if not any(term in query for term in overview_terms):
        raise ValueError(
            "RETRIEVAL_ANSWER_MISMATCH: no governed rendering matched the "
            "query; refusing to answer with unrelated project totals"
        )
    selected = {
        "stage_totals_kgco2e": stages,
        "combined_total_kgco2e": float(overview.get("combined_total_kgco2e") or sum(stages.values())),
    }
    return answer(
        f"Project stages: {stages}; combined total {selected['combined_total_kgco2e']:,.3f} kgCO2e.",
        f"项目阶段排放为：{stages}；合计 {selected['combined_total_kgco2e']:,.3f} kgCO2e。",
        selected,
    )


def build_tool_registry(
    config: ProjectConfig,
    cq_catalogue: CQCatalogue,
    knowledge: Any,
    transactions: TransactionService,
    metadata_pipeline: MetadataPipeline | None = None,
) -> ToolRegistry:
    profiles = ontology_profiles()
    metadata = metadata_pipeline or build_metadata_pipeline(config, cq_catalogue)
    reference_retriever = ReferenceDatabaseRetriever(config, metadata.index.backend)

    def reader_plan(arguments: ReaderPlanArguments) -> dict[str, Any]:
        query = arguments.query.lower().replace("–", "-").replace("—", "-")
        normalised_query = query.replace("-", " ")
        contract = QueryContract.model_validate(
            arguments.query_contract
            or parse_query_contract(
                arguments.query,
                QueryMode.INFORMATION_RETRIEVAL,
            ).model_dump(mode="json")
        )
        operations = set(contract.operation_ids)
        llm_contract = contract.llm_contract_validated
        steps: list[RetrievalStep] = [
            RetrievalStep(
                step_id="READ-CKG-OVERVIEW",
                protocol="CKG_TEMPLATE",
                target="project assessment, system and lifecycle-stage baseline",
                template_id="CKG-PROJECT-OVERVIEW",
                bindings={},
            )
        ]
        target_databases = ["CKG"]

        def has(*terms: str) -> bool:
            return any(term in query or term in normalised_query for term in terms)

        if "retrieve_elements" in operations or (
            not llm_contract
            and has(
                "hotspot", "top 10", "top ten", "highest", "largest", "most",
                "热点", "最高", "最大", "前十",
            )
        ):
            requested_stage = None
            if contract.target_stages == ["A1-A3_ProductStage"]:
                requested_stage = "A1-A3"
            elif contract.target_stages == ["A5_ConstructionStage"]:
                requested_stage = "A5"
            steps.append(
                RetrievalStep(
                    step_id="READ-CKG-HOTSPOTS",
                    protocol="CKG_TEMPLATE",
                    target="ranked physical-element carbon emissions",
                    template_id="CKG-TOP-ELEMENTS",
                    bindings={"limit": 10, "stage": requested_stage},
                    dependencies=["READ-CKG-OVERVIEW"],
                )
            )
        if "retrieve_element_trace" in operations or (
            not llm_contract
            and has("trace", "provenance", "audit chain", "s-e-p-r-c", "追溯", "溯源", "审计链")
        ):
            steps.append(
                RetrievalStep(
                    step_id="READ-CKG-TRACE-SEED",
                    protocol="CKG_TEMPLATE",
                    target="highest-carbon element used as an explicit trace seed",
                    template_id="CKG-TOP-ELEMENTS",
                    bindings={"limit": 1, "stage": "A1-A3"},
                    dependencies=["READ-CKG-OVERVIEW"],
                )
            )
            steps.append(
                RetrievalStep(
                    step_id="READ-CKG-SEPRC-TRACE",
                    protocol="CKG_TEMPLATE",
                    target="System-Element-Process-Resource-Carbon provenance chain",
                    template_id="CKG-ELEMENT-TRACE",
                    bindings={"element_uri_from_step": "READ-CKG-TRACE-SEED"},
                    dependencies=["READ-CKG-TRACE-SEED"],
                )
            )
        resource_terms: list[str] = []
        resource_targets = set(contract.resource_targets)

        def targets_controlled_resource(label: str) -> bool:
            controlled = entity_label_tokens(label)
            return any(
                controlled <= entity_label_tokens(target)
                for target in resource_targets
            )

        if targets_controlled_resource("concrete") or (
            not llm_contract and has("concrete", "ggbs", "slag", "混凝土", "矿渣", "矿粉")
        ):
            resource_terms.append("Concrete")
        if targets_controlled_resource("steel") or (
            not llm_contract and has("steel", "rebar", "钢材", "钢筋", "钢结构")
        ):
            resource_terms.append("Steel")
        if targets_controlled_resource("formwork") or (
            not llm_contract and has("formwork", "reuse", "模板", "周转")
        ):
            resource_terms.append("Formwork")
        for resource_index, resource_term in enumerate(dict.fromkeys(resource_terms), start=1):
            steps.append(
                RetrievalStep(
                    step_id=f"READ-CKG-RESOURCE-{resource_index:02d}",
                    protocol="CKG_TEMPLATE",
                    target=f"{resource_term} resource and carbon aggregates",
                    template_id="CKG-RESOURCE-BREAKDOWN",
                    bindings={"term": resource_term, "stage": "A1-A3", "limit": 30},
                    dependencies=["READ-CKG-OVERVIEW"],
                )
            )
        if "retrieve_work_packages" in operations or (
            not llm_contract
            and has("work package", "process", "工作包", "工序", "施工过程")
        ):
            steps.append(
                RetrievalStep(
                    step_id="READ-CKG-WORK-PACKAGES",
                    protocol="CKG_TEMPLATE",
                    target="instantiated work-package template aggregates",
                    template_id="CKG-WORK-PACKAGES",
                    bindings={"term": "", "limit": 30},
                    dependencies=["READ-CKG-OVERVIEW"],
                )
            )

        demand_categories = [
            str(item.get("category"))
            for item in arguments.delta_graph_specification.get("reference_demands", [])
            if item.get("category") in {"CPD", "EFD", "TRANSPORT"}
        ]
        categories = list(dict.fromkeys(demand_categories or contract.reference_categories))
        if not categories and not llm_contract:
            categories = reference_retriever.predict_categories(arguments.query)
        use_reference = bool(demand_categories) or bool(
            operations & {"retrieve_transport_references", "retrieve_cpd_efd"}
        ) or bool(contract.reference_categories) or (
            not llm_contract
            and has(
                "cpd", "efd", "factor", "work package", "quota", "a4", "transport", "vehicle", "distance",
                "foundation pit", "earthwork", "enclosure", "diaphragm", "定额", "因子", "运输", "运距",
                "基坑", "土方", "围护", "地下连续墙",
            )
        )
        reference_steps: list[str] = []
        if use_reference:
            explicit_material = re.search(r"\bM-\d+-\d+\b", arguments.query, re.I)
            explicit_vehicle = re.search(r"\bV-\d+-\d+\b", arguments.query, re.I)
            for index, category in enumerate(categories, start=1):
                filters: dict[str, Any] = {}
                if category == "EFD":
                    filters["region"] = "China"
                if category == "TRANSPORT" and explicit_material:
                    filters["material_code"] = explicit_material.group(0).upper()
                if category == "TRANSPORT" and explicit_vehicle:
                    filters["vehicle_code"] = explicit_vehicle.group(0).upper()
                step_id = f"READ-REFERENCE-{index:02d}-{category}"
                reference_steps.append(step_id)
                steps.append(
                    RetrievalStep(
                        step_id=step_id,
                        protocol="REFERENCE_HYBRID",
                        target=f"{category} parameter candidates",
                        category_filter=category,
                        hard_filters=filters,
                    )
                )
                if category not in target_databases:
                    target_databases.append(category)
        if "retrieve_ifc_inventory" in operations or (
            not llm_contract
            and (
                has("globalid", "ifc构件", "ifc产品")
                or (has("ifc") and has("product", "products", "产品", "构件"))
            )
        ):
            steps.append(
                RetrievalStep(
                    step_id="READ-REFERENCE-IFC-INVENTORY",
                    protocol="REFERENCE_EXACT",
                    target="versioned IFC inventory summary",
                    hard_filters={},
                )
            )
            reference_steps.append("READ-REFERENCE-IFC-INVENTORY")
            target_databases.append("IFC_INVENTORY")

        integration_dependencies = [step.step_id for step in steps]
        steps.append(
            RetrievalStep(
                step_id="READ-INTEGRATE-CONTEXT",
                protocol="INTEGRATION",
                target="provenance-preserving composite data context",
                dependencies=integration_dependencies,
            )
        )
        return ReaderRetrievalPlan(
            query=arguments.query,
            steps=steps,
            target_databases=target_databases,
            uses_ckg_template_path=True,
            uses_reference_hybrid_path=bool(reference_steps),
        ).model_dump(mode="json")

    template_definitions = {
        "CKG-PROJECT-OVERVIEW": ("CQ-C04.rq", "project_overview"),
        "CKG-TOP-ELEMENTS": ("CQ-C09.rq", "top_elements"),
        "CKG-RESOURCE-BREAKDOWN": ("CQ-P04.rq", "resource_breakdown"),
        "CKG-WORK-PACKAGES": ("CQ-P03.rq", "work_package_breakdown"),
        "CKG-ELEMENT-TRACE": ("CQ-C02.rq", "element_trace"),
    }

    def execute_ckg_template(arguments: CKGTemplateArguments) -> dict[str, Any]:
        definition = template_definitions.get(arguments.template_id)
        if definition is None:
            raise ValueError(f"unknown preverified CKG template: {arguments.template_id}")
        filename, operation = definition
        path = config.cq_query_dir / filename
        query_text = path.read_text(encoding="utf-8")
        if not re.search(r"\b(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\b", query_text, re.I):
            raise ValueError(f"template {filename} is not a read query")
        if re.search(r"\b(?:INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|MOVE|COPY|ADD)\b", query_text, re.I):
            raise ValueError(f"template {filename} contains a mutating SPARQL operation")
        bindings = arguments.bindings
        if operation == "project_overview":
            result = knowledge.project_overview()
        elif operation == "top_elements":
            result = knowledge.top_elements(int(bindings.get("limit", 10)), bindings.get("stage"))
        elif operation == "resource_breakdown":
            result = knowledge.resource_breakdown(
                str(bindings.get("term", "")),
                bindings.get("stage"),
                int(bindings.get("limit", 30)),
            )
        elif operation == "work_package_breakdown":
            result = knowledge.work_package_breakdown(
                str(bindings.get("term", "")),
                int(bindings.get("limit", 30)),
            )
        elif operation == "element_trace":
            element_uri = str(bindings.get("element_uri", ""))
            if not element_uri:
                raise ValueError("element_uri is required by CKG-ELEMENT-TRACE")
            result = knowledge.element_trace(element_uri, int(bindings.get("limit", 250)))
        else:  # pragma: no cover - protected by the closed registry above
            raise ValueError(operation)

        # Check indexed results against SPARQL on the release graph.
        parity_note: dict[str, Any] = _sparql_parity(config, knowledge, arguments.template_id, bindings)
        return {
            "template_id": arguments.template_id,
            "sparql_file": str(path),
            "sparql_sha256": sha256_file(path),
            "bindings": bindings,
            "execution_backend": (
                "materialized_CKG_index_with_runtime_sparql_parity"
                if parity_note.get("status") == "verified"
                else "materialized_CKG_index_equivalent_to_preverified_SPARQL"
            ),
            "sparql_parity": parity_note,
            "result": result,
        }

    def _swrl_inference_store_summary(config: ProjectConfig) -> dict[str, Any]:
        """Identity of the materialised SWRL inference store, when present."""

        import sqlite3

        from .ontology_runtime import swrl_inference_path

        store = swrl_inference_path(config)
        if not store.exists():
            return {"status": "absent", "path": str(store)}
        try:
            connection = sqlite3.connect(store)
            try:
                meta = dict(connection.execute("SELECT key, value FROM meta"))
                count = connection.execute("SELECT COUNT(*) FROM inferences").fetchone()[0]
            finally:
                connection.close()
        except sqlite3.Error as exc:
            return {"status": "unreadable", "error": str(exc)}
        return {
            "status": "ready",
            "path": str(store),
            "inference_count": count,
            "rule_source_sha256": meta.get("rule_source_sha256"),
            "engine": meta.get("engine"),
            "rule_firings": meta.get("rule_firings"),
        }

    def _sparql_parity(
        config: ProjectConfig,
        knowledge: Any,
        template_id: str,
        bindings: dict[str, Any],
    ) -> dict[str, Any]:
        import os

        from .knowledge import KnowledgeIndex

        if os.environ.get("MSCA_SPARQL_PARITY", "").lower() in {"off", "0", "false"}:
            return {"status": "disabled_by_env"}
        if not isinstance(knowledge, KnowledgeIndex):
            # Skip graph parity for test doubles that have no release graph.
            return {"status": "skipped_non_governed_index"}
        from .ontology_runtime import assert_template_parity

        report = assert_template_parity(config, knowledge, template_id, bindings)
        return {
            "status": "verified",
            "template_id": report["template_id"],
            "compared_rows": report["compared_rows"],
        }

    def hybrid_reference_lookup(arguments: ReferenceLookupArguments) -> dict[str, Any]:
        return reference_retriever.search(
            arguments.query,
            categories=arguments.categories,
            hard_filters=arguments.hard_filters,
            top_k=arguments.top_k,
            lexical_only=arguments.lexical_only,
        )

    def resolve_reference_demands(
        arguments: ReferenceDemandResolutionArguments,
    ) -> dict[str, Any]:
        delta = DeltaGraphSpecification.model_validate(arguments.delta_graph)
        overlay = ScenarioOverlay.model_validate(arguments.overlay)
        exact = arguments.exact_artifacts

        def rows(name: str) -> list[dict[str, Any]]:
            value = exact.get(name, {})
            return list(value.get("rows", [])) if isinstance(value, dict) else []

        def hybrid_results(category: str) -> list[dict[str, Any]]:
            return [
                result
                for search in arguments.reference_searches
                if category in search.get("category_prediction", [])
                for result in search.get("results", [])
            ]

        defaults = rows("material_transport_defaults")
        vehicles = rows("vehicle_factors")
        efd_results = hybrid_results("EFD")
        updated_demands: list[ReferenceDemand] = []
        resolved_values: dict[str, dict[str, Any]] = {}
        for demand in delta.reference_demands:
            material_code_match = re.search(r"\bM-\d+-\d+\b", demand.query_text, re.I)
            vehicle_code_match = re.search(r"\bV-\d+-\d+\b", demand.query_text, re.I)
            material_code = material_code_match.group(0).upper() if material_code_match else None
            vehicle_code = vehicle_code_match.group(0).upper() if vehicle_code_match else None
            material_source = next(
                (
                    item
                    for item in defaults
                    if material_code
                    and str(item.get("material_code", "")).upper() == material_code
                ),
                None,
            )
            if material_source and not vehicle_code:
                vehicle_code = str(material_source.get("vehicle_code", "")).upper() or None
            vehicle_source = next(
                (
                    item
                    for item in vehicles
                    if vehicle_code
                    and str(item.get("vehicle_code", "")).upper() == vehicle_code
                ),
                None,
            )
            identified_efd_results = [
                item
                for item in efd_results
                if item.get("code")
                and re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(str(item['code']))}(?![A-Za-z0-9])",
                    demand.query_text,
                    re.I,
                )
            ]
            resolution: dict[str, Any] = {}
            for field in demand.required_fields:
                if field == "distance_km":
                    # Use a transport default only when its material is identified.
                    source = material_source
                    if source:
                        resolution[field] = {
                            "value": float(source["default_distance_km"]),
                            "unit": "km",
                            "source": source.get("source") or "material_transport_defaults",
                            "record_code": source.get("material_code") or source.get("code"),
                        }
                elif field == "factor_kgco2e_per_tkm":
                    source = vehicle_source
                    if source:
                        resolution[field] = {
                            "value": float(source["factor_kgco2e_per_tkm"]),
                            "unit": "kgCO2e/tkm",
                            "source": source.get("source") or "transport_vehicles",
                            "record_code": source.get("vehicle_code") or source.get("code"),
                        }
                elif field in {"baseline_factor", "alternative_factor"}:
                    keywords = (
                        ("concrete", "cement", "ready-mixed")
                        if field == "baseline_factor"
                        else ("ggbs", "slag", "blast furnace")
                    )
                    source = next(
                        (
                            item
                            for item in identified_efd_results
                            if item.get("factor_value") is not None
                            and any(
                                keyword in " ".join(item.get("names", [])).lower()
                                for keyword in keywords
                            )
                        ),
                        None,
                    )
                    if source:
                        resolution[field] = {
                            "value": float(source["factor_value"]),
                            "unit": source.get("factor_unit_uri"),
                            "source": source.get("source"),
                            "record_code": source.get("code"),
                        }
            demand_resolved = all(field in resolution for field in demand.required_fields)
            updated = demand.model_copy(
                update={
                    "resolved": demand_resolved,
                    "resolution": resolution,
                }
            )
            updated_demands.append(updated)
            if demand_resolved:
                resolved_values.update(resolution)

        new_overrides = list(delta.overrides)
        for index, (field, resolution) in enumerate(sorted(resolved_values.items()), start=1):
            new_overrides.append(
                DeltaOverride(
                    override_id=f"REF-OVR-{index:03d}",
                    change_type="reference_resolution",
                    target=updated_demands[0].target if updated_demands else "unspecified",
                    operator="replace",
                    value={"parameter": field, "value": resolution["value"]},
                    unit=resolution.get("unit"),
                    precedence="resolved_reference",
                    provenance={
                        "source": resolution.get("source"),
                        "record_code": resolution.get("record_code"),
                        "resolution_digest": stable_digest(resolution),
                    },
                )
            )
        unresolved = [item for item in updated_demands if not item.resolved]
        ready = not unresolved and bool(updated_demands)
        updated_delta = delta.model_copy(
            update={
                "overrides": new_overrides,
                "reference_demands": updated_demands,
                "ready": ready,
                "feasibility_status": "feasible" if ready else "incomplete",
                "issues": [] if ready else delta.issues,
            }
        )
        updated_overlay = overlay
        if overlay.changes and resolved_values:
            changes = list(overlay.changes)
            parameters = dict(changes[0].parameters)
            parameters.update(
                {field: resolution["value"] for field, resolution in resolved_values.items()}
            )
            assumed = list(changes[0].assumed_parameters)
            assumed.extend(
                f"{field}:resolved_reference" for field in resolved_values if f"{field}:resolved_reference" not in assumed
            )
            changes[0] = changes[0].model_copy(
                update={"parameters": parameters, "assumed_parameters": assumed}
            )
            updated_overlay = overlay.model_copy(
                update={
                    "changes": changes,
                    "ready": ready,
                    "issues": [] if ready else overlay.issues,
                }
            )
        return {
            "delta_graph": updated_delta.model_dump(mode="json"),
            "overlay": updated_overlay.model_dump(mode="json"),
            "resolved_fields": resolved_values,
            "unresolved_demands": [item.model_dump(mode="json") for item in unresolved],
        }

    def read_reference_artifact(arguments: ReferenceArtifactArguments) -> dict[str, Any]:
        paths = {
            "enclosure_assignments": config.artifact_dir / "reference_case" / "element_work_package_assignments.csv",
            "enclosure_cpd_efd": config.artifact_dir / "reference_case" / "enclosure_cpd_efd_mapping.csv",
            "foundation_cpd_efd": config.artifact_dir / "reference_case" / "foundation_pit_work_packages.csv",
            "a4_transport_legs": config.artifact_dir / "reference_case" / "a4_transport_legs.csv",
            "material_transport_defaults": config.reference_case_dir / "material_transport_defaults.csv",
            "vehicle_factors": config.reference_case_dir / "transport_vehicles.csv",
            "ifc_inventory": config.artifact_dir / "ifc" / "ifc_inventory_summary.json",
        }
        path = paths[arguments.dataset]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            return {
                "dataset": arguments.dataset,
                "source": str(path),
                "data": json.loads(path.read_text(encoding="utf-8")),
            }
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for field, expected in arguments.filters.items():
            rows = [row for row in rows if str(row.get(field, "")) == str(expected)]
        if arguments.sort_by:
            def key(row: dict[str, str]) -> tuple[int, Any]:
                value = row.get(arguments.sort_by or "", "")
                try:
                    return 0, float(value)
                except (TypeError, ValueError):
                    return 1, str(value)

            rows.sort(key=key, reverse=arguments.descending)
        return {
            "dataset": arguments.dataset,
            "source": str(path),
            "matched_row_count": len(rows),
            "returned_row_count": min(len(rows), arguments.limit),
            "rows": rows[: arguments.limit],
        }

    def transaction_preview(arguments: TransactionPreviewArguments) -> dict[str, Any] | None:
        preview = transactions.preview(
            ScenarioOverlay.model_validate(arguments.overlay),
            query_contract=arguments.query_contract,
            material_transport_defaults=arguments.material_transport_defaults,
        )
        return preview.model_dump(mode="json") if preview else None

    def construct_delta_graph(arguments: DeltaGraphArguments) -> dict[str, Any]:
        overlay = ScenarioOverlay.model_validate(arguments.overlay)
        predicate_map = {
            "mass": "http://qudt.org/schema/qudt/numericValue",
            "mass_unit": "http://qudt.org/schema/qudt/unit",
            "distance_km": "http://bimlca.uni.edu/ontology#hasTravelDistance",
            "factor_kgco2e_per_tkm": "http://bimlca.uni.edu/efd/schema#factorValue",
            "substitution_fraction": "http://bimlca.uni.edu/ontology#substitutionFraction",
            "baseline_factor": "http://bimlca.uni.edu/efd/schema#factorValue",
            "alternative_factor": "http://bimlca.uni.edu/efd/schema#factorValue",
            "reduction_fraction": "http://bimlca.uni.edu/scenario#reductionFraction",
            "reuse_cycles": "http://bimlca.uni.edu/scenario#reuseCycles",
            "parameter_values": "http://bimlca.uni.edu/scenario#parameterSweep",
            "baseline_parameter": "http://bimlca.uni.edu/scenario#baselineParameter",
        }
        unit_map = {
            "mass": "declared-by-mass_unit",
            "distance_km": "km",
            "factor_kgco2e_per_tkm": "kgCO2e/tkm",
            "baseline_factor": "query-declared-compatible-unit",
            "alternative_factor": "query-declared-compatible-unit",
            "reduction_fraction": "fraction",
            "substitution_fraction": "fraction",
            "reuse_cycles": "cycle",
            "parameter_values": "dimensionless",
            "baseline_parameter": "dimensionless",
        }
        operator_map = {
            "transport_override": "replace",
            "material_substitution": "substitute",
            "reuse_cycle_override": "replace",
            "factor_reduction": "multiply",
            "sensitivity_analysis": "sweep",
        }
        overrides: list[DeltaOverride] = []
        demands: list[ReferenceDemand] = []
        for change_index, change in enumerate(overlay.changes, start=1):
            explicit = set(change.explicit_parameters)
            assumed = set(change.assumed_parameters)
            for parameter_index, (name, value) in enumerate(change.parameters.items(), start=1):
                precedence = (
                    "explicit_user"
                    if name in explicit
                    else "governed_default"
                    if any(name in item for item in assumed)
                    else "resolved_reference"
                )
                overrides.append(
                    DeltaOverride(
                        override_id=f"OVR-{change_index:02d}-{parameter_index:02d}",
                        change_type=change.change_type,
                        target=change.target,
                        predicate_uri=predicate_map.get(name),
                        operator=operator_map.get(change.change_type, "replace"),
                        value={"parameter": name, "value": value},
                        unit=unit_map.get(name),
                        precedence=precedence,
                        provenance={
                            "source": "user_query" if precedence == "explicit_user" else "declared_reference_profile",
                            "query_digest": stable_digest(arguments.query),
                            "overlay_id": overlay.overlay_id,
                        },
                    )
                )
        missing_fields = list(
            dict.fromkeys(
                field
                for issue in overlay.issues
                for field in issue.missing_fields
            )
        )
        for index, field in enumerate(missing_fields, start=1):
            first_change_type = overlay.changes[0].change_type if overlay.changes else ""
            if "factor" in field and first_change_type == "transport_override":
                category = "TRANSPORT"
            elif "factor" in field:
                category = "EFD"
            elif field in {"mass", "mass_unit", "distance_km"}:
                category = "TRANSPORT"
            elif "work" in field or "quantity" in field:
                category = "CPD"
            else:
                category = "OTHER"
            demands.append(
                ReferenceDemand(
                    demand_id=f"REF-{index:03d}",
                    category=category,
                    query_text=arguments.query,
                    target=overlay.changes[0].target if overlay.changes else "unspecified",
                    required_fields=[field],
                )
            )
        forbidden = any(
            term in arguments.query.lower()
            for term in ("operation stage", "end of life", "demolition", "recycling benefit", "运营阶段", "拆除阶段", "回收收益")
        )
        ready = overlay.ready and not demands and not forbidden
        return DeltaGraphSpecification(
            delta_id=f"delta-{stable_digest({'overlay': arguments.overlay, 'mode': arguments.mode.value})[:16]}",
            mode=arguments.mode,
            persistence=overlay.persistence,
            overrides=overrides,
            reference_demands=demands,
            ready=ready,
            feasibility_status="forbidden" if forbidden else "feasible" if ready else "incomplete",
            issues=overlay.issues,
        ).model_dump(mode="json")

    def parse_semantic_query(arguments: SemanticParseArguments) -> dict[str, Any]:
        interpretation = metadata.parse_query(
            arguments.query,
            arguments.mode,
            anchor_terms=arguments.anchor_terms,
            cq_ids=arguments.cq_ids,
        )
        return {
            "interpretation": interpretation.model_dump(mode="json"),
            "query_phrases": metadata.query_phrases(interpretation),
            "backend": metadata.index.backend_info.model_dump(mode="json"),
            "index_manifest": metadata.index.manifest,
        }

    def retrieve_ontology(arguments: HybridOntologyRetrievalArguments) -> list[dict[str, Any]]:
        interpretation = QueryInterpretation.model_validate(arguments.interpretation)
        return [
            item.model_dump(mode="json")
            for item in metadata.retrieve(interpretation, arguments.phrases)
        ]

    def expand_ontology(arguments: OntologyExpansionArguments) -> dict[str, Any]:
        interpretation = QueryInterpretation.model_validate(arguments.interpretation)
        candidates, relations = metadata.expand(
            interpretation,
            [OntologyEntityCandidate.model_validate(item) for item in arguments.seeds],
        )
        return {
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "relations": [item.model_dump(mode="json") for item in relations],
        }

    def resolve_ontology_paths(arguments: OntologyPathArguments) -> list[dict[str, Any]]:
        interpretation = QueryInterpretation.model_validate(arguments.interpretation)
        return [
            item.model_dump(mode="json")
            for item in metadata.resolve_paths(
                interpretation,
                [OntologyEntityCandidate.model_validate(value) for value in arguments.candidates],
            )
        ]

    def verify_ontology_backbone(arguments: BackboneVerificationArguments) -> dict[str, Any]:
        interpretation = QueryInterpretation.model_validate(arguments.interpretation)
        verification = metadata.verify(
            interpretation,
            [OntologyEntityCandidate.model_validate(value) for value in arguments.candidates],
            [OntologyRelation.model_validate(value) for value in arguments.relations],
            [ResolvedSemanticPath.model_validate(value) for value in arguments.paths],
        )
        return verification.model_dump(mode="json")

    def supplement_metadata_query(arguments: MetadataSupplementArguments) -> list[str]:
        return metadata.supplement(
            QueryInterpretation.model_validate(arguments.interpretation),
            BackboneVerification.model_validate(arguments.verification),
            arguments.previous_phrases,
        )

    specs = [
        ToolSpec(
            "parse_query_contract",
            AgentName.ORCHESTRATOR,
            "Parse scope, operators, dependency shape and postconditions into a governed non-numeric query contract.",
            QueryContractArguments,
            lambda value: parse_query_contract(
                value.query,
                value.mode,
                advisory_task_family=value.advisory_task_family,
                advisory_operation_hints=value.advisory_operation_hints,
                advisory_semantic_anchors=value.advisory_semantic_anchors,
                advisory_contract=value.advisory_contract,
            ).model_dump(mode="json"),
            "Typed query contract; never engineering values, arithmetic results or graph mutations.",
        ),
        ToolSpec("build_execution_plan", AgentName.ORCHESTRATOR, "Build the validated agent sequence for the selected mode.", ExecutionPlanArguments, _execution_plan, "Ordered agent names and analysis flag."),
        ToolSpec("search_competency_questions", AgentName.METADATA, "Retrieve governed competency questions for a natural-language query.", CQSearchArguments, lambda value: [item.model_dump(mode="json") for item in cq_catalogue.search(value.query, value.limit)], "Ranked CQ records."),
        ToolSpec("parse_semantic_query", AgentName.METADATA, "Parse engineering terms, ontology roles, granularity and assessment boundary into a typed query interpretation.", SemanticParseArguments, parse_semantic_query, "Query interpretation, enhanced phrases and retrieval-backend manifest."),
        ToolSpec("hybrid_ontology_retrieval", AgentName.METADATA, "Retrieve ontology classes, properties and individuals with BM25 plus dense retrieval followed by reranking.", HybridOntologyRetrievalArguments, retrieve_ontology, "Ranked ontology seed candidates with per-stage scores."),
        ToolSpec("expand_related_ontology", AgentName.METADATA, "Recall adjacent ontology entities and relations over explicit and schema-derived graph edges.", OntologyExpansionArguments, expand_ontology, "Expanded entities and evidence-bearing relations."),
        ToolSpec("resolve_ontology_paths", AgentName.METADATA, "Discover connected semantic paths between required S-E-P-R-C roles.", OntologyPathArguments, resolve_ontology_paths, "Ranked resolved ontology paths."),
        ToolSpec("verify_query_backbone", AgentName.METADATA, "Verify role, property, granularity, boundary and relational sufficiency before data retrieval.", BackboneVerificationArguments, verify_ontology_backbone, "Typed backbone sufficiency decision and missing concepts."),
        ToolSpec("supplement_missing_concepts", AgentName.METADATA, "Generate bounded schema-grounded supplementary terms for a failed sufficiency check.", MetadataSupplementArguments, supplement_metadata_query, "Novel retry phrases only."),
        ToolSpec("map_ontology_entities", AgentName.METADATA, "Map bilingual entity mentions to governed ontology URIs.", QueryArguments, lambda value: [item.model_dump(mode="json") for item in map_entity_mentions(value.query)], "Entity pointers with URIs and confidence."),
        ToolSpec("select_ontology_profiles", AgentName.METADATA, "Select reference subontology profiles required by the query.", OntologySelectionArguments, _select_ontology_profiles, "Ontology profile IDs and selection basis."),
        ToolSpec("plan_reader_retrieval", AgentName.READER, "Create the dual-path CKG-template/reference-database retrieval plan from verified metadata.", ReaderPlanArguments, reader_plan, "Typed retrieval steps, dependencies and target databases."),
        ToolSpec("execute_ckg_sparql_template", AgentName.READER, "Fill and execute a whitelisted, preverified CKG SPARQL template through its materialized index view.", CKGTemplateArguments, execute_ckg_template, "Template digest, bindings, execution backend and governed CKG result."),
        ToolSpec("hybrid_reference_lookup", AgentName.READER, "Retrieve CPD, EFD or transport parameter records with category prediction, BM25/dense/rerank scoring and hard filters.", ReferenceLookupArguments, hybrid_reference_lookup, "Ranked parameter records and full retrieval trace."),
        ToolSpec("resolve_reference_demands", AgentName.READER, "Resolve qualitative delta parameters from filtered governed references without overriding explicit user values.", ReferenceDemandResolutionArguments, resolve_reference_demands, "Updated delta graph, overlay, resolved fields and unresolved demands."),
        ToolSpec(
            "check_knowledge_readiness",
            AgentName.READER,
            "Check whether the governed CKG index is current before reading facts.",
            EmptyArguments,
            lambda _: {
                "ready": bool(knowledge.is_current()),
                "swrl_inference_store": _swrl_inference_store_summary(config),
            },
            "Boolean knowledge-index readiness plus the materialised SWRL inference store identity.",
        ),
        ToolSpec("read_project_overview", AgentName.READER, "Read project, stage, scope, foundation and transport reference facts.", EmptyArguments, lambda _: knowledge.project_overview(), "Project reference-case overview."),
        ToolSpec("read_top_elements", AgentName.READER, "Read the highest-carbon physical elements from the CKG index.", TopElementsArguments, lambda value: knowledge.top_elements(value.limit, value.stage), "Ranked element carbon records."),
        ToolSpec("read_resource_breakdown", AgentName.READER, "Read resource-level carbon aggregates from the CKG index.", ResourceBreakdownArguments, lambda value: knowledge.resource_breakdown(value.term, value.stage, value.limit), "Resource descriptor aggregates."),
        ToolSpec("read_work_package_breakdown", AgentName.READER, "Read instantiated work-package template aggregates.", WorkPackageBreakdownArguments, lambda value: knowledge.work_package_breakdown(value.term, value.limit), "Work-package aggregates."),
        ToolSpec("read_element_trace", AgentName.READER, "Read an auditable System-Element-Process-Resource-Carbon chain.", ElementTraceArguments, lambda value: knowledge.element_trace(value.element_uri, value.limit), "Complete S-E-P-R-C trace."),
        ToolSpec("read_subontology_signature", AgentName.READER, "Read the required URI and semantic-path signature for selected ontology profiles.", SubontologyArguments, _subontology_signature, "Ontology terms, semantic paths, sources and signature files."),
        ToolSpec("read_reference_artifact", AgentName.READER, "Read governed reference-case CSV or IFC-inventory rows with explicit filters and limits.", ReferenceArtifactArguments, read_reference_artifact, "Fingerprintable reference rows or JSON data."),
        ToolSpec("extract_scenario_overlay", AgentName.EXTRACTOR, "Parse a governed transient scenario or persistent correction into typed function arguments.", OverlayArguments, lambda value: extract_overlay(value.query, value.persistent, value.semantic_contract), "Validated ScenarioOverlay with readiness issues."),
        ToolSpec("construct_delta_graph_specification", AgentName.EXTRACTOR, "Convert explicit parameters and qualitative gaps into a provenance-annotated transient or persistent delta graph.", DeltaGraphArguments, construct_delta_graph, "Typed overrides, reference demands, precedence and feasibility status."),
        ToolSpec("compile_calculation_program", AgentName.CALCULATOR, "Compile the shared query contract, retrieved evidence and typed delta into dependency-ordered function calls.", CalculationProgramCompileArguments, lambda value: compile_mas_calculation_program(value.query, value.mode, value.query_contract, value.context, value.overlay), "Executable calculation steps plus a governed final-result template; the compiler performs no carbon arithmetic."),
        ToolSpec("aggregate_named_values", AgentName.CALCULATOR, "Aggregate non-overlapping named carbon values.", NamedValuesArguments, lambda value: aggregate_named_values(value.values, value.unit), "Total, formula, items and unit."),
        ToolSpec("compare_named_values", AgentName.CALCULATOR, "Compare named values with ranking, range and pairwise differences.", ComparisonArguments, lambda value: compare_named_values(value.values, value.unit, value.baseline_name), "Ranked comparison and pairwise deltas."),
        ToolSpec("calculate_shares", AgentName.CALCULATOR, "Calculate fractions and percentages of a governed total.", NamedValuesArguments, lambda value: calculate_shares(value.values, value.unit), "Total and named shares."),
        ToolSpec("rank_named_values", AgentName.CALCULATOR, "Rank named values deterministically.", RankingArguments, lambda value: rank_named_values(value.values, value.unit, value.descending, value.limit), "Stable ranked records."),
        ToolSpec("one_at_a_time_sensitivity", AgentName.CALCULATOR, "Run a direct-proportional one-at-a-time parameter sweep.", SensitivityArguments, lambda value: one_at_a_time_sensitivity(value.baseline_value, value.baseline_parameter, value.parameter_values, parameter_name=value.parameter_name, result_unit=value.result_unit), "Sensitivity points, deltas, extrema and range."),
        ToolSpec("carbon_intensity", AgentName.CALCULATOR, "Calculate carbon intensity from a declared numerator and denominator.", IntensityArguments, lambda value: carbon_intensity(value.emission_kgco2e, value.denominator, value.denominator_unit), "Intensity, formula and unit."),
        ToolSpec("material_emission", AgentName.CALCULATOR, "Calculate A1-A3 material carbon from work quantity, resource consumption rate and emission factor.", MaterialEmissionArguments, lambda value: material_emission(value.work_quantity, value.consumption_rate, value.emission_factor, value.result_unit), "A1-A3 material result, formula and explicit inputs."),
        ToolSpec("equipment_emission", AgentName.CALCULATOR, "Calculate A5 equipment carbon from work volume, productivity, hourly energy use and energy factor.", EquipmentEmissionArguments, lambda value: equipment_emission(value.work_volume, value.productivity_per_hour, value.energy_per_hour, value.energy_factor), "A5 result, machine hours, formula and explicit inputs."),
        ToolSpec("factor_reduction", AgentName.CALCULATOR, "Apply a factor-reduction scenario to a governed baseline.", FactorReductionArguments, lambda value: factor_reduction(value.baseline_emission, value.reduction_fraction), "Baseline, alternative, delta and percentage."),
        ToolSpec("material_substitution", AgentName.CALCULATOR, "Apply partial material substitution with explicit factors.", SubstitutionArguments, lambda value: material_substitution(value.baseline_emission, value.substitution_fraction, value.baseline_factor, value.alternative_factor), "Baseline, alternative, delta and percentage."),
        ToolSpec("transport_emission", AgentName.CALCULATOR, "Calculate A4 from mass, distance and tonne-kilometre factor.", TransportArguments, lambda value: transport_emission(value.mass, value.mass_unit, value.distance_km, value.factor_kgco2e_per_tkm), "A4 result, normalized mass and formula."),
        ToolSpec("amortise_reusable_resource", AgentName.CALCULATOR, "Allocate reusable-resource carbon over explicit reuse cycles.", ReuseArguments, lambda value: amortise_reusable_resource(value.baseline_emission, value.reuse_cycles), "Allocated result, delta and percentage."),
        ToolSpec("apply_named_deltas", AgentName.CALCULATOR, "Compose independently calculated deltas onto one governed baseline without double counting.", NamedDeltaArguments, lambda value: apply_named_deltas(value.baseline_value, value.deltas, value.unit), "Alternative total, net delta, reduction and percentage."),
        ToolSpec("calculate_ratio", AgentName.CALCULATOR, "Calculate a dimensionless ratio for comparison or correction-impact requests.", RatioArguments, lambda value: calculate_ratio(value.numerator, value.denominator, numerator_name=value.numerator_name, denominator_name=value.denominator_name), "Named numerator, denominator and ratio."),
        ToolSpec("evaluate_threshold", AgentName.CALCULATOR, "Evaluate an explicit decision threshold against a governed calculated value.", ThresholdArguments, lambda value: evaluate_threshold(value.value, value.threshold, value.comparison), "Boolean threshold decision and signed margin."),
        ToolSpec("project_foundation_geometry", AgentName.CALCULATOR, "Project foundation excavation, backfill and off-site volumes at a selected utilisation point.", FoundationGeometryArguments, lambda value: project_foundation_geometry(value.gross_envelope_m3, value.plan_utilisation_factor, value.backfill_ratio, value.loose_volume_factor), "Formula-visible foundation geometry projection."),
        ToolSpec("compare_to_reference", AgentName.CALCULATOR, "Validate named alternatives against one governed reference value.", ReferenceComparisonArguments, lambda value: compare_to_reference(value.values, value.reference_value, value.reference_name, value.tolerance), "Per-option differences and reference-conflict flags."),
        ToolSpec("resolve_run_status", AgentName.SYNTHESIS, "Resolve complete, partial, blocked or pending-confirmation status.", StatusArguments, _resolve_status, "Run status and confirmation flag."),
        ToolSpec("render_calculation_result", AgentName.SYNTHESIS, "Render a governed calculation result without recomputing it.", RenderArguments, _render_result, "Natural-language answer tied to tool output."),
        ToolSpec("render_retrieval_result", AgentName.SYNTHESIS, "Select and render the query-relevant facts from Reader context without recomputation.", RetrievalRenderArguments, _render_retrieval, "Natural-language answer plus selected structured facts."),
        ToolSpec("prepare_transaction_preview", AgentName.SYNTHESIS, "Create an append-only transaction preview without mutating source graphs.", TransactionPreviewArguments, transaction_preview, "Transaction preview or null when unsupported."),
    ]
    registry = ToolRegistry(specs)
    expected_profiles = set(profiles)
    if not expected_profiles:
        raise RuntimeError("ontology profile registry is empty")
    return registry


def append_tool_call(
    state_calls: list[dict[str, Any]] | None,
    registry: ToolRegistry,
    agent: AgentName,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], ToolCallRecord]:
    existing = list(state_calls or [])
    record = registry.invoke(agent, tool_name, arguments, len(existing) + 1)
    calls = existing + [record.model_dump(mode="json")]
    return record.result, calls, record
