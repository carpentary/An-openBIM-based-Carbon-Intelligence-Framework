from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .models import PersistenceMode, QueryMode, ReadinessIssue


class SemanticRole(StrEnum):
    """S-E-P-R-C roles used by the paper's semantic backbone verifier."""

    SYSTEM = "S"
    ELEMENT = "E"
    PROCESS = "P"
    RESOURCE = "R"
    CARBON = "C"
    PROPERTY = "PROPERTY"
    SCOPE = "SCOPE"
    OTHER = "OTHER"


class OntologyEntityKind(StrEnum):
    CLASS = "class"
    OBJECT_PROPERTY = "object_property"
    DATATYPE_PROPERTY = "datatype_property"
    ANNOTATION_PROPERTY = "annotation_property"
    INDIVIDUAL = "individual"
    CONCEPT = "concept"
    OTHER = "other"


class ParsedEngineeringTerm(BaseModel):
    term_id: str
    original_text: str
    definition: str
    enhanced_phrases: list[str] = Field(default_factory=list)
    requested_properties: list[str] = Field(default_factory=list)
    source: Literal["query", "orchestrator_anchor", "cq_context", "retry_supplement"]
    out_of_scope: bool = False


class QueryInterpretation(BaseModel):
    query: str
    mode: QueryMode
    intent_summary: str
    in_scope: bool = True
    boundary_status: Literal["inside_active_boundary", "outside_active_boundary", "undetermined"] = (
        "inside_active_boundary"
    )
    out_of_scope_concepts: list[str] = Field(default_factory=list)
    terms: list[ParsedEngineeringTerm] = Field(default_factory=list)
    required_roles: list[SemanticRole] = Field(default_factory=list)
    required_entity_kinds: list[OntologyEntityKind] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    requested_granularity: Literal["project", "system", "element", "process", "resource", "property", "mixed"] = (
        "mixed"
    )
    explicit_identifiers: list[str] = Field(default_factory=list)
    governing_cq_ids: list[str] = Field(default_factory=list)
    parser_backend: str = "deterministic_schema_grounded"


class QueryTaskFamily(StrEnum):
    """Governed operation family passed from Orchestrator to specialist agents."""

    RETRIEVAL = "retrieval"
    RETRIEVAL_CALCULATION = "retrieval_calculation"
    ONTOLOGY_DESIGN = "ontology_design"
    SCENARIO = "scenario"
    COMPOUND_SCENARIO = "compound_scenario"
    CORRECTION_PREVIEW = "correction_preview"
    CORRECTION_IMPACT_PREVIEW = "correction_impact_preview"


class QueryOperator(StrEnum):
    NONE = "none"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    RANK = "rank"
    SHARE = "share"
    INTENSITY = "intensity"
    TRACE = "trace"
    SENSITIVITY = "sensitivity"
    FACTOR_REDUCTION = "factor_reduction"
    MATERIAL_SUBSTITUTION = "material_substitution"
    TRANSPORT = "transport"
    REUSE_ALLOCATION = "reuse_allocation"
    TRANSACTION_PREVIEW = "transaction_preview"


class QueryContract(BaseModel):
    """Typed, non-numeric execution contract shared by the MAS specialists.

    The contract records scope, requested operators and dependency shape.  It
    may contain model-side advisory hints, but only governed fields determine
    tool execution; the model cannot provide quantities, arithmetic results or
    graph mutations through this object.
    """

    query: str
    mode: QueryMode
    intent_kind: Literal[
        "read_existing",
        "derive_from_existing",
        "hypothetical_analysis",
        "persistent_change",
        "boundary_validation",
        "ambiguous",
    ] = "ambiguous"
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    read_only: bool = True
    existing_data_requested: bool = True
    calculation_requested: bool = False
    scenario_requested: bool = False
    persistence_requested: bool = False
    outside_boundary_referenced: bool = False
    boundary_information_request: bool = False
    boundary_operation_requested: bool = False
    requires_human_confirmation: bool = False
    task_family: QueryTaskFamily
    primary_operator: QueryOperator = QueryOperator.NONE
    operation_ids: list[str] = Field(default_factory=list)
    target_scopes: list[str] = Field(default_factory=list)
    target_stages: list[str] = Field(default_factory=list)
    resource_targets: list[str] = Field(default_factory=list)
    explicit_identifiers: list[str] = Field(default_factory=list)
    compound: bool = False
    compound_template: str | None = None
    requires_calculator: bool = False
    requires_post_calculation_retrieval: bool = False
    meta_ontology_query: bool = False
    meta_boundary_query: bool = False
    correction_impact_preview: bool = False
    advisory_task_family: str | None = None
    advisory_operation_hints: list[str] = Field(default_factory=list)
    advisory_semantic_anchors: list[str] = Field(default_factory=list)
    planner_profile_ids: list[str] = Field(default_factory=list)
    reference_categories: list[Literal["CPD", "EFD", "TRANSPORT"]] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    parameter_mentions: list[dict[str, Any]] = Field(default_factory=list)
    semantic_risks: list[dict[str, Any]] = Field(default_factory=list)
    # Record unmatched risk labels for audit only.
    discarded_semantic_risks: list[dict[str, Any]] = Field(default_factory=list)
    llm_contract_validated: bool = False
    ontology_obligations: list[str] = Field(default_factory=list)
    routing_evidence: list[str] = Field(default_factory=list)
    parser_backend: str = "deterministic_query_contract"


class RetrievalBackendInfo(BaseModel):
    lexical_backend: str
    dense_backend: str
    reranker_backend: str
    dense_model_id: str | None = None
    reranker_model_id: str | None = None
    model_revision: str | None = None
    dense_model_revision: str | None = None
    reranker_model_revision: str | None = None
    offline: bool = True
    fallback_used: bool = False


class OntologyEntityCandidate(BaseModel):
    uri: str
    label: str
    alt_labels: list[str] = Field(default_factory=list)
    definition: str = ""
    entity_kind: OntologyEntityKind
    semantic_roles: list[SemanticRole] = Field(default_factory=list)
    source_graphs: list[str] = Field(default_factory=list)
    matched_phrases: list[str] = Field(default_factory=list)
    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dense_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fused_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: bool = True
    discovered_at_hop: int = Field(default=0, ge=0)
    evidence: list[str] = Field(default_factory=list)


class OntologyRelation(BaseModel):
    subject_uri: str
    predicate_uri: str
    object_uri: str
    subject_label: str = ""
    predicate_label: str = ""
    object_label: str = ""
    source: str
    hop: int = Field(default=1, ge=0)
    inferred_from_schema: bool = False
    direction: Literal["forward", "inverse"] = "forward"
    evidence: str = ""

    @property
    def relation_id(self) -> str:
        return f"{self.subject_uri}|{self.predicate_uri}|{self.object_uri}"


class ResolvedSemanticPath(BaseModel):
    path_id: str
    node_uris: list[str] = Field(min_length=1)
    predicate_uris: list[str] = Field(default_factory=list)
    semantic_roles: list[SemanticRole] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["graph_path_finder", "schema_domain_range", "reference_signature"]
    complete_backbone: bool = False

    @model_validator(mode="after")
    def validate_path_shape(self) -> "ResolvedSemanticPath":
        if self.predicate_uris and len(self.predicate_uris) != len(self.node_uris) - 1:
            raise ValueError("a semantic path requires exactly one predicate between adjacent nodes")
        return self


class BackboneVerification(BaseModel):
    sufficient: bool
    structural_backbone_present: bool
    required_roles: list[SemanticRole] = Field(default_factory=list)
    present_roles: list[SemanticRole] = Field(default_factory=list)
    missing_roles: list[SemanticRole] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    present_properties: list[str] = Field(default_factory=list)
    missing_properties: list[str] = Field(default_factory=list)
    granularity_satisfied: bool = True
    boundary_satisfied: bool = True
    reasons: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)


class MetadataRetrievalRound(BaseModel):
    round_no: int = Field(ge=1)
    query_phrases: list[str] = Field(default_factory=list)
    seed_candidates: list[OntologyEntityCandidate] = Field(default_factory=list)
    expanded_candidates: list[OntologyEntityCandidate] = Field(default_factory=list)
    relations: list[OntologyRelation] = Field(default_factory=list)
    semantic_paths: list[ResolvedSemanticPath] = Field(default_factory=list)
    verification: BackboneVerification
    supplementary_terms: list[str] = Field(default_factory=list)
    novel_candidate_count: int = Field(default=0, ge=0)
    stop_reason: Literal[
        "sufficient",
        "retry_required",
        "no_novel_candidates",
        "iteration_limit",
        "outside_active_boundary",
    ]


class MetadataRetrievalContext(BaseModel):
    interpretation: QueryInterpretation
    backend: RetrievalBackendInfo
    rounds: list[MetadataRetrievalRound] = Field(default_factory=list)
    final_candidates: list[OntologyEntityCandidate] = Field(default_factory=list)
    final_relations: list[OntologyRelation] = Field(default_factory=list)
    final_paths: list[ResolvedSemanticPath] = Field(default_factory=list)
    final_verification: BackboneVerification
    retry_count: int = Field(default=0, ge=0)
    max_rounds: int = Field(ge=1)
    termination_reason: str
    reference_profile_ids: list[str] = Field(default_factory=list)
    reference_signature_terms: list[str] = Field(default_factory=list)
    reference_signature_path_ids: list[str] = Field(default_factory=list)


class RetrievalStep(BaseModel):
    step_id: str
    protocol: Literal["CKG_TEMPLATE", "REFERENCE_HYBRID", "REFERENCE_EXACT", "INTEGRATION"]
    target: str
    template_id: str | None = None
    bindings: dict[str, Any] = Field(default_factory=dict)
    category_filter: str | None = None
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class ReaderRetrievalPlan(BaseModel):
    query: str
    steps: list[RetrievalStep]
    target_databases: list[str]
    uses_ckg_template_path: bool
    uses_reference_hybrid_path: bool


class CompositeDataContext(BaseModel):
    baseline_facts: dict[str, Any] = Field(default_factory=dict)
    reference_parameters: dict[str, Any] = Field(default_factory=dict)
    retrieval_plan: ReaderRetrievalPlan | None = None
    executed_templates: list[dict[str, Any]] = Field(default_factory=list)
    reference_searches: list[dict[str, Any]] = Field(default_factory=list)
    ontology_entities: list[str] = Field(default_factory=list)
    ontology_relations: list[str] = Field(default_factory=list)
    semantic_path_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    complete: bool = False
    missing_context: list[str] = Field(default_factory=list)


class ReferenceDemand(BaseModel):
    demand_id: str
    category: Literal["CPD", "EFD", "TRANSPORT", "DENSITY", "OTHER"]
    query_text: str
    target: str
    required_fields: list[str] = Field(default_factory=list)
    resolved: bool = False
    resolution: dict[str, Any] = Field(default_factory=dict)


class DeltaOverride(BaseModel):
    override_id: str
    change_type: str
    target: str
    target_uri: str | None = None
    predicate_uri: str | None = None
    operator: Literal["replace", "multiply", "add", "subtract", "substitute", "sweep"]
    value: Any
    unit: str | None = None
    precedence: Literal["explicit_user", "resolved_reference", "governed_default"]
    provenance: dict[str, Any] = Field(default_factory=dict)


class DeltaGraphSpecification(BaseModel):
    delta_id: str
    mode: QueryMode
    persistence: PersistenceMode
    overrides: list[DeltaOverride] = Field(default_factory=list)
    reference_demands: list[ReferenceDemand] = Field(default_factory=list)
    ready: bool
    feasibility_status: Literal["feasible", "incomplete", "forbidden"]
    issues: list[ReadinessIssue] = Field(default_factory=list)


class CalculationProgramStep(BaseModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)
    expected_unit: str | None = None
    rationale: str


class CalculationStateTransition(BaseModel):
    step_id: str
    tool_name: str
    input_state_digest: str
    output_state_digest: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"]
    error: str | None = None


class CalculationProgram(BaseModel):
    program_id: str
    initial_context_digest: str
    steps: list[CalculationProgramStep]
    transitions: list[CalculationStateTransition] = Field(default_factory=list)
    final_result: dict[str, Any] = Field(default_factory=dict)
    complete: bool = False


class EpistemicValidation(BaseModel):
    passed: bool
    metadata_sufficient: bool
    baseline_context_complete: bool
    delta_complete: bool
    calculation_complete: bool
    query_answer_alignment: bool
    blocking_issue_codes: list[str] = Field(default_factory=list)
    missing_variables: list[str] = Field(default_factory=list)
    clarification_request: str | None = None
    write_confirmation_required: bool = False


class TaskControlBlock(BaseModel):
    mode: QueryMode
    selected_plan_id: str
    tdg: dict[str, Any]
    pending_agents: list[str] = Field(default_factory=list)
    current_agent: str | None = None
    completed_agents: list[str] = Field(default_factory=list)
    plan_candidates: list[dict[str, Any]] = Field(default_factory=list)


class SharedGlobalSpace(BaseModel):
    """Five-category blackboard shown in Fig. 13 of the manuscript."""

    task_control_block: TaskControlBlock | None = None
    entity_pointer_registry: dict[str, Any] = Field(default_factory=dict)
    baseline_factual_context: dict[str, Any] = Field(default_factory=dict)
    transient_delta_specifications: dict[str, Any] = Field(default_factory=dict)
    analytical_metrics_and_provenance: dict[str, Any] = Field(default_factory=dict)
