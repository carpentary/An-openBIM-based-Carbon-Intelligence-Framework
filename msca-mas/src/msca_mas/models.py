from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class QueryMode(StrEnum):
    INFORMATION_RETRIEVAL = "IR"
    KNOWLEDGE_CORRECTION = "KC"
    STRATEGIC_ANALYSIS = "SA"


class AgentName(StrEnum):
    ORCHESTRATOR = "orchestrator"
    METADATA = "metadata"
    EXTRACTOR = "extractor"
    READER = "reader"
    CALCULATOR = "calculator"
    SYNTHESIS = "synthesis"
    SINGLE_AGENT_BASELINE = "single_agent_baseline"


class RunStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    PENDING_CONFIRMATION = "pending_confirmation"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class PersistenceMode(StrEnum):
    TRANSIENT = "transient"
    PERSISTENT = "persistent"


class ToolCallStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class TaskNode(BaseModel):
    node_id: str
    agent: AgentName
    instruction: str
    operation: str = "unspecified_operation"
    dependencies: list[str] = Field(default_factory=list)
    anchor_terms: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    execution_constraints: list[str] = Field(default_factory=list)


class TaskDependencyGraph(BaseModel):
    mode: QueryMode
    tasks: list[TaskNode]
    rationale_summary: str

    @model_validator(mode="after")
    def validate_dag(self) -> "TaskDependencyGraph":
        ids = [task.node_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task node IDs must be unique")
        id_set = set(ids)
        for task in self.tasks:
            missing = set(task.dependencies) - id_set
            if missing:
                raise ValueError(f"{task.node_id} has unknown dependencies: {sorted(missing)}")

        edges = {task.node_id: set(task.dependencies) for task in self.tasks}
        pending = set(ids)
        resolved: set[str] = set()
        while pending:
            ready = {node for node in pending if edges[node] <= resolved}
            if not ready:
                raise ValueError("task dependency graph contains a cycle")
            resolved.update(ready)
            pending -= ready
        return self

    def execution_order(self) -> list[AgentName]:
        pending = {task.node_id: task for task in self.tasks}
        done: set[str] = set()
        ordered: list[AgentName] = []
        while pending:
            ready = [task for task in pending.values() if set(task.dependencies) <= done]
            ready.sort(key=lambda task: task.node_id)
            for task in ready:
                ordered.append(task.agent)
                done.add(task.node_id)
                pending.pop(task.node_id)
        return ordered


class PlanCandidate(BaseModel):
    plan_id: str
    strategy: str
    agents: list[AgentName]
    tool_focus: dict[str, list[str]] = Field(default_factory=dict)
    valid: bool
    constraint_violations: list[str] = Field(default_factory=list)
    score_components: dict[str, float] = Field(default_factory=dict)
    total_score: float = Field(ge=0.0, le=1.0)
    selected: bool = False


class CQMatch(BaseModel):
    cq_id: str
    group: str
    question_en: str
    question_zh: str
    semantic_path: str
    support_status: str
    score: float = Field(ge=0.0, le=1.0)
    path_valid: bool = True


class EntityPointer(BaseModel):
    mention: str
    uri: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class ReadinessIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    missing_fields: list[str] = Field(default_factory=list)
    source: str = "system"


class ScenarioChange(BaseModel):
    change_type: str
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    explicit_parameters: list[str] = Field(default_factory=list)
    assumed_parameters: list[str] = Field(default_factory=list)


class ScenarioOverlay(BaseModel):
    overlay_id: str
    persistence: PersistenceMode
    changes: list[ScenarioChange]
    ready: bool
    issues: list[ReadinessIssue] = Field(default_factory=list)


class ProvenanceEvent(BaseModel):
    sequence: int
    agent: AgentName
    action: str
    sources: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    call_id: str
    sequence: int = Field(ge=1)
    agent: AgentName
    tool_name: str
    arguments: dict[str, Any]
    status: ToolCallStatus
    result: Any = None
    result_digest: str | None = None
    error: str | None = None
    # Internal tool crashes use TOOL_CALL_CRASH; domain failures use a separate status.
    error_kind: Literal["domain", "crash"] | None = None
    duration_seconds: float = Field(ge=0.0)


class FinalReport(BaseModel):
    status: RunStatus
    mode: QueryMode
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    issues: list[ReadinessIssue] = Field(default_factory=list)
    execution_process: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceEvent] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    requires_human_confirmation: bool = False
    transaction_preview: dict[str, Any] | None = None


class SharedGlobalState(TypedDict, total=False):
    query: str
    mode: str
    query_contract: dict[str, Any]
    shared_global_space: dict[str, Any]
    tdg: dict[str, Any]
    pending_agents: list[str]
    analysis_required: bool
    planning_candidates: list[dict[str, Any]]
    selected_plan_id: str
    cq_matches: list[dict[str, Any]]
    entity_pointers: list[dict[str, Any]]
    ontology_context: dict[str, Any]
    metadata_retrieval: dict[str, Any]
    metadata_rounds: list[dict[str, Any]]
    retrieved_context: dict[str, Any]
    composite_data_context: dict[str, Any]
    scenario_overlay: dict[str, Any]
    delta_graph_specification: dict[str, Any]
    calculation_results: dict[str, Any]
    calculation_program: dict[str, Any]
    epistemic_validation: dict[str, Any]
    issues: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    metrics: dict[str, Any]
    final_report: dict[str, Any]


AgentUpdate = dict[str, Any]
