from __future__ import annotations

import copy
import json
import os
import ssl
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .intent import GovernedIntentSignals, analyse_query_intent
from .models import QueryMode
from .util import atomic_write_json


class PlanningIntentEvidence(BaseModel):
    """Answer-free semantic evidence returned with a planning proposal."""

    model_config = ConfigDict(extra="forbid")

    request_kind: Literal[
        "read_existing",
        "derive_from_existing",
        "hypothetical_analysis",
        "persistent_change",
        "boundary_validation",
        "ambiguous",
    ] = "ambiguous"
    existing_data_requested: bool = False
    calculation_requested: bool = False
    counterfactual_requested: bool = False
    persistence_requested: bool = False
    active_boundary_status: Literal[
        "inside_active_boundary",
        "outside_active_boundary",
        "undetermined",
    ] = "undetermined"
    missing_parameter_categories: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PlanningParameterMention(BaseModel):
    """A query-grounded parameter span with a semantic value role.

    The model is not allowed to invent or normalise an engineering value here.
    It identifies the exact surface span and its role; deterministic tools later
    verify that the span occurs in the query and parse its number/unit.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "mass",
        "distance_km",
        "factor_kgco2e_per_tkm",
        "substitution_fraction",
        "baseline_factor",
        "alternative_factor",
        "reduction_fraction",
        "reuse_cycles",
        "baseline_emission",
        "parameter_value",
        "threshold_fraction",
    ]
    role: Literal[
        "proposed",
        "current",
        "baseline",
        "alternative",
        "candidate",
        "threshold",
        "context",
        "reference",
    ]
    surface_form: str = Field(min_length=1, max_length=120)
    unit: str | None = Field(default=None, max_length=48)
    target: str | None = Field(default=None, max_length=96)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PlanningSemanticRisk(BaseModel):
    """An explicit query-grounded condition that a downstream safety gate must verify."""

    model_config = ConfigDict(extra="forbid")

    risk: Literal[
        "outside_assessment_boundary",
        "unresolved_or_unverified_reference",
        "semantic_calculation_precondition_violation",
        "invalid_physical_parameter",
        "ambiguous_persistent_correction",
        "unsupported_persistent_operation",
        "contradictory_governed_premise",
        "unverified_dimensional_mapping",
        "missing_persistent_parameter",
        "confirmation_gate_bypass",
    ]
    surface_form: str = Field(min_length=1, max_length=180)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PlanningProposal(BaseModel):
    """Minimal structured output accepted from an optional reasoning model."""

    model_config = ConfigDict(extra="forbid")

    mode: QueryMode
    rationale_summary: str = Field(min_length=1, max_length=500)
    task_family: Literal[
        "retrieval",
        "retrieval_calculation",
        "ontology_design",
        "scenario",
        "compound_scenario",
        "correction_preview",
        "correction_impact_preview",
    ] | None = None
    operation_hints: list[
        Literal[
            "retrieve_overview",
            "retrieve_resource",
            "retrieve_elements",
            "retrieve_work_packages",
            "retrieve_cpd_efd",
            "retrieve_transport",
            "retrieve_ifc_inventory",
            "aggregate",
            "compare",
            "rank",
            "share",
            "intensity",
            "trace",
            "sensitivity",
            "factor_reduction",
            "material_substitution",
            "transport_emission",
            "reuse_allocation",
            "post_calculation_retrieval",
            "transaction_preview",
            "return_subontology",
            "boundary_validation",
        ]
    ] = Field(default_factory=list, max_length=16)
    semantic_anchors: list[str] = Field(default_factory=list, max_length=16)
    primary_operator: Literal[
        "none",
        "aggregate",
        "compare",
        "rank",
        "share",
        "intensity",
        "trace",
        "sensitivity",
        "factor_reduction",
        "material_substitution",
        "transport",
        "reuse_allocation",
        "transaction_preview",
    ] = "none"
    target_scopes: list[
        Literal["project", "main_structure", "enclosure_structure", "foundation_pit"]
    ] = Field(default_factory=list, max_length=4)
    target_stages: list[
        Literal[
            "A1-A3_ProductStage",
            "A4_TransportationStage",
            "A5_ConstructionStage",
        ]
    ] = Field(default_factory=list, max_length=3)
    resource_targets: list[str] = Field(default_factory=list, max_length=12)
    explicit_identifiers: list[str] = Field(default_factory=list, max_length=12)
    ontology_profile_hints: list[
        Literal[
            "ONT-PROJECT-CARBON",
            "ONT-SYSTEM-CARBON",
            "ONT-ENCLOSURE-CPD",
            "ONT-FOUNDATION-EARTHWORK",
            "ONT-A4-TRANSPORT",
            "ONT-SEPRC-PROVENANCE",
            "ONT-SCENARIO",
            "ONT-SCOPE-QUALITY",
        ]
    ] = Field(default_factory=list, max_length=8)
    reference_categories: list[Literal["CPD", "EFD", "TRANSPORT"]] = Field(
        default_factory=list,
        max_length=3,
    )
    required_capabilities: list[
        Literal[
            "metadata_retrieval",
            "typed_extraction",
            "governed_reader",
            "deterministic_calculation",
            "post_calculation_retrieval",
            "transaction_preview",
            "synthesis",
        ]
    ] = Field(default_factory=list, max_length=7)
    compound: bool = False
    compound_template: Literal[
        "sensitivity_argmin_post_retrieval",
        "substitution_rank_project_delta",
        "reuse_reduction_composition",
        "multi_transport_rank_post_retrieval",
        "dual_reduction_threshold",
        "generic_dependency_ordered_program",
    ] | None = None
    requires_calculator: bool = False
    requires_post_calculation_retrieval: bool = False
    meta_ontology_query: bool = False
    correction_impact_preview: bool = False
    parameter_mentions: list[PlanningParameterMention] = Field(default_factory=list, max_length=32)
    semantic_risks: list[PlanningSemanticRisk] = Field(default_factory=list, max_length=8)
    intent_evidence: PlanningIntentEvidence = Field(default_factory=PlanningIntentEvidence)


class PlanningDecision(PlanningProposal):
    route_source: str
    structured_output_attempts: int = Field(default=1, ge=1)
    transport_request_attempts: int = Field(default=1, ge=1)
    transport_retry_count: int = Field(default=0, ge=0)
    model_proposed_mode: QueryMode | None = None
    planning_arbitration: str = "not_applicable"
    rule_signal_mode: QueryMode | None = None
    arbitration_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    arbitration_evidence: list[str] = Field(default_factory=list)
    api_usage: dict[str, int] = Field(default_factory=dict)
    cumulative_api_usage: dict[str, int] = Field(default_factory=dict)
    token_budget: int | None = Field(default=None, ge=1)


def _normalise_schema_bounded_planning_text(payload: Any) -> Any:
    """Drop unknown fields and trim lists and text to the proposal schema limits."""

    if not isinstance(payload, dict):
        return payload
    normalised = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key in PlanningProposal.model_fields
    }

    list_limits = {
        "operation_hints": 16,
        "semantic_anchors": 16,
        "target_scopes": 4,
        "target_stages": 3,
        "resource_targets": 12,
        "explicit_identifiers": 12,
        "ontology_profile_hints": 8,
        "reference_categories": 3,
        "required_capabilities": 7,
        "parameter_mentions": 32,
        "semantic_risks": 8,
    }
    for field, maximum in list_limits.items():
        value = normalised.get(field)
        if isinstance(value, list) and len(value) > maximum:
            normalised[field] = value[:maximum]

    rationale = normalised.get("rationale_summary")
    if isinstance(rationale, str) and len(rationale) > 500:
        normalised["rationale_summary"] = rationale[:500].rstrip()

    parameter_mentions = normalised.get("parameter_mentions")
    if isinstance(parameter_mentions, list):
        for index, mention in enumerate(parameter_mentions):
            if not isinstance(mention, dict):
                continue
            mention = {
                key: value
                for key, value in mention.items()
                if key in PlanningParameterMention.model_fields
            }
            parameter_mentions[index] = mention
            for field, maximum in (("surface_form", 120), ("unit", 48), ("target", 96)):
                value = mention.get(field)
                if isinstance(value, str) and len(value) > maximum:
                    mention[field] = value[:maximum].rstrip()

    semantic_risks = normalised.get("semantic_risks")
    if isinstance(semantic_risks, list):
        for index, risk in enumerate(semantic_risks):
            if not isinstance(risk, dict):
                continue
            risk = {
                key: value
                for key, value in risk.items()
                if key in PlanningSemanticRisk.model_fields
            }
            semantic_risks[index] = risk
            surface = risk.get("surface_form")
            if isinstance(surface, str) and len(surface) > 180:
                risk["surface_form"] = surface[:180].rstrip()

    intent = normalised.get("intent_evidence")
    if isinstance(intent, dict):
        intent = {
            key: value
            for key, value in intent.items()
            if key in PlanningIntentEvidence.model_fields
        }
        missing = intent.get("missing_parameter_categories")
        if isinstance(missing, list) and len(missing) > 12:
            intent["missing_parameter_categories"] = missing[:12]
        normalised["intent_evidence"] = intent
    return normalised


class PlanningReasoner(Protocol):
    def decide(self, query: str, deterministic_default: QueryMode) -> PlanningDecision: ...


class CallbackPlanningReasoner:
    """Vendor-neutral LLM adapter boundary.

    The host application owns networking, credentials, model pinning and timeout
    policy. This project sends only a compact schema request and accepts no free-
    form chain of thought, SPARQL, arithmetic result or graph mutation from the
    callback.
    """

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        max_retries: int = 2,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.callback = callback
        self.max_retries = max_retries

    def decide(self, query: str, deterministic_default: QueryMode) -> PlanningDecision:
        request = {
            "task": (
                "Classify the requested operation as IR, KC or SA; identify its non-numeric "
                "task family; return semantic intent evidence and whitelisted operation hints."
            ),
            "query": query,
            "classification_policy": {
                "IR": "read, trace, aggregate, compare or rank existing governed data",
                "SA": "transient hypothetical, counterfactual, sensitivity or new-parameter analysis",
                "KC": "explicit persistent add, update, correction or deletion request",
                "boundary": "out-of-boundary operations are validation requests, not executable scenarios",
            },
            "semantic_contract_policy": {
                "focal_operation": (
                    "Infer the user's actual decision operation from the whole scenario; do not route "
                    "from isolated topical words or surrounding audit constraints. Except for an "
                    "operation outside the declared assessment boundary, classify an invalid, unsafe "
                    "or incomplete request by the operation it attempted: a transient design option "
                    "remains SA and a persistent correction remains KC; semantic_risks control refusal."
                ),
                "parameter_roles": (
                    "For every relevant number, quote its exact surface span and distinguish proposed, "
                    "current, baseline, alternative, candidate, threshold, context and reference roles."
                ),
                "grounding": (
                    "Never invent a span, value, identifier, scope, stage or resource. Every explicit "
                    "identifier and parameter surface_form must occur verbatim in the query."
                ),
                "decomposition": (
                    "Select required capabilities and preserve retrieve-calculate-retrieve dependencies "
                    "when the requested conclusion depends on a calculated intermediate."
                ),
                "safety_evidence": (
                    "Report a semantic_risk only when the query explicitly states the problem, and quote "
                    "an exact supporting surface_form. Relevant conditions are an excluded assessment "
                    "boundary; an identity/source/reference that has not been verified; an instruction to "
                    "double count or combine semantically incompatible quantities; a non-physical input; "
                    "a contradictory governed premise; incompatible factor dimensions without a verified "
                    "conversion; an ambiguous persistent correction; a missing value required by a supported "
                    "persistent template; or a persistent operation that deletes history or bypasses confirmation."
                ),
                "risk_distinctions": (
                    "Use contradictory_governed_premise when a stated ordering or premise conflicts with the "
                    "governed facts, invalid_physical_parameter for a non-physical magnitude or sign, "
                    "unverified_dimensional_mapping only when one factor definition is substituted for another "
                    "without a verified conversion; direct arithmetic over incompatible units is a "
                    "semantic_calculation_precondition_violation, "
                    "missing_persistent_parameter when an otherwise supported correction omits its new value, and "
                    "confirmation_gate_bypass when direct application is requested without the required preview "
                    "and exact-digest confirmation. Use unresolved_or_unverified_reference for missing identity, "
                    "source or provenance. Report the most specific root condition rather than derivative symptoms."
                ),
                "supported_persistent_template": (
                    "The current confirmation-gated transaction template supports one append-only change "
                    "to one material A4 default distance. Persistent factor, unit, ontology/schema, IFC "
                    "quantity or history-deletion requests are unsupported operations; do not misclassify "
                    "missing provenance fields for those operations as target ambiguity."
                ),
            },
            "ontology_profile_catalogue": {
                "ONT-PROJECT-CARBON": "project lifecycle-stage totals and assessment context",
                "ONT-SYSTEM-CARBON": "main/enclosure/foundation system carbon partition",
                "ONT-ENCLOSURE-CPD": "enclosure IFC element, work-package and CPD mapping",
                "ONT-FOUNDATION-EARTHWORK": "foundation-pit geometry, earthwork and process carbon",
                "ONT-A4-TRANSPORT": "material transport mass, distance, vehicle, factor and A4 emission",
                "ONT-SEPRC-PROVENANCE": "System-Element-Process-Resource-Carbon trace and provenance",
                "ONT-SCENARIO": "transient override and scenario result semantics",
                "ONT-SCOPE-QUALITY": "assessment boundary, missing-input and quality constraints",
            },
            "capability_catalogue": {
                "metadata_retrieval": "query-specific ontology, relation and semantic-path retrieval",
                "typed_extraction": "role-aware extraction of scenario/correction parameters",
                "governed_reader": "CKG templates and filtered CPD/EFD/transport reference retrieval",
                "deterministic_calculation": "whitelisted arithmetic, comparison, ranking or sensitivity functions",
                "post_calculation_retrieval": "retrieve evidence selected by an intermediate calculation",
                "transaction_preview": "append-only, confirmation-gated correction preview",
                "synthesis": "grounded completion, refusal or clarification",
            },
            "routing_examples": [
                {
                    "request": "Compare two stored lifecycle-stage totals.",
                    "mode": "IR",
                    "request_kind": "derive_from_existing",
                },
                {
                    "request": "Run a new sensitivity sweep for a design parameter.",
                    "mode": "SA",
                    "request_kind": "hypothetical_analysis",
                },
                {
                    "request": "Return the sensitivity results already recorded in the project.",
                    "mode": "IR",
                    "request_kind": "read_existing",
                },
                {
                    "request": "Change a named default in the CKG and persist it.",
                    "mode": "KC",
                    "request_kind": "persistent_change",
                },
                {
                    "request": "Prepare an append-only correction preview for a future default; leave the graph unchanged until confirmation.",
                    "mode": "KC",
                    "request_kind": "persistent_change",
                },
                {
                    "request": "Compare a proposed default with its current value, calculate its multiplier, then prepare a confirmation-gated preview.",
                    "mode": "KC",
                    "request_kind": "persistent_change",
                    "task_family": "correction_impact_preview",
                },
                {
                    "request": "A team is considering a revised haul distance; compare it with the current default and prepare the proposed change for approval without applying it yet.",
                    "mode": "KC",
                    "request_kind": "persistent_change",
                    "task_family": "correction_impact_preview",
                },
                {
                    "request": "Evaluate a temporary delivery option from explicit mass, distance and factor inputs without revising the governed default.",
                    "mode": "SA",
                    "request_kind": "hypothetical_analysis",
                },
                {
                    "request": "Calculate an excluded lifecycle module.",
                    "mode": "IR",
                    "request_kind": "boundary_validation",
                },
                {
                    "request": "Which query-specific subontology supports a hypothetical scenario?",
                    "mode": "IR",
                    "request_kind": "read_existing",
                },
            ],
            "output_schema": PlanningProposal.model_json_schema(),
            "forbidden_outputs": [
                "chain_of_thought",
                "SPARQL execution result",
                "carbon arithmetic result",
                "graph mutation",
                "a parameter value not quoted from the query",
            ],
        }
        proposal: PlanningProposal | None = None
        last_error: Exception | None = None
        attempts = 0
        for attempts in range(1, self.max_retries + 2):
            try:
                proposal = PlanningProposal.model_validate(
                    _normalise_schema_bounded_planning_text(self.callback(request))
                )
                break
            except (ValidationError, TypeError, ValueError) as exc:
                last_error = exc
                if attempts > self.max_retries:
                    raise ValueError(
                        f"structured planning output remained invalid after {attempts} attempts"
                    ) from exc
                request = {
                    **request,
                    "repair_attempt": attempts,
                    "validation_error": str(exc),
                    "repair_instruction": (
                        "Return only an object conforming exactly to output_schema; "
                        "do not add fields or prose."
                    ),
                }
        if proposal is None:  # pragma: no cover - loop either validates or raises
            raise RuntimeError("planning proposal is unavailable") from last_error
        return PlanningDecision(
            **proposal.model_dump(mode="python"),
            route_source="validated_structured_callback",
            structured_output_attempts=attempts,
            model_proposed_mode=proposal.mode,
            planning_arbitration="validated_callback_proposal",
        )


class DeepSeekAPIError(RuntimeError):
    """Raised when the governed DeepSeek planning boundary cannot return JSON."""


class DeepSeekTokenBudgetExceeded(DeepSeekAPIError):
    """Raised before a request that could exceed the configured API-token cap."""


class DeepSeekPlanningReasoner:
    """DeepSeek V4 Flash adapter with thinking mode permanently disabled.

    The model is limited to the compact IR/KC/SA planning proposal. It never
    receives authority to execute tools, SPARQL, calculations or graph writes.
    Every request explicitly sends ``thinking.type=disabled`` because the
    provider default is thinking-enabled.
    """

    provider = "deepseek"
    required_model = "deepseek-v4-flash"
    thinking_type = "disabled"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = required_model,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        transport_max_retries: int = 3,
        transport_backoff_seconds: float = 1.0,
        token_budget: int = 3_000_000,
        usage_ledger: str | Path | None = None,
        ontology_enabled: bool = True,
    ):
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY must not be empty")
        if model != self.required_model:
            raise ValueError(
                f"this release pins the planning model to {self.required_model}; received {model}"
            )
        if timeout_seconds <= 0:
            raise ValueError("DeepSeek timeout must be positive")
        if transport_max_retries < 0:
            raise ValueError("DeepSeek transport retries must be non-negative")
        if transport_backoff_seconds < 0:
            raise ValueError("DeepSeek transport backoff must be non-negative")
        if token_budget <= 0:
            raise ValueError("DeepSeek token budget must be positive")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.transport_max_retries = int(transport_max_retries)
        self.transport_backoff_seconds = float(transport_backoff_seconds)
        self.token_budget = int(token_budget)
        self.usage_ledger = Path(usage_ledger).expanduser().resolve() if usage_ledger else None
        self.ontology_enabled = bool(ontology_enabled)
        self._decision_usage = self._empty_usage()
        self._decision_transport_attempts = 0
        self._decision_transport_retries = 0
        self._cumulative_usage = self._load_usage_ledger()
        self._validated_reasoner = CallbackPlanningReasoner(
            self._request_proposal,
            max_retries=max_retries,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def configuration_summary(self) -> dict[str, Any]:
        """Return non-secret deployment metadata suitable for logs and reports."""

        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "thinking": {"type": self.thinking_type},
            "structured_output": True,
            "planning_arbitration": "semantic_evidence_with_typed_operation_and_safety_vetoes",
            "timeout_seconds": self.timeout_seconds,
            "transport_retry_policy": {
                "max_retries": self.transport_max_retries,
                "initial_backoff_seconds": self.transport_backoff_seconds,
                "retryable_http_statuses": [408, 429, 500, 502, 503, 504],
            },
            "token_budget": self.token_budget,
            "cumulative_api_usage": dict(self._cumulative_usage),
            "remaining_tokens": max(
                0, self.token_budget - self._cumulative_usage["total_tokens"]
            ),
            "api_key_configured": True,
        }

    def decide(self, query: str, deterministic_default: QueryMode) -> PlanningDecision:
        self._decision_usage = self._empty_usage()
        self._decision_transport_attempts = 0
        self._decision_transport_retries = 0
        decision = self._validated_reasoner.decide(query, deterministic_default)
        proposed_mode = decision.mode
        signals = analyse_query_intent(query)
        final_mode, arbitration, arbitration_evidence, confidence = self._arbitrate(
            proposed_mode,
            decision.intent_evidence,
            signals,
        )
        rationale_summary = (
            f"{arbitration}: selected {final_mode.value} from semantic intent evidence and "
            f"independent safety signals. {decision.rationale_summary}"
        )[:500]
        return decision.model_copy(
            update={
                # Veto or adjust a proposed route only for scope, persistence, or evidence conflicts.
                "mode": final_mode,
                "rationale_summary": rationale_summary,
                "route_source": "deepseek_structured_planning",
                "model_proposed_mode": proposed_mode,
                "planning_arbitration": arbitration,
                "rule_signal_mode": signals.rule_mode,
                "arbitration_confidence": confidence,
                "arbitration_evidence": arbitration_evidence,
                "api_usage": dict(self._decision_usage),
                "transport_request_attempts": self._decision_transport_attempts,
                "transport_retry_count": self._decision_transport_retries,
                "cumulative_api_usage": dict(self._cumulative_usage),
                "token_budget": self.token_budget,
            }
        )

    @staticmethod
    def _arbitrate(
        proposed_mode: QueryMode,
        model_evidence: PlanningIntentEvidence,
        rule_signals: GovernedIntentSignals,
    ) -> tuple[QueryMode, str, list[str], float]:
        """Reconcile independent semantic signals without benchmark-specific rules."""

        evidence = list(rule_signals.evidence)
        model_confident = model_evidence.confidence >= 0.60

        if rule_signals.persistence_requested:
            final = QueryMode.KNOWLEDGE_CORRECTION
            arbitration = (
                "semantic_proposal_accepted"
                if proposed_mode == final
                else "hard_persistence_safety_veto"
            )
            return final, arbitration, evidence, rule_signals.confidence

        if rule_signals.out_of_boundary_operation_requested:
            final = QueryMode.INFORMATION_RETRIEVAL
            arbitration = (
                "semantic_proposal_accepted"
                if proposed_mode == final
                else "hard_boundary_safety_veto"
            )
            return final, arbitration, evidence, rule_signals.confidence

        if rule_signals.ontology_design_information_request:
            final = QueryMode.INFORMATION_RETRIEVAL
            arbitration = (
                "semantic_proposal_accepted"
                if proposed_mode == final
                else "hard_ontology_information_veto"
            )
            return final, arbitration, evidence, rule_signals.confidence

        if model_confident:
            if (
                model_evidence.persistence_requested
                or model_evidence.request_kind == "persistent_change"
            ):
                final = QueryMode.KNOWLEDGE_CORRECTION
                evidence.append("model_semantic_persistence_evidence")
            elif (
                model_evidence.active_boundary_status == "outside_active_boundary"
                and not model_evidence.persistence_requested
            ):
                final = QueryMode.INFORMATION_RETRIEVAL
                evidence.append("model_semantic_boundary_evidence")
            elif (
                model_evidence.counterfactual_requested
                or model_evidence.request_kind == "hypothetical_analysis"
            ):
                final = QueryMode.STRATEGIC_ANALYSIS
                evidence.append("model_semantic_counterfactual_evidence")
            elif model_evidence.request_kind in {
                "read_existing",
                "derive_from_existing",
                "boundary_validation",
            }:
                final = QueryMode.INFORMATION_RETRIEVAL
                evidence.append("model_semantic_existing_data_evidence")
            else:
                final = proposed_mode
        elif proposed_mode == QueryMode.KNOWLEDGE_CORRECTION:
            # Require an independent persistence signal before routing to a correction.
            final = rule_signals.rule_mode
            evidence.append("unsubstantiated_persistence_route_rejected")
        else:
            final = proposed_mode

        arbitration = (
            "semantic_proposal_accepted"
            if final == proposed_mode
            else "semantic_evidence_consistency_repair"
        )
        confidence = max(rule_signals.confidence, model_evidence.confidence)
        return final, arbitration, list(dict.fromkeys(evidence)), confidence

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            # Track cached and uncached prompt tokens separately.
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }

    def _load_usage_ledger(self) -> dict[str, int]:
        usage = self._empty_usage()
        if self.usage_ledger is None or not self.usage_ledger.exists():
            return usage
        try:
            payload = json.loads(self.usage_ledger.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid DeepSeek usage ledger: {self.usage_ledger}") from exc
        if payload.get("provider") not in {None, self.provider}:
            raise ValueError("DeepSeek usage ledger provider does not match this adapter")
        if payload.get("model") not in {None, self.model}:
            raise ValueError("DeepSeek usage ledger model does not match this adapter")
        for key in usage:
            value = payload.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid DeepSeek usage ledger field: {key}")
            usage[key] = value
        return usage

    def _record_usage(self, provider_usage: Any) -> None:
        normalised = self._empty_usage()
        if isinstance(provider_usage, dict):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                value = provider_usage.get(key, 0)
                if isinstance(value, int) and value >= 0:
                    normalised[key] = value
        if normalised["total_tokens"] == 0:
            normalised["total_tokens"] = (
                normalised["prompt_tokens"] + normalised["completion_tokens"]
            )
        normalised["request_count"] = 1
        for key, value in normalised.items():
            self._decision_usage[key] += value
            self._cumulative_usage[key] += value
        if self.usage_ledger is not None:
            atomic_write_json(
                self.usage_ledger,
                {
                    "provider": self.provider,
                    "model": self.model,
                    "token_budget": self.token_budget,
                    **self._cumulative_usage,
                },
            )

    def _request_proposal(self, planning_request: dict[str, Any]) -> dict[str, Any]:
        # Do not send ontology profile and capability catalogues in the schema-only arm.
        if not self.ontology_enabled:
            planning_request = {
                key: value
                for key, value in planning_request.items()
                if key not in {"ontology_profile_catalogue", "capability_catalogue"}
            }
            planning_request["ontology_availability"] = (
                "none: this system runs without an ontology; do not reference "
                "ontology profiles, schema classes or S-E-P-R-C terms"
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You plan queries for an ontology-constrained construction-carbon MAS. "
                        "Return one JSON object conforming exactly to the supplied output schema. "
                        "Classify the operation, not the topic. IR covers retrieval, trace, audit, "
                        "aggregation, comparison, ranking, or calculation over existing governed "
                        "facts, including retrieval of an already stored sensitivity result. SA "
                        "covers a new transient hypothetical, design alternative, parameter sweep, "
                        "substitution, reduction, or calculation using proposed parameters. KC "
                        "requires an explicit request to persist, add, update, correct, or delete a "
                        "graph fact/default; it will only produce a confirmation-gated preview. A "
                        "comparison alone is not SA: distinguish existing values from hypothetical "
                        "alternatives. A request involving modules outside A1-A3, A4 and A5 is a "
                        "boundary-validation route and must not become an executable SA merely "
                        "because it says calculate. In long project narratives, identify the focal "
                        "decision operation and treat surrounding audit, provenance and non-mutation "
                        "constraints as constraints rather than as the requested calculation. A "
                        "temporary option remains SA even when its premise, units or inputs are invalid; "
                        "preserve the attempted SA/KC/IR operation and express the defect through "
                        "semantic_risks. Only the excluded-boundary special case is routed to IR validation. "
                        "Direct arithmetic over values already displayed, recorded or contained in a project "
                        "dashboard remains IR even if the requested output is described as a revised total; "
                        "the word revised alone does not create a hypothetical scenario. "
                        "A temporary option with proposed numeric parameters is SA even when it mentions "
                        "a governed default that must remain unchanged. An append-only correction or "
                        "transaction preview intended for future lookups is KC even though no graph is "
                        "written before confirmation. Asking to compare a proposed default with its current "
                        "value and prepare that proposed change for approval is KC, even if the proposal is "
                        "described as being considered and must not be applied yet; asking only to evaluate "
                        "an option while explicitly retaining the current default is SA. Distinguish proposed, current, reference and "
                        "context-only values; report ambiguity rather than collapsing their roles. "
                        "Populate the complete semantic contract consistently with mode: task_family, "
                        "primary_operator, scopes, stages, resources, ontology profiles, reference "
                        "categories, required capabilities and dependency flags. For each relevant "
                        "number, return a parameter_mention with its exact surface_form from the query "
                        "and its semantic role; never invent or normalise a value. Select only enumerated "
                        "fields and return concise semantic_anchors containing query nouns/verbs or "
                        "schema concepts. Distinguish a KC correction_preview, which only prepares a "
                        "new value for confirmation, from correction_impact_preview, which must first "
                        "compare the current and proposed values or calculate a delta, ratio, multiplier "
                        "or impact; the latter requires deterministic_calculation. Identify explicit "
                        "semantic risks using semantic_risks and quote their exact query evidence. Do not "
                        "mark ordinary requests for audit evidence as risky: use a risk only when the query "
                        "itself states that an identity/source is unresolved, quantities are being double "
                        "counted or combined incompatibly, a physical input is invalid, a correction is "
                        "ambiguous, or a requested write is outside the supported template. Distinguish the root "
                        "condition: a stated premise that contradicts governed facts is "
                        "contradictory_governed_premise; a non-physical sign or magnitude is "
                        "invalid_physical_parameter; substituting one factor definition for another without a "
                        "verified conversion is unverified_dimensional_mapping, while adding or otherwise directly "
                        "combining incompatible units is semantic_calculation_precondition_violation; a supported correction with no proposed value is "
                        "missing_persistent_parameter; and a request to write directly while skipping preview or "
                        "confirmation is confirmation_gate_bypass. Missing identity, source or provenance is "
                        "unresolved_or_unverified_reference. Return the most specific root risk, not every downstream "
                        "symptom. Asking which evidence subset is needed to support a hypothetical operation is an "
                        "IR ontology-design request; do not execute the hypothetical operation. The only supported "
                        "persistent template is an append-only, confirmation-gated correction of one material "
                        "A4 default distance. Persistent emission-factor, unit, ontology/schema, IFC quantity "
                        "or history-deletion changes are unsupported_persistent_operation; missing provenance "
                        "for such an unsupported operation is not target ambiguity. The "
                        "compressed schema is S=assessment system/scope/stage, E=physical element, "
                        "P=work package or calculation activity, R=material/equipment/transport/CPD/EFD, "
                        "and C=emission record/factor/unit/provenance. Preserve scopes "
                        "and multi-step dependencies, including retrieve-calculate-retrieve sequences. "
                        "Do not return chain of thought, executable tool arguments, inferred quantities, "
                        "SPARQL, arithmetic results, graph edits, or an answer to the query."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(planning_request, ensure_ascii=False, sort_keys=True),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking_type},
            "temperature": 0,
            "max_tokens": 900,
            "stream": False,
        }
        request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # Reserve a conservative token bound before sending the request.
        request_token_reserve = len(request_data) + int(payload["max_tokens"])
        used_tokens = self._cumulative_usage["total_tokens"]
        if used_tokens + request_token_reserve > self.token_budget:
            raise DeepSeekTokenBudgetExceeded(
                "DeepSeek token budget would be exceeded; "
                f"used={used_tokens}, reserve={request_token_reserve}, "
                f"budget={self.token_budget}"
            )
        response_payload: dict[str, Any] | None = None
        retryable_http = {408, 429, 500, 502, 503, 504}
        attempt_limit = self.transport_max_retries + 1
        for transport_attempt in range(1, attempt_limit + 1):
            self._decision_transport_attempts += 1
            request = Request(
                self.endpoint,
                data=request_data,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "msca-mas/0.1",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = response.read().decode("utf-8")
                response_payload = json.loads(decoded)
                break
            except HTTPError as exc:
                try:
                    provider_message = exc.read(2048).decode("utf-8", errors="replace")
                except Exception:
                    provider_message = ""
                detail = f": {provider_message}" if provider_message else ""
                if exc.code not in retryable_http or transport_attempt >= attempt_limit:
                    raise DeepSeekAPIError(f"DeepSeek HTTP {exc.code}{detail}") from exc
            except (URLError, TimeoutError, ConnectionError, ssl.SSLError) as exc:
                if transport_attempt >= attempt_limit:
                    reason = exc.reason if isinstance(exc, URLError) else exc
                    raise DeepSeekAPIError(f"DeepSeek connection failed: {reason}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DeepSeekAPIError("DeepSeek returned a non-JSON HTTP response") from exc

            self._decision_transport_retries += 1
            backoff = min(
                self.transport_backoff_seconds * (2 ** (transport_attempt - 1)),
                8.0,
            )
            if backoff:
                time.sleep(backoff)

        if response_payload is None:  # pragma: no cover - loop returns or raises
            raise DeepSeekAPIError("DeepSeek transport retry loop ended without a response")

        self._record_usage(response_payload.get("usage"))

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekAPIError("DeepSeek response omitted choices[0].message.content") from exc
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise DeepSeekAPIError("DeepSeek structured content was not a JSON object")
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek planning content was not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("DeepSeek planning content must be a JSON object")
        if not self.ontology_enabled and "ontology_profile_hints" in value:
            value = {**value, "ontology_profile_hints": []}
        return value


def planning_reasoner_from_env(
    *,
    max_retries: int = 2,
    ontology_enabled: bool = True,
) -> PlanningReasoner | None:
    """Build the optional planning adapter without ever reading a credential file."""

    provider = os.environ.get("MSCA_LLM_PROVIDER", "disabled").strip().lower()
    if provider in {"", "none", "disabled", "deterministic"}:
        return None
    if provider != "deepseek":
        raise ValueError(f"unsupported MSCA_LLM_PROVIDER: {provider}")
    thinking = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()
    if thinking != "disabled":
        raise ValueError("DEEPSEEK_THINKING must be disabled for this release")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", DeepSeekPlanningReasoner.required_model).strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    timeout = float(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "45"))
    transport_max_retries = int(os.environ.get("DEEPSEEK_TRANSPORT_MAX_RETRIES", "3"))
    transport_backoff_seconds = float(
        os.environ.get("DEEPSEEK_TRANSPORT_BACKOFF_SECONDS", "1")
    )
    token_budget = int(os.environ.get("DEEPSEEK_TOKEN_BUDGET", "3000000"))
    usage_ledger_value = os.environ.get("DEEPSEEK_USAGE_LEDGER", "").strip()
    return DeepSeekPlanningReasoner(
        api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout,
        max_retries=max_retries,
        transport_max_retries=transport_max_retries,
        transport_backoff_seconds=transport_backoff_seconds,
        token_budget=token_budget,
        usage_ledger=usage_ledger_value or None,
        ontology_enabled=ontology_enabled,
    )
