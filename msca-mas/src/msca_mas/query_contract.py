from __future__ import annotations

import re
from typing import Any

from .intent import analyse_query_intent
from .models import QueryMode
from .semantic_models import QueryContract, QueryOperator, QueryTaskFamily


_STAGE_ORDER = (
    "A1-A3_ProductStage",
    "A4_TransportationStage",
    "A5_ConstructionStage",
)


def _normalise(query: str) -> str:
    text = query.lower()
    for value in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(value, "-")
    return re.sub(r"\s+", " ", text.replace("-", " ")).strip()


def _has(text: str, *terms: str) -> bool:
    for term in terms:
        if re.fullmatch(r"[a-z0-9][a-z0-9 ._-]*", term):
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
                return True
        elif term in text:
            return True
    return False


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _verbatim_span_present(query: str, surface: str) -> bool:
    """Match an exact span with no adjacent ASCII letters or digits."""

    return (
        re.search(rf"(?<![0-9A-Za-z]){re.escape(surface)}(?![0-9A-Za-z])", query, re.I)
        is not None
    )


def parse_query_contract(
    query: str,
    mode: QueryMode,
    *,
    advisory_task_family: str | None = None,
    advisory_operation_hints: list[str] | None = None,
    advisory_semantic_anchors: list[str] | None = None,
    advisory_contract: dict[str, Any] | None = None,
) -> QueryContract:
    """Parse a safe operation-level contract without producing engineering values.

    This is deliberately broader than the route classifier: it normalises
    punctuation, preserves explicit record identifiers and records multi-step
    dependencies.  It never selects a numeric answer or supplies a missing
    engineering parameter.
    """

    lowered = query.lower()
    normalised = _normalise(query)
    advisory_hints = set(advisory_operation_hints or [])
    model_contract = dict(advisory_contract or {})
    intent_signals = analyse_query_intent(query)
    identifiers = sorted(
        {
            match.upper()
            for match in re.findall(
                r"(?<![A-Za-z0-9])(?:M-\d+-\d+|V-\d+-\d+)(?![A-Za-z0-9])",
                query,
                re.I,
            )
        }
    )

    scopes: list[str] = []
    if _has(normalised, "main structure", "main structural", "主体结构"):
        scopes.append("main_structure")
    if _has(normalised, "enclosure", "diaphragm", "retaining wall", "retaining structure", "strut", "围护结构", "地下连续墙"):
        scopes.append("enclosure_structure")
    if _has(
        normalised,
        "foundation pit",
        "foundation excavation",
        "foundation haul",
        "earthwork",
        "excavation",
        "backfill",
        "off site volume",
        "基坑",
        "土方",
    ):
        scopes.append("foundation_pit")
    if not scopes and _has(
        normalised,
        "three engineering scopes",
        "each engineering scope",
        "engineering scopes",
        "三个工程范围",
        "各工程范围",
    ):
        scopes.extend(("main_structure", "enclosure_structure", "foundation_pit"))

    stages: list[str] = []
    if re.search(r"(?<![a-z0-9])a1\s*(?:to|through)?\s*a5(?![a-z0-9])", normalised):
        stages.extend(_STAGE_ORDER)
    else:
        if re.search(r"(?<![a-z0-9])a1\s*a3(?![a-z0-9])", normalised):
            stages.append("A1-A3_ProductStage")
        if re.search(r"(?<![a-z0-9])a4(?![a-z0-9])", normalised):
            stages.append("A4_TransportationStage")
        if re.search(r"(?<![a-z0-9])a5(?![a-z0-9])", normalised):
            stages.append("A5_ConstructionStage")

    resources: list[str] = []
    if _has(normalised, "concrete", "ggbs", "slag"):
        resources.append("concrete")
    if _has(normalised, "steel", "rebar", "reinforcement"):
        resources.append("steel")
    if _has(normalised, "formwork", "reuse"):
        resources.append("formwork")

    meta_ontology = _has(
        normalised,
        "query specific subontology",
        "subontology",
        "ontology term",
        "ontology content",
        "semantic path",
    ) or bool(
        re.search(
            r"\bwhich\b.{0,96}\b(?:question specific )?(?:evidence subset|evidence package)\b",
            normalised,
        )
    )
    outside_boundary = bool(
        re.search(r"(?<![a-z0-9])(?:b[1-7]|c[1-4]|module d)(?![a-z0-9])", normalised)
    ) or _has(
        normalised,
        "outside the active",
        "outside the declared",
        "beyond the system boundary",
        "operational use",
        "end of life",
    )
    meta_boundary = meta_ontology and outside_boundary

    scenario_markers = _has(
        normalised,
        "what if",
        "scenario",
        "simulate",
        "sensitivity",
        "substitution",
        "reduce",
        "reduction",
        "amortise",
        "amortize",
        "reuse cycles",
        "alternative",
    )
    explicit_formula = (
        _has(normalised, "work quantity", "material emission", "material carbon")
        and _has(normalised, "consumption rate", "emission factor")
    ) or (
        _has(normalised, "work volume", "equipment emission", "equipment carbon")
        and _has(normalised, "productivity")
        and _has(normalised, "energy per hour", "energy rate")
        and _has(normalised, "energy factor")
    )
    explicit_compound = _has(
        normalised,
        "compound",
        "coupled option",
        "combined option",
        "combined with",
    )
    named_parameter_sweep = bool(
        re.search(
            r"\b(?:plan[ _-]*utili[sz]ation|a4[ _-]*scaling|project[ _-]*scaling)[ _-]*factor\b"
            r".{0,48}?\bat\s+",
            lowered,
        )
    )
    reuse_request = _has(
        normalised,
        "reuse",
        "reusable",
        "amortise",
        "amortize",
        "allocated over",
        "allocated burden per use",
    )
    decision_operation_count = sum(
        (
            _has(normalised, "minimum carbon", "argmin", "minimum-carbon"),
            _has(normalised, "rank", "ranked", "decision order"),
            _has(normalised, "threshold"),
            _has(normalised, "compare", "compared"),
            _has(normalised, "project total", "project delta"),
            _has(normalised, "excavation", "off site volume", "off-site volume"),
            _has(normalised, "governed concrete default", "check every route", "check every option"),
        )
    )
    operation_marker_count = sum(
        (
            _has(normalised, "sensitivity", "sweep") or named_parameter_sweep,
            _has(normalised, "ggbs", "substitution"),
            _has(normalised, "reduce", "reduction"),
            reuse_request,
            _has(normalised, "transport", "a4 alternative"),
        )
    )
    dependency_language = _has(
        normalised,
        " then ",
        "after which",
        "carry into",
        "carried into",
        "carries into",
        "carrying into",
        "carry only",
        "lead back",
        "combined with",
        "before the alternatives",
        "after the revised",
        "and then rank",
        "and rank",
    )
    compound = mode == QueryMode.STRATEGIC_ANALYSIS and (
        explicit_compound
        or advisory_task_family == "compound_scenario"
        or (
            dependency_language
            and operation_marker_count >= 1
            and decision_operation_count >= 1
        )
    )
    correction_impact = mode == QueryMode.KNOWLEDGE_CORRECTION and (
        advisory_task_family == "correction_impact_preview"
        or _has(
            normalised,
            "before writing",
            "before persist",
            "calculate the a4 distance multiplier",
            "calculate impact",
            "impact before",
            "distance multiplier",
            "impact multiplier",
            "quantify the impact",
        )
    )

    compound_template: str | None = None
    if compound:
        distance_values = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*km\b", lowered)
        distance_list = re.search(
            r"((?:\d+(?:\.\d+)?\s*(?:,|/|\band\b)\s*)+\d+(?:\.\d+)?)\s*km\b",
            lowered,
            re.I,
        )
        if distance_list:
            distance_values = re.findall(r"\d+(?:\.\d+)?", distance_list.group(1))
        if (_has(normalised, "sensitivity", "sweep") or named_parameter_sweep) and _has(
            normalised, "minimum carbon", "minimum carbon point", "argmin"
        ):
            compound_template = "sensitivity_argmin_post_retrieval"
        elif _has(normalised, "ggbs", "slag", "substitution", "binder option"):
            compound_template = "substitution_rank_project_delta"
        elif reuse_request and _has(
            normalised, "steel", "factor reduction", "factor improvement"
        ):
            compound_template = "reuse_reduction_composition"
        elif (
            _has(
                normalised,
                "three a4 alternatives",
                "a4 alternatives",
                "three a4 routes",
                "a4 routes",
            )
            or (_has(normalised, "a4 alternative", "transport") and len(distance_values) >= 3)
        ):
            compound_template = "multi_transport_rank_post_retrieval"
        elif _has(normalised, "concrete") and _has(normalised, "a4") and _has(
            normalised, "reduce", "reduction", "lower", "lowers", "improvement"
        ):
            compound_template = "dual_reduction_threshold"
        else:
            compound_template = "generic_dependency_ordered_program"

    multi_stage_aggregate = len(stages) >= 2 and _has(normalised, "separately", "分别", "各阶段") and _has(
        normalised, "carbon", "emission", "碳", "排放"
    )
    aggregate_request = multi_stage_aggregate or _has(
        normalised,
        "aggregate",
        "re aggregate",
        "sum of",
        "stage totals",
        "carbon totals",
        "scope totals",
        "system totals",
        "totals separately",
        "gross station envelope volume",
        "各阶段排放",
        "阶段排放合计",
        "总长度",
        "总面积",
        "总体积",
        "聚合",
        "合计",
    ) or (
        _has(normalised, "total", "totals", "总碳", "总排放", "总量")
        and _has(
            normalised,
            "what is",
            "what are",
            "how much",
            "how many",
            "give",
            "return",
            "calculate",
            "compute",
            "计算",
            "是多少",
        )
        and not _has(normalised, "reconcile", "audit", "stored", "核对", "审计")
    )
    conditional_boundary_calculation = outside_boundary and _has(
        normalised,
        "calculate the revised total only if",
        "calculate revised total only if",
        "only if",
    ) and _has(normalised, "calculate", "compute", "计算")
    extrema_ranking = _has(
        normalised,
        "highest",
        "largest",
        "lowest",
        "smallest",
        "contributes the most",
        "最高",
        "最大",
        "最低",
        "最小",
    ) and _has(
        normalised,
        "element",
        "elements",
        "hotspot",
        "work package",
        "material",
        "option",
        "alternative",
        "构件",
        "热点",
        "工作包",
        "材料",
        "运输",
        "方案",
        "选项",
    )
    if _has(normalised, "sensitivity", "parameter sweep", "sweep") or named_parameter_sweep:
        operator = QueryOperator.SENSITIVITY
    elif _has(normalised, "ggbs", "slag", "substitution", "binder option"):
        operator = QueryOperator.MATERIAL_SUBSTITUTION
    elif reuse_request and mode == QueryMode.STRATEGIC_ANALYSIS:
        operator = QueryOperator.REUSE_ALLOCATION
    elif mode == QueryMode.STRATEGIC_ANALYSIS and _has(normalised, "transport", "a4 alternative"):
        operator = QueryOperator.TRANSPORT
    elif mode == QueryMode.STRATEGIC_ANALYSIS and _has(
        normalised, "reduce", "reduction", "lower", "lowers", "factor improvement"
    ):
        operator = QueryOperator.FACTOR_REDUCTION
    elif (
        re.search(r"\b(?:rank|ranking|ranked)\b", normalised)
        or extrema_ranking
        or _has(
            normalised,
            "from highest to lowest",
            "from lowest to highest",
            "ordered ranking",
            "decision order",
            "priority order",
            "order by carbon",
        )
    ) and not _has(normalised, "compare", "comparison", "pairwise", "extrema", "range"):
        operator = QueryOperator.RANK
    elif _has(normalised, "share", "shares", "percentage", "proportion", "ratio", "占比", "百分比", "比例"):
        operator = QueryOperator.SHARE
    elif _has(normalised, "carbon intensity", "per m2", "per m3"):
        operator = QueryOperator.INTENSITY
    elif _has(normalised, "compare", "comparison", "pairwise", "extrema", "range", "比较", "差值") or (
        _has(normalised, "highest", "largest", "lowest", "smallest", "最高", "最大", "最低", "最小")
        and not _has(normalised, "trace", "element", "hotspot", "追溯", "构件", "热点")
    ):
        operator = QueryOperator.COMPARE
    elif aggregate_request or conditional_boundary_calculation:
        operator = QueryOperator.AGGREGATE
    elif _has(normalised, "trace", "provenance", "audit chain"):
        operator = QueryOperator.TRACE
    elif mode == QueryMode.KNOWLEDGE_CORRECTION:
        operator = QueryOperator.TRANSACTION_PREVIEW
    else:
        operator = QueryOperator.NONE

    # Callback planners may return a hint-only proposal.
    if mode == QueryMode.INFORMATION_RETRIEVAL:
        if "rank" in advisory_hints and operator in {QueryOperator.NONE, QueryOperator.AGGREGATE}:
            operator = QueryOperator.RANK
        elif "share" in advisory_hints and operator == QueryOperator.NONE:
            operator = QueryOperator.SHARE
        elif "compare" in advisory_hints and operator == QueryOperator.NONE:
            operator = QueryOperator.COMPARE
    elif mode == QueryMode.STRATEGIC_ANALYSIS:
        if "sensitivity" in advisory_hints:
            operator = QueryOperator.SENSITIVITY
        elif "material_substitution" in advisory_hints:
            operator = QueryOperator.MATERIAL_SUBSTITUTION
        elif "reuse_allocation" in advisory_hints:
            operator = QueryOperator.REUSE_ALLOCATION
        elif "transport_emission" in advisory_hints:
            operator = QueryOperator.TRANSPORT
        elif "factor_reduction" in advisory_hints and operator == QueryOperator.NONE:
            operator = QueryOperator.FACTOR_REDUCTION

    trace_seed_ranking = mode == QueryMode.INFORMATION_RETRIEVAL and _has(
        normalised,
        "trace the highest a1 a3 element through",
        "trace the highest element through",
    )
    if trace_seed_ranking:
        operator = QueryOperator.RANK

    if correction_impact:
        family = QueryTaskFamily.CORRECTION_IMPACT_PREVIEW
    elif mode == QueryMode.KNOWLEDGE_CORRECTION:
        family = QueryTaskFamily.CORRECTION_PREVIEW
    elif compound:
        family = QueryTaskFamily.COMPOUND_SCENARIO
    elif meta_ontology:
        family = QueryTaskFamily.ONTOLOGY_DESIGN
    elif mode == QueryMode.STRATEGIC_ANALYSIS:
        family = QueryTaskFamily.SCENARIO
    elif explicit_formula or operator in {
        QueryOperator.AGGREGATE,
        QueryOperator.COMPARE,
        QueryOperator.RANK,
        QueryOperator.SHARE,
        QueryOperator.INTENSITY,
    }:
        family = QueryTaskFamily.RETRIEVAL_CALCULATION
    else:
        family = QueryTaskFamily.RETRIEVAL

    requires_calculator = (
        family
        in {
            QueryTaskFamily.RETRIEVAL_CALCULATION,
            QueryTaskFamily.SCENARIO,
            QueryTaskFamily.COMPOUND_SCENARIO,
            QueryTaskFamily.CORRECTION_IMPACT_PREVIEW,
        }
        and not meta_boundary
    )

    governed_operator = operator
    llm_contract_validated = bool(model_contract)
    planner_profile_ids: list[str] = []
    reference_categories: list[str] = []
    required_capabilities: list[str] = []
    parameter_mentions: list[dict[str, Any]] = []
    semantic_risks: list[dict[str, Any]] = []
    discarded_semantic_risks: list[dict[str, Any]] = []
    if llm_contract_validated:
        allowed_profiles = {
            "ONT-PROJECT-CARBON",
            "ONT-SYSTEM-CARBON",
            "ONT-ENCLOSURE-CPD",
            "ONT-FOUNDATION-EARTHWORK",
            "ONT-A4-TRANSPORT",
            "ONT-SEPRC-PROVENANCE",
            "ONT-SCENARIO",
            "ONT-SCOPE-QUALITY",
        }
        allowed_capabilities = {
            "metadata_retrieval",
            "typed_extraction",
            "governed_reader",
            "deterministic_calculation",
            "post_calculation_retrieval",
            "transaction_preview",
            "synthesis",
        }
        planner_profile_ids = _ordered_unique(
            [
                str(value)
                for value in model_contract.get("ontology_profile_hints", [])
                if str(value) in allowed_profiles
            ]
        )
        reference_categories = _ordered_unique(
            [
                str(value)
                for value in model_contract.get("reference_categories", [])
                if str(value) in {"CPD", "EFD", "TRANSPORT"}
            ]
        )
        required_capabilities = _ordered_unique(
            [
                str(value)
                for value in model_contract.get("required_capabilities", [])
                if str(value) in allowed_capabilities
            ]
        )

        proposed_family = model_contract.get("task_family")
        if proposed_family:
            try:
                family = QueryTaskFamily(str(proposed_family))
            except ValueError:
                pass
        proposed_operator = model_contract.get("primary_operator")
        if proposed_operator is not None:
            try:
                proposed = QueryOperator(str(proposed_operator))
                explicit_ir_operators = {
                    QueryOperator.COMPARE,
                    QueryOperator.RANK,
                    QueryOperator.SHARE,
                    QueryOperator.INTENSITY,
                }
                if not (
                    mode == QueryMode.INFORMATION_RETRIEVAL
                    and governed_operator in explicit_ir_operators
                    and proposed in {QueryOperator.NONE, QueryOperator.AGGREGATE}
                ):
                    operator = proposed
            except ValueError:
                pass

        proposed_scopes = [
            str(value)
            for value in model_contract.get("target_scopes", [])
            if str(value) in {"project", "main_structure", "enclosure_structure", "foundation_pit"}
        ]
        if proposed_scopes:
            scopes = _ordered_unique(proposed_scopes)
        proposed_stages = [
            str(value)
            for value in model_contract.get("target_stages", [])
            if str(value) in set(_STAGE_ORDER)
        ]
        if proposed_stages:
            stages = _ordered_unique(proposed_stages)

        grounded_resources: list[str] = []
        for value in model_contract.get("resource_targets", []):
            candidate = re.sub(r"\s+", " ", str(value).strip().lower())
            surface = candidate.replace("_", " ").replace("-", " ")
            if candidate and len(candidate) <= 80 and (
                surface in normalised
                or candidate in {"all_resources", "transport"}
            ):
                grounded_resources.append(candidate.replace(" ", "_"))
        if grounded_resources:
            resources = _ordered_unique(grounded_resources)

        grounded_identifiers = [
            str(value)
            for value in model_contract.get("explicit_identifiers", [])
            if str(value).strip() and _verbatim_span_present(query, str(value).strip())
        ]
        if grounded_identifiers:
            identifiers = _ordered_unique(grounded_identifiers)

        compound = bool(model_contract.get("compound")) or family == QueryTaskFamily.COMPOUND_SCENARIO
        proposed_template = model_contract.get("compound_template")
        if proposed_template:
            compound_template = str(proposed_template)
        meta_ontology = bool(model_contract.get("meta_ontology_query"))
        correction_impact = bool(model_contract.get("correction_impact_preview"))
        if correction_impact:
            family = QueryTaskFamily.CORRECTION_IMPACT_PREVIEW

        for raw_mention in model_contract.get("parameter_mentions", []):
            if not isinstance(raw_mention, dict):
                continue
            surface = str(raw_mention.get("surface_form", "")).strip()
            confidence = raw_mention.get("confidence", 0.0)
            if not surface or not _verbatim_span_present(query, surface):
                continue
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                continue
            parameter_mentions.append(
                {
                    "name": str(raw_mention.get("name", "")),
                    "role": str(raw_mention.get("role", "")),
                    "surface_form": surface,
                    "unit": raw_mention.get("unit"),
                    "target": raw_mention.get("target"),
                    "confidence": float(confidence),
                }
            )

        allowed_risks = {
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
        }
        for raw_risk in model_contract.get("semantic_risks", []):
            if not isinstance(raw_risk, dict):
                continue
            risk = str(raw_risk.get("risk", ""))
            surface = str(raw_risk.get("surface_form", "")).strip()
            confidence = raw_risk.get("confidence", 0.0)
            if risk not in allowed_risks:
                continue
            if not surface or not _verbatim_span_present(query, surface):
                # Keep unmatched risk labels for audit, but do not use them to block execution.
                discarded_semantic_risks.append(
                    {"risk": risk, "surface_form": surface, "reason": "evidence_span_not_in_query"}
                )
                continue
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                discarded_semantic_risks.append(
                    {"risk": risk, "surface_form": surface, "reason": "invalid_confidence"}
                )
                continue
            semantic_risks.append(
                {
                    "risk": risk,
                    "surface_form": surface,
                    "confidence": float(confidence),
                }
            )

        arithmetic_operators = {
            QueryOperator.AGGREGATE,
            QueryOperator.COMPARE,
            QueryOperator.RANK,
            QueryOperator.SHARE,
            QueryOperator.INTENSITY,
            QueryOperator.SENSITIVITY,
            QueryOperator.FACTOR_REDUCTION,
            QueryOperator.MATERIAL_SUBSTITUTION,
            QueryOperator.TRANSPORT,
            QueryOperator.REUSE_ALLOCATION,
        }
        requires_calculator = (
            bool(model_contract.get("requires_calculator"))
            or "deterministic_calculation" in required_capabilities
            or operator in arithmetic_operators
            or family
            in {
                QueryTaskFamily.RETRIEVAL_CALCULATION,
                QueryTaskFamily.SCENARIO,
                QueryTaskFamily.COMPOUND_SCENARIO,
                QueryTaskFamily.CORRECTION_IMPACT_PREVIEW,
            }
        ) and not meta_boundary

    post_retrieval = compound and (
        _has(normalised, "then retrieve", "retrieve its implied", "retrieve whether")
        or compound_template
        in {
            "sensitivity_argmin_post_retrieval",
            "multi_transport_rank_post_retrieval",
        }
    )
    if llm_contract_validated:
        post_retrieval = bool(model_contract.get("requires_post_calculation_retrieval")) or (
            "post_calculation_retrieval" in required_capabilities
        )

    operations = ["resolve_query_contract", "resolve_query_specific_subontology"]
    hint_to_operation = {
        "retrieve_overview": "retrieve_project_overview",
        "retrieve_resource": "retrieve_resource_breakdown",
        "retrieve_elements": "retrieve_elements",
        "retrieve_work_packages": "retrieve_work_packages",
        "retrieve_cpd_efd": "retrieve_cpd_efd",
        "retrieve_transport": "retrieve_transport_references",
        "retrieve_ifc_inventory": "retrieve_ifc_inventory",
        "trace": "retrieve_element_trace",
        "return_subontology": "return_subontology_signature",
        "boundary_validation": "validate_assessment_boundary",
    }
    operations.extend(
        operation
        for hint, operation in hint_to_operation.items()
        if hint in advisory_hints
    )
    if meta_ontology or "return_subontology" in advisory_hints:
        operations.append("return_subontology_signature")
    else:
        operations.append("retrieve_project_overview")
    if "retrieve_work_packages" in advisory_hints or (
        not llm_contract_validated
        and _has(normalised, "work package", "work package template", "most instantiated", "process")
    ):
        operations.append("retrieve_work_packages")
    if "retrieve_elements" in advisory_hints or "trace" in advisory_hints or (
        not llm_contract_validated
        and _has(normalised, "element", "hotspot", "top 10", "top ten")
    ):
        operations.append("retrieve_elements")
    if resources:
        operations.append("retrieve_resource_breakdown")
    if "retrieve_transport" in advisory_hints or "TRANSPORT" in reference_categories or (
        not llm_contract_validated
        and (identifiers or _has(normalised, "transport", "a4", "vehicle", "t km", "distance"))
    ):
        operations.append("retrieve_transport_references")
    if "retrieve_cpd_efd" in advisory_hints or set(reference_categories) & {"CPD", "EFD"} or (
        not llm_contract_validated and _has(normalised, "cpd", "efd")
    ):
        operations.append("retrieve_cpd_efd")
    if mode != QueryMode.INFORMATION_RETRIEVAL:
        operations.append("extract_typed_delta")
    if requires_calculator:
        operations.append("execute_calculation_program")
    if post_retrieval:
        operations.append("retrieve_post_calculation_evidence")
    if mode == QueryMode.KNOWLEDGE_CORRECTION:
        operations.append("prepare_non_mutating_preview")
    operations.append("verify_epistemic_completeness")

    ontology_obligations = [
        "assessment boundary and lifecycle stage",
        "System Element Process Resource Carbon provenance",
    ]
    if scopes:
        ontology_obligations.append("engineering system and physical element classification")
    if stages:
        ontology_obligations.append("lifecycle stage carbon emission record and unit")
    if resources:
        ontology_obligations.append("resource quantity emission factor and carbon emission")
    if "retrieve_transport_references" in operations:
        ontology_obligations.append(
            "A4 material transport distance vehicle payload and transport emission factor"
        )
    if "retrieve_cpd_efd" in operations:
        ontology_obligations.append(
            "construction process CPD activity data EFD energy factor and provenance"
        )
    if "retrieve_elements" in operations:
        ontology_obligations.append("physical element GlobalId work package and emission trace")
    if requires_calculator:
        ontology_obligations.append("calculation activity input output unit and derivation provenance")
    if mode == QueryMode.KNOWLEDGE_CORRECTION:
        ontology_obligations.append(
            "proposed assertion validation provenance version and confirmation gate"
        )

    contract_intent_kind = intent_signals.request_kind
    contract_intent_confidence = intent_signals.confidence
    contract_existing_data = intent_signals.existing_data_requested
    contract_calculation = intent_signals.calculation_requested
    contract_scenario = intent_signals.scenario_requested
    contract_persistence = intent_signals.persistence_requested
    contract_outside_boundary = intent_signals.outside_boundary_referenced
    if llm_contract_validated and isinstance(model_contract.get("intent_evidence"), dict):
        model_evidence = model_contract["intent_evidence"]
        proposed_kind = str(model_evidence.get("request_kind", "ambiguous"))
        if proposed_kind in {
            "read_existing",
            "derive_from_existing",
            "hypothetical_analysis",
            "persistent_change",
            "boundary_validation",
            "ambiguous",
        }:
            contract_intent_kind = proposed_kind
        proposed_confidence = model_evidence.get("confidence", 0.0)
        if isinstance(proposed_confidence, (int, float)):
            contract_intent_confidence = max(0.0, min(1.0, float(proposed_confidence)))
        contract_existing_data = bool(model_evidence.get("existing_data_requested"))
        contract_calculation = bool(model_evidence.get("calculation_requested"))
        contract_scenario = (
            bool(model_evidence.get("counterfactual_requested"))
            or mode == QueryMode.STRATEGIC_ANALYSIS
        )
        contract_persistence = (
            bool(model_evidence.get("persistence_requested"))
            or mode == QueryMode.KNOWLEDGE_CORRECTION
        )
        contract_outside_boundary = (
            str(model_evidence.get("active_boundary_status")) == "outside_active_boundary"
            or intent_signals.outside_boundary_referenced
        )

    return QueryContract(
        query=query,
        mode=mode,
        intent_kind=contract_intent_kind,
        intent_confidence=contract_intent_confidence,
        read_only=not contract_persistence,
        existing_data_requested=contract_existing_data,
        calculation_requested=contract_calculation,
        scenario_requested=contract_scenario,
        persistence_requested=contract_persistence,
        outside_boundary_referenced=contract_outside_boundary,
        boundary_information_request=intent_signals.boundary_information_request,
        boundary_operation_requested=intent_signals.out_of_boundary_operation_requested,
        requires_human_confirmation=contract_persistence,
        task_family=family,
        primary_operator=operator,
        operation_ids=_ordered_unique(operations),
        target_scopes=_ordered_unique(scopes),
        target_stages=_ordered_unique(stages),
        resource_targets=_ordered_unique(resources),
        explicit_identifiers=identifiers,
        compound=compound,
        compound_template=compound_template,
        requires_calculator=requires_calculator,
        requires_post_calculation_retrieval=post_retrieval,
        meta_ontology_query=meta_ontology,
        meta_boundary_query=meta_boundary,
        correction_impact_preview=correction_impact,
        advisory_task_family=advisory_task_family,
        advisory_operation_hints=list(advisory_operation_hints or []),
        advisory_semantic_anchors=list(advisory_semantic_anchors or []),
        planner_profile_ids=planner_profile_ids,
        reference_categories=reference_categories,
        required_capabilities=required_capabilities,
        parameter_mentions=parameter_mentions,
        semantic_risks=semantic_risks,
        discarded_semantic_risks=discarded_semantic_risks,
        llm_contract_validated=llm_contract_validated,
        ontology_obligations=_ordered_unique(ontology_obligations),
        routing_evidence=intent_signals.evidence,
        parser_backend=(
            "llm_semantic_contract_with_governed_validation"
            if llm_contract_validated
            else "deterministic_query_contract"
        ),
    )
