from __future__ import annotations

import re

from .models import QueryMode, ReadinessIssue, Severity
from .scenarios import grounded_persistent_distance_values
from .semantic_models import QueryContract


def _confirmation_gate_bypass_requested(query: str) -> bool:
    """Detect an explicit request to evade the confirmation-gated write protocol."""

    text = re.sub(r"\s+", " ", query.casefold())
    return bool(
        re.search(
            r"\b(?:skip|bypass|omit|without)\b.{0,72}\b(?:confirmation|approval|preview)\b",
            text,
        )
        or re.search(
            r"\b(?:write|commit|apply|persist)\b.{0,72}\b(?:directly|immediately)\b",
            text,
        )
        or re.search(
            r"\bno confirmation (?:code|digest)\b.{0,120}\b(?:write|commit|apply|persist|change)\b",
            text,
        )
        or re.search(
            # Text in a query is not a confirmation digest.
            r"\b(?:text|code|token|reference)\b.{0,60}\bas authority to\b.{0,40}\b(?:commit|write|apply|persist)\b",
            text,
        )
    )


def _percentage_semantics_conflict(query: str) -> bool:
    """Same target qualified by both 'reduce BY n%' and 'reduce TO n%' semantics.

    The two wordings define different arithmetic (multiplicative reduction vs
    replacement level); executing either one silently would guess which
    semantics the requester meant.
    """

    text = re.sub(r"\s+", " ", query.casefold())
    by_targets = [
        match.group("target")
        for match in re.finditer(
            r"\b(?:reduce|reduces|reduced|lower|cut)\s+(?P<target>[a-z0-9 \-]{1,40}?)\s+by\s+\d+(?:\.\d+)?\s*%",
            text,
        )
    ]
    to_targets = [
        match.group("target")
        for match in re.finditer(
            r"\b(?:reduce|reduces|reduced|lower|cut)\s+(?P<target>[a-z0-9 \-]{1,40}?)\s+to\s+\d+(?:\.\d+)?\s*%\s*(?:of|from)",
            text,
        )
    ]
    for by_target in by_targets:
        for to_target in to_targets:
            if by_target and to_target and (
                by_target in to_target or to_target in by_target
            ):
                return True
    return False


def _non_independent_scenario_aggregation(query: str) -> bool:
    """Requests to sum scenario cases or hierarchical views that double count.

    Two families: (a) low/base/high sensitivity cases are alternative views of
    the same design, so an 'exposure envelope' or combined total over them
    counts one design several times; (b) a project total combined with its
    partition totals repeats every partition's contribution.
    """

    text = re.sub(r"\s+", " ", query.casefold())
    scenario_cases = bool(
        re.search(r"\blow\b[^.]{0,60}?\bbase\b[^.]{0,60}?\bhigh\b", text)
        or re.search(r"\b(?:low|base|high)\s+utilisation", text)
    )
    envelope_request = bool(
        re.search(r"\b(?:exposure envelope|conservative envelope|combined total|overall total|sum(?:ming)? (?:the |all |both )?(?:three|low|base|high|cases|scenarios))", text)
        or re.search(r"\badd(?:ing)?\s+(?:the\s+)?(?:low|base|high|three|all)\b", text)
    )
    if scenario_cases and envelope_request:
        return True
    hierarchical = bool(
        re.search(r"\bproject (?:total|baseline)\b", text)
        and re.search(r"\b(?:main[- ]structure|enclosure|foundation)\b", text)
        and re.search(r"\b(?:add|plus|sum|combine|together with)\b", text)
    )
    return hierarchical


def _conflicting_persistent_instructions(query: str) -> bool:
    """Two co-valid records with different values and an invitation to pick one.

    Neither supersedes the other, so any silent choice between them is a
    decision without a governed basis -- the correction is ambiguous even if
    the planner bound exactly one of the values as the proposal.
    """

    import re as _re

    text = _re.sub(r"\s+", " ", query.casefold())
    two_records = bool(
        _re.search(r"\btwo\b[^.]{0,80}\b(?:notes?|instructions?|records?|drafts?|directives?)\b", text)
        or _re.search(r"\b(?:both|the other)\b[^.]{0,40}\b(?:note|instruction|record|draft)\b", text)
    )
    no_supersession = bool(_re.search(r"\b(?:neither|not|no one)\b[^.]{0,40}\bsupersed\w*\b", text))
    silent_choice = bool(
        _re.search(r"\b(?:choose|pick|select|decide)\b[^.]{0,40}\b(?:one|silently|without|quietly)\b", text)
        or _re.search(r"\bsilently\b", text)
    )
    return two_records and (no_supersession or silent_choice)


def _explicit_disambiguation_provided(query: str) -> bool:
    """The requester already ruled which record is authoritative.

    A superseded draft plus an explicit 'use the effective/signed instruction'
    is a verification decision the user made; treating the pair as an
    unresolved reference would refuse a resolvable request.
    """

    text = re.sub(r"\s+", " ", query.casefold())
    has_supersession = bool(
        re.search(r"\b(?:superseded|supersede|replaced|older|outdated|withdrawn)\b", text)
    )
    has_authority = bool(
        re.search(r"\b(?:signed|effective|current|governing|controlled)\b", text)
        and re.search(
            r"\b(?:use|apply|follow|take)\s+(?:the\s+)?(?:signed|effective|current|governing|controlled|instruction|note|record|version)",
            text,
        )
    )
    return has_supersession and has_authority


def _distance_mention(mention: dict[str, object]) -> bool:
    """Recognise a distance by governed name or a query-grounded kilometre unit."""

    if str(mention.get("name")) == "distance_km":
        return True
    unit = str(mention.get("unit") or "").casefold().strip()
    surface = str(mention.get("surface_form") or "").casefold()
    return unit in {"km", "kilometre", "kilometer"} or bool(
        re.search(r"\bkm\b|\bkilomet(?:re|er)s?\b", surface)
    )


_NUMBER = r"[-+]?\d+(?:\.\d+)?"

# Check units and numeric grounding for typed parameters.
_UNIT_EXPECTATIONS = {
    "mass": (r"\b(?:kg|t|tonne|ton|吨)\b", r"(?:kg|t|tonnes?|tons?)"),
    "distance_km": (r"\b(?:km|kilomet(?:re|er)s?)\b|\b公里\b", r"(?:km|kilomet(?:re|er)s?)"),
    "factor_kgco2e_per_tkm": (r"kg\s*co2e?\s*(?:per|/)\s*(?:t|tonne)[- *]?km", r"kg\s*co2e?\s*(?:per|/)\s*(?:t|tonne)[- *]?km"),
}

# Change and decision parameters need a unit in or next to the matched text.
_STRICT_UNIT_ROLES = {"proposed", "candidate", "threshold"}


def _unit_in_surface_or_window(surface: str, query: str, unit_pattern: str) -> bool:
    if re.search(unit_pattern, surface, re.I):
        return True
    position = query.casefold().find(surface.casefold())
    if position < 0:
        return False
    window = query[max(0, position - 24) : position + len(surface) + 24]
    return bool(re.search(unit_pattern, window, re.I))


def _parameter_grounding_issues(contract: Any) -> list[ReadinessIssue]:
    """Independent re-verification of unit-explicit parameter mentions."""

    issues: list[ReadinessIssue] = []
    for mention in contract.parameter_mentions:
        name = str(mention.get("name") or "")
        surface = str(mention.get("surface_form") or "")
        expectation = _UNIT_EXPECTATIONS.get(name)
        if expectation is None or not surface:
            continue
        unit_pattern, value_pattern = expectation
        role = str(mention.get("role") or "").casefold()
        strict = role in _STRICT_UNIT_ROLES
        if strict and not _unit_in_surface_or_window(surface, contract.query, unit_pattern):
            issues.append(
                ReadinessIssue(
                    code="PARAMETER_ROLE_UNIT_MISMATCH",
                    severity=Severity.BLOCKING,
                    message=(
                        f"The span declared as '{name}' does not carry the governed "
                        f"unit for that role: {surface!r}."
                    ),
                    missing_fields=[f"{name} with its governed unit"],
                    source="independent_safety_gate",
                )
            )
            continue
        query_level = {
            float(match.group(1))
            for match in re.finditer(
                rf"({_NUMBER})\s*{value_pattern}",
                contract.query,
                re.I,
            )
        }
        span_values = {
            float(match.group(1))
            for match in re.finditer(
                rf"({_NUMBER})\s*{value_pattern}",
                surface,
                re.I,
            )
        }
        if span_values and not span_values <= query_level:
            issues.append(
                ReadinessIssue(
                    code="PARAMETER_VALUE_UNGROUNDED",
                    severity=Severity.BLOCKING,
                    message=(
                        f"The value declared for '{name}' in {surface!r} does not "
                        "occur in the query with the governed unit."
                    ),
                    missing_fields=[f"{name} grounded in the query with its unit"],
                    source="independent_safety_gate",
                )
            )
    return issues


def assess_query_contract_safety(
    contract: QueryContract,
    mode: QueryMode,
) -> list[ReadinessIssue]:
    """Validate safety independently of retrieval and answer generation."""

    issues: list[ReadinessIssue] = []
    if contract.boundary_operation_requested:
        issues.append(
            ReadinessIssue(
                code="OUTSIDE_A1_A5_SCOPE",
                severity=Severity.BLOCKING,
                message=(
                    "The requested operation targets a lifecycle module outside the active "
                    "A1-A3, A4 and A5 assessment boundary; no value was calculated or written."
                ),
                missing_fields=["an approved boundary extension and governed source data"],
                source="independent_safety_gate",
            )
        )
    if contract.persistence_requested and mode != QueryMode.KNOWLEDGE_CORRECTION:
        issues.append(
            ReadinessIssue(
                code="PERSISTENCE_ROUTE_MISMATCH",
                severity=Severity.BLOCKING,
                message=(
                    "A persistent change was requested but the selected workflow is not the "
                    "confirmation-gated knowledge-correction workflow."
                ),
                source="independent_safety_gate",
            )
        )
    issues.extend(_parameter_grounding_issues(contract))
    if _percentage_semantics_conflict(contract.query):
        issues.append(
            ReadinessIssue(
                code="CALCULATION_PRECONDITION_FAILED",
                severity=Severity.BLOCKING,
                message=(
                    "The query states two different percentage semantics (reduce BY and "
                    "reduce TO) for the same target; executing either silently would guess "
                    "which arithmetic the requester meant."
                ),
                missing_fields=["one reconciled percentage semantics"],
                source="independent_safety_gate",
            )
        )
    if _non_independent_scenario_aggregation(contract.query):
        issues.append(
            ReadinessIssue(
                code="CALCULATION_PRECONDITION_FAILED",
                severity=Severity.BLOCKING,
                message=(
                    "The requested aggregate would double count: the selected cases are "
                    "alternative views of one design (or a total combined with its own "
                    "partitions), not independent contributions."
                ),
                missing_fields=["independent quantities to aggregate"],
                source="independent_safety_gate",
            )
        )
    if _conflicting_persistent_instructions(contract.query):
        issues.append(
            ReadinessIssue(
                code="AMBIGUOUS_PERSISTENT_CORRECTION",
                severity=Severity.BLOCKING,
                message=(
                    "Two co-valid records state different values and neither supersedes "
                    "the other; choosing one silently would be a decision without a "
                    "governed basis."
                ),
                missing_fields=["one atomic target/value pair with a governed basis"],
                source="independent_safety_gate",
            )
        )
    if mode == QueryMode.KNOWLEDGE_CORRECTION and not contract.persistence_requested:
        issues.append(
            ReadinessIssue(
                code="PERSISTENCE_INTENT_UNCONFIRMED",
                severity=Severity.BLOCKING,
                message=(
                    "The planner selected a persistent workflow without explicit persistence "
                    "intent; no transaction preview or graph change was produced."
                ),
                missing_fields=["an explicit request to persist a named graph fact or default"],
                source="independent_safety_gate",
            )
        )

    proposed_mentions = [
        mention
        for mention in contract.parameter_mentions
        if str(mention.get("role")) in {"proposed", "alternative"}
    ]
    unsupported_persistent_parameter = any(
        not _distance_mention(mention)
        for mention in proposed_mentions
    )
    if (
        mode == QueryMode.KNOWLEDGE_CORRECTION
        and contract.persistence_requested
        and unsupported_persistent_parameter
    ):
        issues.append(
            ReadinessIssue(
                code="UNSUPPORTED_PERSISTENT_WRITE",
                severity=Severity.BLOCKING,
                message=(
                    "The requested persistent parameter is outside the supported append-only "
                    "material A4 distance-correction template."
                ),
                missing_fields=[
                    "a supported append-only material A4 distance correction"
                ],
                source="independent_safety_gate",
            )
        )

    if (
        mode == QueryMode.KNOWLEDGE_CORRECTION
        and contract.persistence_requested
        and _confirmation_gate_bypass_requested(contract.query)
    ):
        issues.append(
            ReadinessIssue(
                code="UNSUPPORTED_PERSISTENT_WRITE",
                severity=Severity.BLOCKING,
                message=(
                    "The request explicitly attempts to bypass the mandatory preview and "
                    "exact-digest confirmation gate; no persistent change was prepared or applied."
                ),
                missing_fields=["a governed preview and exact-digest confirmation"],
                source="independent_safety_gate",
            )
        )

    proposed_distance_mentions = [
        mention
        for mention in proposed_mentions
        if _distance_mention(mention)
    ]
    proposed_distance_values = {
        float(match.group(1))
        for mention in proposed_distance_mentions
        for match in [
            re.search(
                r"(?<![A-Za-z0-9.])([-+]?\d+(?:\.\d+)?)\s*km\b",
                str(mention.get("surface_form") or ""),
                re.I,
            )
        ]
        if match
    }
    grounded_query_distances = set(
        grounded_persistent_distance_values(contract.query)
    )
    proposed_distance_values.update(grounded_query_distances)
    proposed_distance = bool(proposed_distance_values)
    bound_persistent_targets = {
        re.sub(r"[^a-z0-9]+", "", str(value).casefold())
        for value in (
            mention.get("target") for mention in proposed_distance_mentions
        )
        if value is not None and str(value).strip()
    }
    bound_persistent_targets.discard("")
    ambiguous_atomic_correction = (
        mode == QueryMode.KNOWLEDGE_CORRECTION
        and contract.persistence_requested
        and (
            len(bound_persistent_targets) > 1
            or len(proposed_distance_values) > 1
        )
    )
    if ambiguous_atomic_correction:
        issues.append(
            ReadinessIssue(
                code="AMBIGUOUS_PERSISTENT_CORRECTION",
                severity=Severity.BLOCKING,
                message=(
                    "One confirmation package cannot contain multiple independent targets "
                    "or conflicting proposed distance values."
                ),
                missing_fields=["one atomic target/value pair"],
                source="independent_safety_gate",
            )
        )
    supported_a4_correction_context = (
        mode == QueryMode.KNOWLEDGE_CORRECTION
        and contract.persistence_requested
        and contract.task_family.value in {"correction_preview", "correction_impact_preview"}
        and (
            "A4_TransportationStage" in contract.target_stages
            or "ONT-A4-TRANSPORT" in contract.planner_profile_ids
            or "TRANSPORT" in contract.reference_categories
        )
        and not unsupported_persistent_parameter
    )
    if supported_a4_correction_context and not proposed_distance:
        issues.append(
            ReadinessIssue(
                code="MISSING_A4_PARAMETERS",
                severity=Severity.BLOCKING,
                message=(
                    "The supported material A4 distance correction has no explicit proposed distance."
                ),
                missing_fields=["distance_km"],
                source="independent_safety_gate",
            )
        )

    risk_issue_map = {
        "outside_assessment_boundary": (
            "OUTSIDE_A1_A5_SCOPE",
            "The requested operation depends on a lifecycle module outside the active assessment boundary.",
            ["an approved boundary extension and governed source data"],
        ),
        "unresolved_or_unverified_reference": (
            "REFERENCE_DEMAND_UNRESOLVED",
            "The query explicitly depends on an identity, source or parameter record that has not been uniquely verified.",
            ["a unique governed identity or fingerprinted source"],
        ),
        "semantic_calculation_precondition_violation": (
            "CALCULATION_PRECONDITION_FAILED",
            "The requested arithmetic would combine non-independent or semantically incompatible quantities, or relies on an unverified premise.",
            ["semantically compatible and independently verified inputs"],
        ),
        "invalid_physical_parameter": (
            "CALCULATION_PRECONDITION_FAILED",
            "The query explicitly supplies a non-physical value for a governed calculation parameter.",
            ["a physically valid calculation parameter"],
        ),
        "ambiguous_persistent_correction": (
            "AMBIGUOUS_PERSISTENT_CORRECTION",
            "The requested persistent correction does not identify one atomic target/value decision.",
            ["one atomic target/value pair"],
        ),
        "unsupported_persistent_operation": (
            "UNSUPPORTED_PERSISTENT_WRITE",
            "The requested write cannot be represented by the append-only, confirmation-gated correction template.",
            ["a supported append-only correction with complete provenance and confirmation"],
        ),
        "contradictory_governed_premise": (
            "CALCULATION_PRECONDITION_FAILED",
            "The query states a premise or ordering that must be rejected because it conflicts with governed facts.",
            ["a premise reconciled with governed baseline facts"],
        ),
        "unverified_dimensional_mapping": (
            "REFERENCE_DEMAND_UNRESOLVED",
            "The query combines factor dimensions without a governed conversion or equivalence mapping.",
            ["a provenance-backed dimensional conversion or compatible factor"],
        ),
        "missing_persistent_parameter": (
            "MISSING_A4_PARAMETERS",
            "The otherwise supported persistent distance correction omits its required proposed value.",
            ["distance_km"],
        ),
        "confirmation_gate_bypass": (
            "UNSUPPORTED_PERSISTENT_WRITE",
            "The request attempts to bypass the mandatory preview and exact-digest confirmation gate.",
            ["a governed preview and exact-digest confirmation"],
        ),
    }
    existing_codes = {issue.code for issue in issues}
    semantic_risk_names = {
        str(risk.get("risk", "")) for risk in contract.semantic_risks
    }
    for risk in contract.semantic_risks:
        # The contract parser has already verified each risk's evidence span.
        risk_name = str(risk.get("risk", ""))
        declared_risk_name = risk_name
        if risk_name in {
            "ambiguous_persistent_correction",
            "unsupported_persistent_operation",
            "missing_persistent_parameter",
            "confirmation_gate_bypass",
        } and not (
            mode == QueryMode.KNOWLEDGE_CORRECTION
            and contract.persistence_requested
        ):
            continue
        if (
            risk_name == "unsupported_persistent_operation"
            and supported_a4_correction_context
        ):
            # Several distance edits violate the single-target confirmation scope.
            continue
        if (
            risk_name == "unresolved_or_unverified_reference"
            and _explicit_disambiguation_provided(contract.query)
        ):
            # Allow a correction when the request identifies the effective record.
            continue
        if (
            risk_name == "unverified_dimensional_mapping"
            and contract.primary_operator.value
            in {"aggregate", "compare", "rank", "share", "intensity"}
        ):
            risk_name = "semantic_calculation_precondition_violation"
        if (
            risk_name == "outside_assessment_boundary"
            and (
                not contract.boundary_operation_requested
                or "out_of_boundary_operation" not in contract.routing_evidence
            )
        ):
            continue
        if (
            declared_risk_name == "semantic_calculation_precondition_violation"
            and "unverified_dimensional_mapping" in semantic_risk_names
        ):
            # Report a missing conversion before the resulting arithmetic failure.
            continue
        specification = risk_issue_map.get(risk_name)
        if specification is None or specification[0] in existing_codes:
            continue
        code, message, missing_fields = specification
        issues.append(
            ReadinessIssue(
                code=code,
                severity=Severity.BLOCKING,
                message=message,
                missing_fields=missing_fields,
                source="independent_semantic_risk_gate",
            )
        )
        existing_codes.add(code)
    return issues
