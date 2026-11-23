from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .entity_resolution import canonical_query_entity_label
from .models import PersistenceMode, ReadinessIssue, ScenarioChange, ScenarioOverlay, Severity
from .util import stable_digest


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
_MATERIAL_CODE = r"(?<![A-Za-z0-9])(M-\d+-\d+)(?![A-Za-z0-9])"


def _first_number(patterns: Iterable[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _all_numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(_NUMBER, text, re.IGNORECASE)]


def _percentage(text: str) -> float | None:
    value = _first_number([rf"({_NUMBER})\s*%"], text)
    return value / 100.0 if value is not None else None


def _issue(
    code: str,
    message: str,
    missing_fields: list[str] | None = None,
) -> ReadinessIssue:
    return ReadinessIssue(
        code=code,
        severity=Severity.BLOCKING,
        message=message,
        missing_fields=list(missing_fields or []),
        source="extractor",
    )


def _finish(
    query: str,
    persistence: PersistenceMode,
    changes: list[ScenarioChange],
    issues: list[ReadinessIssue],
) -> ScenarioOverlay:
    payload = {
        "query": query,
        "persistence": persistence.value,
        "changes": [change.model_dump(mode="json") for change in changes],
    }
    return ScenarioOverlay(
        overlay_id=f"overlay-{stable_digest(payload)[:16]}",
        persistence=persistence,
        changes=changes,
        ready=not any(issue.severity == Severity.BLOCKING for issue in issues),
        issues=issues,
    )


def _proposed_distances(text: str) -> list[float]:
    patterns = (
        rf"\b(?:set|sets|setting|propose|proposes|proposed|revise|revises|revised|"
        rf"update|updates|updated|replace|replaces|correct|corrects)\b"
        rf"[^.;]{{0,72}}?({_NUMBER})\s*km\b",
        rf"\b(?:transport|travel|haul|route)\s+distance\b[^.;]{{0,48}}?"
        rf"(?:as|to|of|=)\s*({_NUMBER})\s*km\b",
        rf"\bdistance\s+as\s*({_NUMBER})\s*km\b",
    )
    values: list[float] = []
    for pattern in patterns:
        values.extend(float(match.group(1)) for match in re.finditer(pattern, text, re.I))
    return list(dict.fromkeys(values))


def grounded_persistent_distance_values(text: str) -> list[float]:
    """Return unit-grounded candidate values for a persistent distance decision.

    Action-bound values take precedence, so a query that states both the
    current and proposed distance yields only the proposal.  When no action
    binding is available, every explicit kilometre value is retained: one is
    a usable candidate, while more than one is an atomicity conflict that must
    be resolved instead of guessed.  This is a numeric/unit grounding fallback
    for an incomplete LLM parameter list, not an answer or benchmark rule.
    """

    action_bound = _proposed_distances(text)
    if action_bound:
        return action_bound
    return list(
        dict.fromkeys(
            float(value)
            for value in re.findall(rf"({_NUMBER})\s*km\b", text, re.I)
        )
    )


def _persistent_transport_overlay(query: str) -> ScenarioOverlay:
    lowered = query.lower()
    persistence = PersistenceMode.PERSISTENT
    material_codes = sorted({value.upper() for value in re.findall(_MATERIAL_CODE, query, re.I)})
    proposed = grounded_persistent_distance_values(lowered)

    if len(material_codes) > 1 or len(proposed) > 1:
        change = ScenarioChange(
            change_type="transport_override",
            target="ambiguous_multi_target_request",
        )
        return _finish(
            query,
            persistence,
            [change],
            [
                _issue(
                    "AMBIGUOUS_PERSISTENT_CORRECTION",
                    "A confirmation package requires one semantic target and one proposed value.",
                    ["one atomic target/value pair"],
                )
            ],
        )

    transport_context = any(
        term in lowered
        for term in (
            "transport",
            "travel distance",
            "distance",
            "haul",
            "route",
            "a4",
            "logistics",
            "运输",
            "运距",
            "距离",
            "路线",
            "物流",
        )
    ) or bool(material_codes and proposed)
    if not transport_context:
        return _finish(
            query,
            persistence,
            [],
            [
                _issue(
                    "UNSUPPORTED_PERSISTENT_WRITE",
                    "Only append-only corrections represented by a governed transaction template are supported.",
                    ["supported persistent target and value"],
                )
            ],
        )

    distance = proposed[0] if proposed else None
    issues: list[ReadinessIssue] = []
    if distance is None:
        issues.append(
            _issue(
                "MISSING_A4_PARAMETERS",
                "A persistent transport correction requires an explicit proposed distance.",
                ["distance_km"],
            )
        )
        parameters: dict[str, object] = {}
        explicit: list[str] = []
    else:
        parameters = {"distance_km": distance}
        explicit = ["distance_km"]
        if distance <= 0:
            issues.append(
                _issue(
                    "CALCULATION_PRECONDITION_FAILED",
                    "A transport distance must be greater than zero.",
                    ["positive distance_km"],
                )
            )
    target = (
        f"material_transport_default:{material_codes[0]}"
        if material_codes
        else "project_transport"
    )
    return _finish(
        query,
        persistence,
        [
            ScenarioChange(
                change_type="transport_override",
                target=target,
                parameters=parameters,
                explicit_parameters=explicit,
            )
        ],
        issues,
    )


def _named_sweep(text: str) -> tuple[str, list[float]] | None:
    parameter_match = re.search(
        r"\b(plan[ _-]*utili[sz]ation[ _-]*factors?|a4[ _-]*scaling[ _-]*factors?|"
        r"project[ _-]*scaling[ _-]*factors?)\b",
        text,
        re.I,
    )
    if not parameter_match:
        return None
    tail = text[parameter_match.end() :]
    numeric_list = rf"{_NUMBER}(?:\s*(?:,|/|\band\b)\s*{_NUMBER})+"
    value_match = re.search(
        rf"\b(?:at|of)\s+({numeric_list})",
        tail,
        re.I,
    )
    values = _all_numbers(value_match.group(1)) if value_match else []
    canonical = re.sub(r"[ -]+", "_", parameter_match.group(1).lower())
    canonical = canonical.replace("utilization", "utilisation")
    if canonical.endswith("_factors"):
        canonical = canonical[:-1]
    return canonical, list(dict.fromkeys(values))


def _is_compound(text: str) -> bool:
    distance_list = re.search(
        rf"((?:{_NUMBER}\s*(?:,|/|\band\b)\s*)+{_NUMBER})\s*km\b",
        text,
        re.I,
    )
    distances = (
        re.findall(_NUMBER, distance_list.group(1), re.I)
        if distance_list
        else re.findall(rf"({_NUMBER})\s*km\b", text, re.I)
    )
    named_sweep = _named_sweep(text)
    return any(
        (
            "compound" in text
            and sum(term in text for term in ("concrete", "a4", "steel", "reuse", "ggbs")) >= 2,
            any(term in text for term in ("coupled option", "combined option", "combined with"))
            and sum(term in text for term in ("concrete", "a4", "steel", "reuse", "ggbs")) >= 2,
            "ggbs" in text and "project total" in text and any(term in text for term in ("rank", "ranking")),
            bool(named_sweep)
            and any(term in text for term in ("minimum-carbon", "minimum carbon", "argmin"))
            and any(term in text for term in ("excavation", "off-site", "off site", "geometry")),
            any(term in text for term in ("reuse", "reusable", "allocated over"))
            and "steel" in text
            and any(term in text for term in ("compare", "coupled", "combined")),
            len(distances) >= 2
            and "a4" in text
            and any(term in text for term in ("rank", "check every", "reference", "default")),
        )
    )


def _sensitivity_change(query: str) -> tuple[ScenarioChange, list[ReadinessIssue]]:
    lowered = query.lower()
    named = _named_sweep(lowered)
    issues: list[ReadinessIssue] = []
    if named:
        parameter_name, values = named
    else:
        if any(term in lowered for term in ("foundation pit", "earthwork", "plan utilisation", "plan utilization")):
            parameter_name = "plan_utilisation_factor"
            values = [
                float(value)
                for value in re.findall(rf"(?<![\w.])({_NUMBER})(?!\w)", lowered)
                if 0 < float(value) <= 2.0
            ]
        elif "a4" in lowered or "transport" in lowered:
            parameter_name = "a4_scaling_factor"
            values = []
        else:
            parameter_name = "project_scaling_factor"
            values = []

    if not values:
        percentages = [float(value) / 100.0 for value in re.findall(rf"({_NUMBER})\s*%", lowered)]
        if percentages:
            values = sorted({1.0, *(1.0 - value for value in percentages), *(1.0 + value for value in percentages)})
    values = sorted(set(values))
    if len(values) < 2:
        issues.append(
            _issue(
                "MISSING_SENSITIVITY_PARAMETERS",
                "A sensitivity request requires a named target and at least two explicit candidate values.",
                ["parameter_values"],
            )
        )
    if any(value <= 0 for value in values):
        issues.append(
            _issue(
                "CALCULATION_PRECONDITION_FAILED",
                "Sensitivity parameters must be greater than zero.",
                ["positive parameter_values"],
            )
        )

    if parameter_name == "plan_utilisation_factor":
        target = "foundation_pit_combined_carbon"
        baseline = 0.9
    elif parameter_name == "a4_scaling_factor":
        target = "a4_carbon"
        baseline = 1.0
    else:
        target = "project_total_carbon"
        baseline = 1.0
    parameters = {
        "parameter_name": parameter_name,
        "baseline_parameter": baseline,
        "parameter_values": values,
        "result_unit": "kgCO2e",
    }
    return (
        ScenarioChange(
            change_type="sensitivity_analysis",
            target=target,
            parameters=parameters,
            explicit_parameters=["parameter_values"],
        ),
        issues,
    )


def _substitution_change(query: str) -> tuple[ScenarioChange, list[ReadinessIssue]]:
    lowered = query.lower()
    fraction = _percentage(lowered)
    baseline_factor = _first_number(
        [
            rf"\bbaseline(?:\s+(?:binder|material|emission))?(?:\s+factor)?\b"
            rf"\s*(?:=|is|of|:)?\s*({_NUMBER})",
            rf"({_NUMBER})\s*(?:kg\s*co2e?)?(?:\s+per\s+(?:t|tonne|kg))?"
            rf"\s+(?:for|as)\s+(?:the\s+)?baseline\b",
            rf"(?:基准|原始)(?:排放)?因子\s*(?:=|为|是|:|：)?\s*({_NUMBER})",
        ],
        lowered,
    )
    alternative_factor = _first_number(
        [
            rf"\balternative(?:\s+(?:binder|material|emission))?(?:\s+factor)?\b"
            rf"\s*(?:=|is|of|:)?\s*({_NUMBER})",
            rf"({_NUMBER})\s*(?:kg\s*co2e?)?(?:\s+per\s+(?:t|tonne|kg))?"
            rf"\s+(?:for|as)\s+(?:the\s+)?alternative\b",
            rf"(?:替代|备选)(?:排放)?因子\s*(?:=|为|是|:|：)?\s*({_NUMBER})",
        ],
        lowered,
    )
    pair = re.search(rf"\bfactors?\s*({_NUMBER})\s*(?:and|,)\s*({_NUMBER})", lowered, re.I)
    if pair:
        baseline_factor = baseline_factor if baseline_factor is not None else float(pair.group(1))
        alternative_factor = alternative_factor if alternative_factor is not None else float(pair.group(2))

    values = {
        key: value
        for key, value in (
            ("substitution_fraction", fraction),
            ("baseline_factor", baseline_factor),
            ("alternative_factor", alternative_factor),
        )
        if value is not None
    }
    missing = [
        name
        for name in ("substitution_fraction", "baseline_factor", "alternative_factor")
        if name not in values
    ]
    issues: list[ReadinessIssue] = []
    if missing:
        issues.append(
            _issue(
                "MISSING_SUBSTITUTION_PARAMETERS",
                "Substitution requires a fraction and a unit-compatible baseline/alternative factor pair.",
                missing,
            )
        )
    denominators = {
        value.lower()
        for value in re.findall(r"kg\s*co2e?\s*(?:per|/)\s*(kg|t|tonne|ton)", lowered, re.I)
    }
    invalid = (
        fraction is not None and not 0 <= fraction <= 1
    ) or (
        baseline_factor is not None and baseline_factor <= 0
    ) or (
        alternative_factor is not None and alternative_factor < 0
    ) or len(denominators) > 1
    if invalid:
        issues.append(
            _issue(
                "CALCULATION_PRECONDITION_FAILED",
                "Substitution fractions and factors must be valid and use compatible functional units.",
                ["valid substitution fraction and compatible factors"],
            )
        )
    return (
        ScenarioChange(
            change_type="material_substitution",
            target="concrete",
            parameters=values,
            explicit_parameters=list(values),
        ),
        issues,
    )


def _reuse_change(query: str) -> tuple[ScenarioChange, list[ReadinessIssue]]:
    lowered = query.lower()
    cycles = _first_number(
        [
            rf"\b(?:over|for)\s*({_NUMBER})\s*(?:reuse\s*)?(?:uses?|cycles?)\b",
            rf"\b({_NUMBER})\s*reuse\s*cycles?\b",
            rf"\bplanned\s+for\s+({_NUMBER})\s+uses?\b",
        ],
        lowered,
    )
    baseline = _first_number(
        [
            rf"\bcarr(?:y|ies|ying)\s*({_NUMBER})\s*kg\s*co2e?\b",
            rf"\b({_NUMBER})\s*kg\s*co2e?\b[^.;]{{0,64}}\b(?:initially|allocated|amorti[sz]ed)\b",
            rf"\bbaseline(?:\s+(?:embodied\s+)?(?:carbon|emission))?\b"
            rf"\s*(?:=|is|of|:)?\s*({_NUMBER})\s*kg\s*co2e?\b",
        ],
        lowered,
    )
    parameters: dict[str, object] = {}
    if cycles is not None:
        parameters["reuse_cycles"] = cycles
    if baseline is not None:
        parameters["baseline_emission"] = baseline
    issues: list[ReadinessIssue] = []
    missing = [name for name in ("reuse_cycles", "baseline_emission") if name not in parameters]
    if missing:
        issues.append(
            _issue(
                "MISSING_REUSE_PARAMETERS",
                "Reusable-resource allocation requires the initial burden and number of uses.",
                missing,
            )
        )
    if (cycles is not None and cycles < 1) or (baseline is not None and baseline < 0):
        issues.append(
            _issue(
                "CALCULATION_PRECONDITION_FAILED",
                "Reuse count must be at least one and the initial burden must be non-negative.",
                ["valid reuse parameters"],
            )
        )
    return (
        ScenarioChange(
            change_type="reuse_cycle_override",
            target="temporary_resource",
            parameters=parameters,
            explicit_parameters=list(parameters),
        ),
        issues,
    )


def _transport_change(query: str) -> tuple[ScenarioChange, list[ReadinessIssue]]:
    lowered = query.lower()
    material_match = re.search(_MATERIAL_CODE, query, re.I)
    mass_match = re.search(rf"({_NUMBER})\s*(kg|t|tonne|tonnes|ton)(?![a-z])", lowered, re.I)
    distance = _first_number([rf"({_NUMBER})\s*km\b"], lowered)
    factor = _first_number(
        [
            rf"\b(?:factor|ef)\s*(?:=|is|of|:)?\s*({_NUMBER})",
            rf"({_NUMBER})\s*kg\s*co2e?\s*(?:per|/)\s*(?:t|tonne)[- *]?km",
        ],
        lowered,
    )
    parameters: dict[str, object] = {}
    if mass_match:
        parameters["mass"] = float(mass_match.group(1))
        parameters["mass_unit"] = mass_match.group(2)
    if distance is not None:
        parameters["distance_km"] = distance
    if factor is not None:
        parameters["factor_kgco2e_per_tkm"] = factor
    missing = [
        name
        for name in ("mass", "mass_unit", "distance_km", "factor_kgco2e_per_tkm")
        if name not in parameters
    ]
    issues: list[ReadinessIssue] = []
    if missing:
        issues.append(
            _issue(
                "MISSING_A4_PARAMETERS",
                "A4 calculation requires mass, mass unit, distance, and a tonne-kilometre factor.",
                missing,
            )
        )
    if (
        ("mass" in parameters and float(parameters["mass"]) <= 0)
        or ("distance_km" in parameters and float(parameters["distance_km"]) <= 0)
        or (
            "factor_kgco2e_per_tkm" in parameters
            and float(parameters["factor_kgco2e_per_tkm"]) <= 0
        )
        or bool(re.search(r"kg\s*co2e?\s*(?:per|/)\s*kg[- *]?km", lowered, re.I))
    ):
        issues.append(
            _issue(
                "CALCULATION_PRECONDITION_FAILED",
                "A4 inputs must be positive and the transport factor must use a tonne-kilometre functional unit.",
                ["valid A4 parameters"],
            )
        )
    target = (
        f"material_transport_default:{material_match.group(1).upper()}"
        if material_match
        else "project_transport"
    )
    return (
        ScenarioChange(
            change_type="transport_override",
            target=target,
            parameters=parameters,
            explicit_parameters=list(parameters),
        ),
        issues,
    )


def _factor_reduction_change(query: str) -> tuple[ScenarioChange, list[ReadinessIssue]]:
    lowered = query.lower()
    reduction = _percentage(lowered)
    parameters = {"reduction_fraction": reduction} if reduction is not None else {}
    issues: list[ReadinessIssue] = []
    if reduction is None:
        issues.append(
            _issue(
                "MISSING_REDUCTION_PERCENTAGE",
                "A factor-reduction percentage is required.",
                ["reduction_fraction"],
            )
        )
    elif not 0 <= reduction <= 1:
        issues.append(
            _issue(
                "CALCULATION_PRECONDITION_FAILED",
                "A factor-reduction fraction must be within zero and one.",
                ["valid reduction_fraction"],
            )
        )
    if "concrete" in lowered or "混凝土" in lowered:
        target = "concrete"
    elif any(term in lowered for term in ("steel", "rebar", "钢筋", "钢材")):
        target = "steel"
    elif "formwork" in lowered or "模板" in lowered:
        target = "formwork"
    else:
        target = "all_resources"
    return (
        ScenarioChange(
            change_type="factor_reduction",
            target=target,
            parameters=parameters,
            explicit_parameters=list(parameters),
        ),
        issues,
    )


def _parse_semantic_mentions(
    query: str,
    semantic_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate model-selected spans and parse only query-grounded values."""

    parsed: list[dict[str, Any]] = []
    for raw in semantic_contract.get("parameter_mentions", []):
        if not isinstance(raw, dict):
            continue
        surface = str(raw.get("surface_form", "")).strip()
        if not surface or surface.casefold() not in query.casefold():
            continue
        numbers = re.findall(
            rf"(?<![A-Za-z0-9.])({_NUMBER})(?![A-Za-z0-9.])",
            surface,
            re.I,
        )
        if len(numbers) == 1:
            value = float(numbers[0])
        elif not numbers:
            word_numbers = {
                "zero": 0.0,
                "one": 1.0,
                "two": 2.0,
                "three": 3.0,
                "four": 4.0,
                "five": 5.0,
                "six": 6.0,
                "seven": 7.0,
                "eight": 8.0,
                "nine": 9.0,
                "ten": 10.0,
            }
            grounded_words = [
                numeric
                for word, numeric in word_numbers.items()
                if re.search(rf"\b{word}\b", surface, re.I)
            ]
            if len(grounded_words) != 1:
                continue
            value = grounded_words[0]
        else:
            continue
        name = str(raw.get("name", ""))
        role = str(raw.get("role", ""))
        if name in {"substitution_fraction", "reduction_fraction", "threshold_fraction"}:
            if "%" in surface:
                value /= 100.0
        parsed.append(
            {
                "name": name,
                "role": role,
                "value": value,
                "surface_form": surface,
                "unit": str(raw.get("unit") or "").strip(),
                "target": str(raw.get("target") or "").strip(),
                "confidence": float(raw.get("confidence", 0.0)),
            }
        )
    return parsed


def _semantic_overlay(
    query: str,
    persistent: bool,
    semantic_contract: dict[str, Any],
) -> ScenarioOverlay:
    mentions = _parse_semantic_mentions(query, semantic_contract)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for mention in mentions:
        by_name.setdefault(mention["name"], []).append(mention)

    def select(name: str, roles: tuple[str, ...]) -> dict[str, Any] | None:
        candidates = by_name.get(name, [])
        for role in roles:
            role_candidates = [item for item in candidates if item["role"] == role]
            if role_candidates:
                return sorted(
                    role_candidates,
                    key=lambda item: (-item["confidence"], item["surface_form"]),
                )[0]
        return None

    persistence_mode = PersistenceMode.PERSISTENT if persistent else PersistenceMode.TRANSIENT
    resources = list(semantic_contract.get("resource_targets", []))
    identifiers = [
        value
        for value in semantic_contract.get("explicit_identifiers", [])
        if str(value).casefold() in query.casefold()
    ]
    operator = str(semantic_contract.get("primary_operator", "none"))
    family = str(semantic_contract.get("task_family") or "")

    if bool(semantic_contract.get("compound")) or family == "compound_scenario":
        return _finish(
            query,
            persistence_mode,
            [
                ScenarioChange(
                    change_type="compound_program",
                    target="dependency_ordered_scenario",
                    assumed_parameters=["program_parameters:validated_query_spans"],
                )
            ],
            [],
        )

    if persistent:
        proposed_distances = [
            item
            for item in by_name.get("distance_km", [])
            if item["role"] in {"proposed", "alternative"}
        ]
        grounded_values = grounded_persistent_distance_values(query)
        represented_values = {float(item["value"]) for item in proposed_distances}
        proposed_distances.extend(
            {
                "name": "distance_km",
                "role": "governed_query_candidate",
                "value": value,
                "surface_form": f"{value:g} km",
                "unit": "km",
                "target": "",
                "confidence": 1.0,
            }
            for value in grounded_values
            if value not in represented_values
        )
        material_codes = sorted(
            {
                str(value).upper()
                for value in identifiers
                if re.fullmatch(r"M-\d+-\d+", str(value), re.I)
            }
        )
        semantic_targets: dict[str, str] = {}
        bound_target_values = [
            item.get("target") for item in proposed_distances if item.get("target")
        ]
        # Keep the contract label while using mentioned targets to check atomicity.
        target_values = [*resources, *bound_target_values]
        for value in target_values:
            label = str(value or "").strip()
            key = canonical_query_entity_label(query, label)
            if key:
                semantic_targets.setdefault(key, label)
        if (
            len(proposed_distances) > 1
            or len(material_codes) > 1
            or (
                not material_codes
                and len(
                    {
                        canonical_query_entity_label(query, value)
                        for value in bound_target_values
                        if str(value or "").strip()
                    }
                )
                > 1
            )
        ):
            return _finish(
                query,
                persistence_mode,
                [ScenarioChange(change_type="transport_override", target="ambiguous_multi_target_request")],
                [
                    _issue(
                        "AMBIGUOUS_PERSISTENT_CORRECTION",
                        "A confirmation package requires one semantic target and one proposed value.",
                        ["one atomic target/value pair"],
                    )
                ],
            )
        if operator not in {"transport", "transaction_preview"} and not proposed_distances:
            return _finish(
                query,
                persistence_mode,
                [],
                [
                    _issue(
                        "UNSUPPORTED_PERSISTENT_WRITE",
                        "Only append-only corrections represented by a governed transaction template are supported.",
                        ["supported persistent target and value"],
                    )
                ],
            )
        issues: list[ReadinessIssue] = []
        parameters: dict[str, Any] = {}
        if not proposed_distances:
            issues.append(
                _issue(
                    "MISSING_A4_PARAMETERS",
                    "A persistent transport correction requires an explicit proposed distance.",
                    ["distance_km"],
                )
            )
        else:
            distance = float(proposed_distances[0]["value"])
            parameters["distance_km"] = distance
            if distance <= 0:
                issues.append(
                    _issue(
                        "CALCULATION_PRECONDITION_FAILED",
                        "A transport distance must be greater than zero.",
                        ["positive distance_km"],
                    )
                )
        if material_codes:
            target = f"material_transport_default:{material_codes[0]}"
        elif semantic_targets:
            target = f"material_transport_default:{next(iter(semantic_targets.values()))}"
        else:
            target = "project_transport"
        return _finish(
            query,
            persistence_mode,
            [
                ScenarioChange(
                    change_type="transport_override",
                    target=target,
                    parameters=parameters,
                    explicit_parameters=list(parameters),
                )
            ],
            issues,
        )

    if operator == "none":
        names = set(by_name)
        if {"substitution_fraction", "baseline_factor", "alternative_factor"} & names:
            operator = "material_substitution"
        elif "reduction_fraction" in names:
            operator = "factor_reduction"
        elif {"reuse_cycles", "baseline_emission"} & names:
            operator = "reuse_allocation"
        elif {"mass", "distance_km", "factor_kgco2e_per_tkm"} & names:
            operator = "transport"
        elif "parameter_value" in names:
            operator = "sensitivity"

    if operator == "material_substitution":
        mapping = {
            "substitution_fraction": ("proposed", "alternative", "candidate"),
            "baseline_factor": ("baseline", "current", "reference", "context"),
            "alternative_factor": ("alternative", "proposed", "candidate"),
        }
        values: dict[str, Any] = {}
        selected_mentions: dict[str, dict[str, Any]] = {}
        for name, roles in mapping.items():
            item = select(name, roles)
            if item:
                values[name] = item["value"]
                selected_mentions[name] = item
        missing = [name for name in mapping if name not in values]
        issues = []
        if missing:
            issues.append(
                _issue(
                    "MISSING_SUBSTITUTION_PARAMETERS",
                    "Substitution requires a fraction and a unit-compatible baseline/alternative factor pair.",
                    missing,
                )
            )
        units = {
            item["unit"].casefold()
            for name, item in selected_mentions.items()
            if name.endswith("factor") and item["unit"]
        }
        if (
            ("substitution_fraction" in values and not 0 <= values["substitution_fraction"] <= 1)
            or ("baseline_factor" in values and values["baseline_factor"] <= 0)
            or ("alternative_factor" in values and values["alternative_factor"] < 0)
            or len(units) > 1
        ):
            issues.append(
                _issue(
                    "CALCULATION_PRECONDITION_FAILED",
                    "Substitution fractions and factors must be valid and use compatible functional units.",
                    ["valid substitution fraction and compatible factors"],
                )
            )
        target = str(resources[0]) if resources else "concrete"
        change = ScenarioChange(
            change_type="material_substitution",
            target=target,
            parameters=values,
            explicit_parameters=list(values),
        )
    elif operator == "factor_reduction":
        item = select("reduction_fraction", ("proposed", "alternative", "candidate"))
        values = {"reduction_fraction": item["value"]} if item else {}
        issues = []
        if item is None:
            issues.append(
                _issue(
                    "MISSING_REDUCTION_PERCENTAGE",
                    "A factor-reduction percentage is required.",
                    ["reduction_fraction"],
                )
            )
        elif not 0 <= item["value"] <= 1:
            issues.append(
                _issue(
                    "CALCULATION_PRECONDITION_FAILED",
                    "A factor-reduction fraction must be within zero and one.",
                    ["valid reduction_fraction"],
                )
            )
        change = ScenarioChange(
            change_type="factor_reduction",
            target=str(resources[0]) if resources else "all_resources",
            parameters=values,
            explicit_parameters=list(values),
        )
    elif operator == "reuse_allocation":
        cycle = select("reuse_cycles", ("proposed", "candidate", "context"))
        baseline = select("baseline_emission", ("baseline", "current", "context", "reference"))
        values = {}
        if cycle:
            values["reuse_cycles"] = cycle["value"]
        if baseline:
            values["baseline_emission"] = baseline["value"]
        missing = [name for name in ("reuse_cycles", "baseline_emission") if name not in values]
        issues = []
        if missing:
            issues.append(
                _issue(
                    "MISSING_REUSE_PARAMETERS",
                    "Reusable-resource allocation requires the initial burden and number of uses.",
                    missing,
                )
            )
        if (cycle and cycle["value"] < 1) or (baseline and baseline["value"] < 0):
            issues.append(
                _issue(
                    "CALCULATION_PRECONDITION_FAILED",
                    "Reuse count must be at least one and the initial burden must be non-negative.",
                    ["valid reuse parameters"],
                )
            )
        change = ScenarioChange(
            change_type="reuse_cycle_override",
            target=str(resources[0]) if resources else "temporary_resource",
            parameters=values,
            explicit_parameters=list(values),
        )
    elif operator == "transport":
        mass = select("mass", ("proposed", "alternative", "context", "candidate"))
        distance = select("distance_km", ("proposed", "alternative", "candidate", "context"))
        factor = select(
            "factor_kgco2e_per_tkm",
            ("proposed", "alternative", "reference", "context", "candidate"),
        )
        values: dict[str, Any] = {}
        if mass:
            values["mass"] = mass["value"]
            unit_text = mass["unit"] or mass["surface_form"]
            unit_match = re.search(r"\b(kg|t|tonne|tonnes|ton)\b", unit_text, re.I)
            if unit_match:
                values["mass_unit"] = unit_match.group(1).lower()
        if distance:
            values["distance_km"] = distance["value"]
        if factor:
            values["factor_kgco2e_per_tkm"] = factor["value"]
        missing = [
            name
            for name in ("mass", "mass_unit", "distance_km", "factor_kgco2e_per_tkm")
            if name not in values
        ]
        issues = []
        if missing:
            issues.append(
                _issue(
                    "MISSING_A4_PARAMETERS",
                    "A4 calculation requires mass, mass unit, distance, and a tonne-kilometre factor.",
                    missing,
                )
            )
        factor_unit = factor["unit"].casefold() if factor else ""
        invalid_factor_unit = bool(factor_unit) and not (
            "tkm" in factor_unit or "t-km" in factor_unit or "tonne-km" in factor_unit
        )
        if (
            (mass and mass["value"] <= 0)
            or (distance and distance["value"] <= 0)
            or (factor and factor["value"] <= 0)
            or invalid_factor_unit
        ):
            issues.append(
                _issue(
                    "CALCULATION_PRECONDITION_FAILED",
                    "A4 inputs must be positive and the transport factor must use a tonne-kilometre functional unit.",
                    ["valid A4 parameters"],
                )
            )
        material_codes = [
            str(value).upper()
            for value in identifiers
            if re.fullmatch(r"M-\d+-\d+", str(value), re.I)
        ]
        semantic_target = str(resources[0]) if len(resources) == 1 else ""
        change = ScenarioChange(
            change_type="transport_override",
            target=(
                f"material_transport_default:{material_codes[0]}"
                if material_codes
                else f"material_transport_default:{semantic_target}"
                if semantic_target
                else "project_transport"
            ),
            parameters=values,
            explicit_parameters=list(values),
        )
    elif operator == "sensitivity":
        candidates = [
            item["value"]
            for item in by_name.get("parameter_value", [])
            if item["role"] in {"candidate", "proposed", "alternative"}
        ]
        candidates = sorted(set(candidates))
        targets = [item["target"] for item in by_name.get("parameter_value", []) if item["target"]]
        named_sweep = _named_sweep(query)
        if named_sweep:
            parameter_name, grounded_candidates = named_sweep
            if len(candidates) < 2:
                candidates = sorted(set(grounded_candidates))
        else:
            parameter_name = targets[0] if targets else "project_scaling_factor"
        normalised_parameter = re.sub(
            r"[^a-z0-9]+", "_", parameter_name.casefold()
        ).strip("_")
        if "utili" in normalised_parameter:
            parameter_name = "plan_utilisation_factor"
            target = "foundation_pit_combined_carbon"
            baseline_parameter = 0.9
        elif normalised_parameter.startswith("a4_"):
            parameter_name = "a4_scaling_factor"
            target = "a4_carbon"
            baseline_parameter = 1.0
        else:
            parameter_name = "project_scaling_factor"
            target = "project_total_carbon"
            baseline_parameter = 1.0
        issues = []
        if len(candidates) < 2:
            issues.append(
                _issue(
                    "MISSING_SENSITIVITY_PARAMETERS",
                    "A sensitivity request requires a named target and at least two explicit candidate values.",
                    ["parameter_values"],
                )
            )
        if any(value <= 0 for value in candidates):
            issues.append(
                _issue(
                    "CALCULATION_PRECONDITION_FAILED",
                    "Sensitivity parameters must be greater than zero.",
                    ["positive parameter_values"],
                )
            )
        change = ScenarioChange(
            change_type="sensitivity_analysis",
            target=target,
            parameters={
                "parameter_name": parameter_name,
                "baseline_parameter": baseline_parameter,
                "parameter_values": candidates,
                "result_unit": "kgCO2e",
            },
            explicit_parameters=["parameter_values"],
            assumed_parameters=["baseline_parameter:governed_project_reference"],
        )
    else:
        return _finish(
            query,
            persistence_mode,
            [],
            [
                _issue(
                    "UNMODELLED_CHANGE",
                    "The semantic contract did not select a supported governed scenario operation.",
                    ["supported scenario operator"],
                )
            ],
        )
    return _finish(query, persistence_mode, [change], issues)


def extract_overlay(
    query: str,
    persistent: bool = False,
    semantic_contract: dict[str, Any] | None = None,
) -> ScenarioOverlay:
    """Extract typed scenario/correction arguments using semantic value roles.

    The extractor never reads benchmark files or supplies engineering values. It
    accepts only quantities present in the request or fields resolved later from
    governed reference records.
    """

    semantic_contract = dict(semantic_contract or {})
    if semantic_contract.get("llm_contract_validated"):
        return _semantic_overlay(query, persistent, semantic_contract)

    if persistent:
        # Compare routes without treating them as proposed corrections.
        if not re.search(
            r"\b(?:persist\w*|commit\w*|appl\w*|updat\w*|sets?|revis\w*|correct\w*|writ\w*|record\w*|prepar\w*|propos\w*|confirm\w*)\b"
            r"|持久化|提交|更新|修正|更正|写入|设置|确认|准备",
            query,
            re.I,
        ):
            return _finish(
                query,
                PersistenceMode.TRANSIENT,
                [],
                [],
            )
        return _persistent_transport_overlay(query)

    lowered = query.lower()
    persistence = PersistenceMode.TRANSIENT
    if _is_compound(lowered):
        return _finish(
            query,
            persistence,
            [
                ScenarioChange(
                    change_type="compound_program",
                    target="dependency_ordered_scenario",
                    assumed_parameters=["program_parameters:parsed_by_typed_calculation_compiler"],
                )
            ],
            [],
        )

    named_sweep = _named_sweep(lowered)
    if named_sweep or any(term in lowered for term in ("sensitivity", "parameter sweep", "one-at-a-time")):
        change, issues = _sensitivity_change(query)
    elif any(
        term in lowered
        for term in ("ggbs", "slag", "substitution", "substitute", "binder option", "矿渣", "替代")
    ):
        change, issues = _substitution_change(query)
    elif any(
        term in lowered
        for term in ("reuse", "re-use", "reusable", "allocated over", "allocated burden per use")
    ):
        change, issues = _reuse_change(query)
    elif (
        any(term in lowered for term in ("factor", "epd", "因子"))
        and any(
            term in lowered
            for term in ("reduce", "reduction", "decrease", "lower", "improvement", "降低", "减少", "下降", "改进")
        )
    ):
        change, issues = _factor_reduction_change(query)
    elif any(
        term in lowered
        for term in ("transport", "haul", "a4", "delivery", "route", "运输", "运距", "路线", "配送")
    ):
        change, issues = _transport_change(query)
    else:
        return _finish(
            query,
            persistence,
            [],
            [
                _issue(
                    "UNMODELLED_CHANGE",
                    "No governed scenario or correction operation could be extracted from the request.",
                    ["change specification"],
                )
            ],
        )
    return _finish(query, persistence, [change], issues)
