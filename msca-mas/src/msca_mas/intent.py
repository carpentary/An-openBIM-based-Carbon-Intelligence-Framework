from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .models import QueryMode


IntentKind = Literal[
    "read_existing",
    "derive_from_existing",
    "hypothetical_analysis",
    "persistent_change",
    "boundary_validation",
    "ambiguous",
]


class GovernedIntentSignals(BaseModel):
    """High-precision, answer-free signals used at the planning safety boundary.

    These signals describe the requested operation.  They never contain a
    benchmark answer, an engineering quantity, a SPARQL statement, or a graph
    mutation.  The rule result is one independent input to semantic arbitration;
    it is not an unconditional replacement for the planning model.
    """

    request_kind: IntentKind
    rule_mode: QueryMode
    read_only: bool
    existing_data_requested: bool
    calculation_requested: bool
    scenario_requested: bool
    persistence_requested: bool
    outside_boundary_referenced: bool
    ontology_design_information_request: bool
    boundary_information_request: bool
    out_of_boundary_operation_requested: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


_PERSISTENCE_PATTERNS = (
    r"\b(?:persist|commit)\b",
    r"\bappend[- ]only\b.{0,64}\b(?:correction|transaction|update|delta|preview)\b",
    r"\b(?:correction|transaction|update)\s+preview\b",
    r"\bconfirmation[- ]gated\b.{0,48}\b(?:correction|transaction|write|preview)\b",
    r"\b(?:set|sets|revise|replace|correct)\b.{0,96}\b(?:future|default|lookup|record)\b",
    r"\bwrite\b.{0,48}\b(?:graph|ckg|ontology|record|default)\b",
    r"\b(?:update|correct|edit|delete|remove|insert)\b.{0,48}\b(?:graph|ckg|ontology|record|default)\b",
    r"\b(?:add|replace|change|set)\b.{0,48}\b(?:in|into|to|from|as)\b.{0,32}\b(?:graph|ckg|ontology|record|default)\b",
    r"\bpermanently\b",
)

_SCENARIO_PATTERNS = (
    r"\bwhat[ -]?if\b",
    r"\bhypothetical(?:ly)?\b",
    r"\bsimulate\b",
    r"\bsensitivity (?:analysis|sweep)\b",
    r"\bparameter sweep\b",
    r"\b(?:run|perform|conduct|simulate)\b.{0,32}\b(?:scenario|sensitivity|sweep|simulation)\b",
    r"\b(?:design|supplier|material|transport) alternative\b",
    r"\b(?:substitut(?:e|ed|ion)|reduc(?:e|ed|tion)|increas(?:e|ed)|decreas(?:e|ed)|amorti[sz]e|reuse cycles?)\b",
    r"\b(?:proposed|counterfactual|custom supplier)\b",
    r"\b(?:option|proposal|temporary|transient|planned)\b.{0,160}\b(?:carbon|emission|factor|a4|transport|reuse|substitut|ggbs|scaling|utili[sz]ation)\b",
    r"\b(?:coupled|combined)\s+(?:option|measure|alternative|scenario)\b",
    r"\b(?:test|screen|evaluate|vary)\b.{0,96}\b(?:factor|scaling|utili[sz]ation|one[- ]at[- ]a[- ]time)\b",
    r"\b(?:allocated?\s+over|planned\s+for)\s+\d+(?:\.\d+)?\s+(?:uses?|cycles?)\b",
    r"\bif\b.{0,80}\b(?:chang(?:e|ed)|replac(?:e|ed)|reduc(?:e|ed)|increas(?:e|ed)|decreas(?:e|ed)|substitut(?:e|ed)|use instead)\b",
)

_CALCULATION_PATTERNS = (
    r"\b(?:calculate|compute|recalculate|recompute|aggregate|sum|add|added|rank|compare)\b",
    r"\b(?:share|percentage|proportion|ratio|intensity|delta|difference|multiplier)\b",
    r"\b(?:decision|priority|carbon)\s+order\b",
)

_EXISTING_RESULT_PATTERNS = (
    r"\b(?:stored|recorded|existing|current|governed|reported|displayed|baseline)\b",
    r"\b(?:retrieve|show|list|report|trace|audit|look up|lookup)\b",
    r"\bsensitivity results?\b",
)

_BOUNDARY_PATTERNS = (
    r"\bmodule\s+(?:b[1-7]|c[1-4]|d)\b",
    r"(?<![a-z0-9])(?:b[1-7]|c[1-4])(?![a-z0-9])",
    r"\b(?:operation(?:al)?(?:[- ]use)? stage|end[- ]of[- ]life|deconstruction|demolition|recycling benefit)\b",
)

_BOUNDARY_INFORMATION_PATTERNS = (
    r"\b(?:which|what)\b.{0,64}\b(?:stage|module|boundary|included|excluded)\b",
    r"\b(?:is|are|whether)\b.{0,64}\b(?:included|excluded|inside|outside|within|supported|applicable)\b",
    r"\b(?:scope|boundary|applicability|coverage)\b",
)


def _normalise(query: str) -> str:
    text = query.lower()
    for value in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(value, "-")
    return re.sub(r"\s+", " ", text).strip()


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _explicit_numeric_scenario(text: str) -> bool:
    """Detect a new parameterised calculation without treating stored totals as SA."""

    parameter_mentions = re.findall(
        r"(?<![a-z0-9])\d+(?:\.\d+)?\s*(?:%|km|t(?:onne)?s?|kg|m[23]|kwh|mj|kgco2e)\b",
        text,
        re.I,
    )
    factor_mentions = re.findall(
        r"\b(?:factor|rate|distance|quantity|mass|volume|productivity)\b\s*(?:=|of|to)?\s*\d+(?:\.\d+)?",
        text,
        re.I,
    )
    return len(parameter_mentions) + len(factor_mentions) >= 2


def _named_parameter_sweep(text: str) -> bool:
    """Recognise a proposed value list without requiring the word sensitivity."""

    match = re.search(
        r"\b(?:plan[ _-]*utili[sz]ation|a4[ _-]*scaling|project[ _-]*scaling)[ _-]*factor\b"
        r".{0,48}?\bat\s+([^.;]+)",
        text,
        re.I,
    )
    if not match:
        return False
    return len(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", match.group(1))) >= 2


def analyse_query_intent(query: str) -> GovernedIntentSignals:
    """Return generic semantic-operation signals for routing and safety checks."""

    text = _normalise(query)
    evidence: list[str] = []
    persistence = _matches(text, _PERSISTENCE_PATTERNS) or any(
        term in query
        for term in ("写入", "提交", "永久", "持久化", "更正知识图谱", "修正知识图谱", "更新知识图谱", "删除事实")
    )
    calculation = _matches(text, _CALCULATION_PATTERNS) or any(
        term in query for term in ("计算", "核算", "聚合", "合计", "比较", "排序", "占比", "比例")
    )
    existing = _matches(text, _EXISTING_RESULT_PATTERNS) or any(
        term in query for term in ("检索", "查询", "当前", "已有", "现有", "追溯", "审计", "结果")
    )
    outside = _matches(text, _BOUNDARY_PATTERNS) or any(
        term in query for term in ("运营阶段", "运营使用", "拆除阶段", "回收收益", "边界外")
    )
    boundary_information = outside and _matches(text, _BOUNDARY_INFORMATION_PATTERNS)
    meta_ontology_request = bool(
        re.search(
            r"\b(?:query[- ]specific )?(?:subontology|ontology term|ontology content|semantic path|schema concept)s?\b",
            text,
        )
        or re.search(
            r"\bwhich\b.{0,96}\b(?:question[- ]specific )?(?:evidence subset|evidence package)\b",
            text,
        )
    )

    stored_sensitivity = bool(
        re.search(r"\b(?:stored|recorded|existing|current|reported)?\s*sensitivity results?\b", text)
    ) and not bool(re.search(r"\b(?:run|perform|conduct|new|recompute)\b", text))
    stored_sensitivity = stored_sensitivity or (
        "敏感性" in query
        and "结果" in query
        and not any(term in query for term in ("运行", "执行", "重新计算", "分析"))
    )
    explicit_a4_calculation = bool(
        re.match(r"^\s*(?:calculate|compute)\b", text)
        and re.search(r"(?<![a-z0-9])a4(?![a-z0-9])|\btransport\b", text)
    )
    explicit_a4_parameters = bool(
        re.search(r"(?<![a-z0-9])a4(?![a-z0-9])|\btransport\b", text)
        and _explicit_numeric_scenario(text)
    )
    scenario = _matches(text, _SCENARIO_PATTERNS) or explicit_a4_calculation or explicit_a4_parameters or _named_parameter_sweep(text) or any(
        term in query for term in ("如果", "假设", "情景", "模拟", "敏感性分析", "替代", "减排")
    )
    if stored_sensitivity:
        scenario = False
        existing = True

    out_of_boundary_operation = outside and (
        persistence
        or scenario
        or calculation
        or bool(re.search(r"\b(?:calculate|compute|simulate|estimate|add|update|change)\b", text))
    )

    request_kind: IntentKind
    if meta_ontology_request and not persistence:
        request_kind = "read_existing"
        rule_mode = QueryMode.INFORMATION_RETRIEVAL
        confidence = 0.98
        scenario = False
        evidence.append("ontology_design_information_request")
    elif persistence:
        request_kind = "persistent_change"
        rule_mode = QueryMode.KNOWLEDGE_CORRECTION
        confidence = 0.99
        evidence.append("explicit_persistence_operation")
    elif outside:
        request_kind = "boundary_validation"
        rule_mode = QueryMode.INFORMATION_RETRIEVAL
        confidence = 0.98
        evidence.append(
            "out_of_boundary_operation"
            if out_of_boundary_operation
            else "boundary_information_request"
        )
    elif scenario:
        request_kind = "hypothetical_analysis"
        rule_mode = QueryMode.STRATEGIC_ANALYSIS
        confidence = 0.93
        evidence.append("counterfactual_or_parameterised_operation")
    elif calculation:
        request_kind = "derive_from_existing"
        rule_mode = QueryMode.INFORMATION_RETRIEVAL
        confidence = 0.86
        evidence.append("derivation_over_governed_existing_data")
    else:
        request_kind = "read_existing"
        rule_mode = QueryMode.INFORMATION_RETRIEVAL
        confidence = 0.78
        evidence.append("governed_read_only_request")

    if existing:
        evidence.append("existing_or_stored_data_signal")
    if calculation:
        evidence.append("calculation_or_comparison_signal")

    return GovernedIntentSignals(
        request_kind=request_kind,
        rule_mode=rule_mode,
        read_only=not persistence and not scenario,
        existing_data_requested=existing or (not persistence and not scenario),
        calculation_requested=calculation,
        scenario_requested=scenario,
        persistence_requested=persistence,
        outside_boundary_referenced=outside,
        ontology_design_information_request=meta_ontology_request and not persistence,
        boundary_information_request=boundary_information and not out_of_boundary_operation,
        out_of_boundary_operation_requested=out_of_boundary_operation,
        confidence=confidence,
        evidence=list(dict.fromkeys(evidence)),
    )
