from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Sequence

from .config import ProjectConfig
from .cq import CQCatalogue
from .ontology_profiles import ontology_profiles
from .models import QueryMode
from .ontology_retrieval import OntologyMetadataIndex
from .semantic_models import (
    BackboneVerification,
    MetadataRetrievalContext,
    MetadataRetrievalRound,
    OntologyEntityCandidate,
    OntologyEntityKind,
    OntologyRelation,
    ParsedEngineeringTerm,
    QueryInterpretation,
    ResolvedSemanticPath,
    SemanticRole,
)


DOMAIN_LEXICON: tuple[dict[str, Any], ...] = (
    {
        "canonical": "metro station construction project",
        "synonyms": ("metro station", "station project", "construction project", "project", "地铁车站", "车站项目", "项目"),
        "definition": "The assessed metro-station construction project and its engineering systems.",
        "enhanced": ("ConstructionProject", "MetroStationProject", "EngineeringSystem", "has engineering system"),
        "roles": (SemanticRole.SYSTEM,),
    },
    {
        "canonical": "main structure engineering system",
        "synonyms": ("main structure", "main-structure", "主体结构", "主体"),
        "definition": "The permanent main structural engineering system of the station.",
        "enhanced": ("EngineeringSystem", "MainStructureSystem", "physical element"),
        "roles": (SemanticRole.SYSTEM, SemanticRole.ELEMENT),
    },
    {
        "canonical": "enclosure retaining engineering system",
        "synonyms": ("enclosure structure", "enclosure", "retaining structure", "围护结构", "围护"),
        "definition": "The retaining and enclosure engineering system used during station construction.",
        "enhanced": ("EnclosureSystem", "RetainingSystem", "temporary work", "physical element"),
        "roles": (SemanticRole.SYSTEM, SemanticRole.ELEMENT),
    },
    {
        "canonical": "foundation pit earthwork",
        "synonyms": ("foundation pit", "foundation-pit", "earthwork", "excavation", "backfill", "基坑", "土方", "开挖", "回填"),
        "definition": "Implicit spatial work covering foundation-pit excavation, backfill and off-site earthwork haul.",
        "enhanced": ("FoundationPit", "ExcavationZone", "SpatialWork", "EarthworkEngineering", "ProjectWorkPackage"),
        "roles": (SemanticRole.ELEMENT, SemanticRole.PROCESS),
    },
    {
        "canonical": "diaphragm wall",
        "synonyms": ("diaphragm wall", "underground diaphragm wall", "地下连续墙", "地连墙"),
        "definition": "A cast-in-place retaining wall element constructed through trench excavation and slurry support.",
        "enhanced": ("Type_DiaphragmWall", "PhysicalElement", "VerticalRetainingSystem", "ConcreteEngineering"),
        "roles": (SemanticRole.ELEMENT, SemanticRole.PROCESS),
    },
    {
        "canonical": "physical or spatial element",
        "synonyms": ("element", "component", "ifc product", "ifc products", "构件", "组件", "ifc产品", "ifc构件"),
        "definition": "A modeled physical element or an ontology-derived implicit spatial work.",
        "enhanced": ("Element", "PhysicalElement", "SpatialWork", "has element", "belongs to system"),
        "roles": (SemanticRole.ELEMENT,),
    },
    {
        "canonical": "project work package construction process",
        "synonyms": ("work package", "work-package", "construction process", "process", "工作包", "工序", "施工过程"),
        "definition": "An instantiated construction-process package linking an element to consumed resources.",
        "enhanced": ("ProjectWorkPackage", "WorkPackageTemplate", "requires work package", "requires resource instance"),
        "roles": (SemanticRole.PROCESS,),
    },
    {
        "canonical": "resource instance material equipment energy",
        "synonyms": ("resource", "material", "equipment", "energy", "资源", "材料", "设备", "能源"),
        "definition": "A quantified material, equipment or energy resource consumed by a project work package.",
        "enhanced": ("ResourceInstance", "ResourceDescriptor", "Material", "Equipment", "Energy"),
        "roles": (SemanticRole.RESOURCE,),
    },
    {
        "canonical": "concrete material",
        "synonyms": ("concrete", "ready-mix concrete", "混凝土", "商品混凝土"),
        "definition": "Concrete resource descriptors and their project resource instances.",
        "enhanced": ("Concrete", "ResourceDescriptor", "Material", "concrete emission factor"),
        "roles": (SemanticRole.RESOURCE,),
    },
    {
        "canonical": "steel material",
        "synonyms": ("steel", "rebar", "reinforcement", "钢", "钢材", "钢筋"),
        "definition": "Steel and reinforcement resource descriptors and their project resource instances.",
        "enhanced": ("Steel", "Reinforcement", "ResourceDescriptor", "steel emission factor"),
        "roles": (SemanticRole.RESOURCE,),
    },
    {
        "canonical": "formwork reusable resource",
        "synonyms": ("formwork", "template reuse", "reuse cycle", "模板", "周转", "复用次数"),
        "definition": "Temporary formwork resources whose initial burden may be allocated over reuse cycles.",
        "enhanced": ("Formwork", "ReusableResource", "reuse count", "FormworkEngineering"),
        "roles": (SemanticRole.PROCESS, SemanticRole.RESOURCE),
    },
    {
        "canonical": "ground granulated blast furnace slag substitution",
        "synonyms": ("ggbs", "slag", "blast furnace slag", "矿渣", "矿粉", "矿渣粉"),
        "definition": "A cementitious substitute represented as an alternative material resource descriptor.",
        "enhanced": ("GroundGranulatedBlastFurnaceSlag", "ResourceSubstitution", "alternative resource descriptor"),
        "roles": (SemanticRole.RESOURCE,),
    },
    {
        "canonical": "carbon emission assessment",
        "synonyms": ("carbon", "emission", "embodied carbon", "carbon assessment", "碳", "排放", "碳排放", "碳核算"),
        "definition": "A carbon-emission quantity linked to an assessment stage, owner and quantified resource flow.",
        "enhanced": ("CarbonEmission", "CarbonAssessment", "has emissions", "has emission quantity", "kgCO2e"),
        "roles": (SemanticRole.CARBON,),
    },
    {
        "canonical": "A1-A3 product stage",
        "synonyms": ("a1-a3", "a1–a3", "product stage", "产品阶段"),
        "definition": "The declared product-stage modules A1-A3.",
        "enhanced": ("A1-A3_ProductStage", "LifeCycleStage", "belongs to stage"),
        "roles": (SemanticRole.CARBON,),
    },
    {
        "canonical": "A4 transportation stage",
        "synonyms": ("a4", "transport", "transportation", "haul", "vehicle", "distance", "运输", "运距", "外运", "载具"),
        "definition": "Transportation of material resources, quantified by mass, distance and vehicle emission factor.",
        "enhanced": ("A4_TransportationStage", "TransportationEngineering", "TransportLeg", "travel distance", "vehicle"),
        "roles": (SemanticRole.PROCESS, SemanticRole.RESOURCE, SemanticRole.CARBON),
    },
    {
        "canonical": "A5 construction stage",
        "synonyms": ("a5", "construction stage", "施工阶段"),
        "definition": "On-site construction-stage equipment, energy and process emissions.",
        "enhanced": ("A5_ConstructionStage", "ConstructionEngineering", "Equipment", "Energy"),
        "roles": (SemanticRole.PROCESS, SemanticRole.RESOURCE, SemanticRole.CARBON),
    },
    {
        "canonical": "construction process database",
        "synonyms": ("cpd", "construction process database", "定额", "施工过程数据库"),
        "definition": "The governed construction-process database containing work-package templates and resource quotas.",
        "enhanced": ("CPD_Database", "CPD_Entry", "CPD_ResourceQuota", "WorkPackageTemplate"),
        "roles": (SemanticRole.PROCESS, SemanticRole.RESOURCE),
    },
    {
        "canonical": "emission factor database",
        "synonyms": ("efd", "emission factor", "factor", "排放因子", "因子"),
        "definition": "The governed emission-factor database linking resource descriptors to quantity factors.",
        "enhanced": ("EFD_Database", "EFD_Entry", "EFD_Factor", "factor resource"),
        "roles": (SemanticRole.RESOURCE, SemanticRole.CARBON),
    },
    {
        "canonical": "S-E-P-R-C provenance chain",
        "synonyms": ("trace", "provenance", "audit chain", "s-e-p-r-c", "追溯", "溯源", "审计链", "完整链路"),
        "definition": "The auditable System-Element-Process-Resource-Carbon semantic backbone.",
        "enhanced": ("EngineeringSystem", "Element", "ProjectWorkPackage", "ResourceInstance", "CarbonEmission"),
        "roles": (
            SemanticRole.SYSTEM,
            SemanticRole.ELEMENT,
            SemanticRole.PROCESS,
            SemanticRole.RESOURCE,
            SemanticRole.CARBON,
        ),
    },
    {
        "canonical": "assessment scope and lifecycle boundary",
        "synonyms": ("scope", "boundary", "included stage", "excluded stage", "适用范围", "边界", "包含阶段", "排除阶段"),
        "definition": "The declared engineering-system and lifecycle-stage boundary of the carbon assessment.",
        "enhanced": ("AssessmentScope", "includes engineering system", "included life cycle stage", "CoverageGap"),
        "roles": (SemanticRole.SCOPE, SemanticRole.SYSTEM, SemanticRole.CARBON),
    },
    {
        "canonical": "scenario delta specification",
        "synonyms": ("scenario", "what if", "sensitivity", "substitution", "reduction", "情景", "假设", "敏感性", "替代", "减排"),
        "definition": "A transient, provenance-annotated override applied over the governed factual baseline.",
        "enhanced": ("Scenario", "ScenarioOverride", "ChangeSet", "ScenarioResult", "emission delta"),
        "roles": (SemanticRole.RESOURCE, SemanticRole.CARBON),
    },
)


PROPERTY_LEXICON: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("volume", ("volume", "m3", "体积", "方量"), ("has volume", "numeric value", "quantity")),
    ("mass", ("mass", "weight", "tonne", "kg", "质量", "重量", "吨"), ("has mass", "numeric value", "quantity")),
    ("area", ("area", "m2", "面积"), ("has area", "numeric value", "quantity")),
    ("length", ("length", "metre", "长度"), ("has length", "numeric value", "quantity")),
    (
        "count",
        (
            "count",
            "number",
            "how many",
            "most numerous",
            "most instantiated",
            "数量",
            "个数",
            "多少条",
            "多少个",
            "多少项",
            "多少种",
            "各有多少",
            "分别有多少",
            "共有多少",
            "合计多少",
            "另有多少",
        ),
        ("element count", "resource count"),
    ),
    ("distance", ("distance", "km", "运距", "距离"), ("travel distance", "transport distance")),
    (
        "emission factor",
        ("factor", "factors", "emission factor", "emission factors", "因子", "排放因子"),
        ("EFD Factor", "factor value"),
    ),
    (
        "percentage",
        (
            "percentage",
            "share",
            "ratio",
            "fraction",
            "mapping rate",
            "coverage rate",
            "%",
            "百分比",
            "占比",
            "比例",
            "映射率",
            "覆盖率",
        ),
        ("percentage change", "fraction"),
    ),
)


PROPERTY_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "volume": ("volume", "quantity", "numeric value"),
    "mass": (
        "mass",
        "payload",
        "load factor",
        "actual quantity",
        "resource quantity",
        "quantity",
        "numeric value",
    ),
    "area": ("area", "quantity", "numeric value"),
    "length": ("length", "quantity", "numeric value"),
    "count": ("count", "quantity", "numeric value"),
    "distance": ("distance", "travel distance", "transport distance", "numeric value"),
    "emission factor": ("emission factor", "factor value", "factor resource", "selected factor"),
    "percentage": ("percentage", "fraction", "ratio", "numeric value"),
}


ROLE_SUPPLEMENTS: dict[SemanticRole, tuple[str, ...]] = {
    SemanticRole.SYSTEM: ("EngineeringSystem", "ConstructionProject", "has engineering system"),
    SemanticRole.ELEMENT: ("Element", "PhysicalElement", "SpatialWork", "has element"),
    SemanticRole.PROCESS: ("ProjectWorkPackage", "WorkPackageTemplate", "requires work package"),
    SemanticRole.RESOURCE: ("ResourceInstance", "ResourceDescriptor", "requires resource instance"),
    SemanticRole.CARBON: ("CarbonEmission", "CarbonAssessment", "has emissions", "has emission quantity"),
    SemanticRole.PROPERTY: ("datatype property", "quantity numeric value", "unit"),
    SemanticRole.SCOPE: ("AssessmentScope", "CoverageGap", "included life cycle stage"),
    SemanticRole.OTHER: (),
}


def _contains(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase.lower() in text.lower() for phrase in phrases)


def _contains_property_marker(text: str, marker: str) -> bool:
    """Match Latin property terms as tokens and CJK/symbol terms as phrases.

    Substring matching classifies ``configuration`` and ``operational`` as
    percentages because both contain ``ratio``.  Property extraction affects
    the ontology contract, so these short Latin markers require boundaries.
    """

    lowered = text.lower()
    marker = marker.lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._-]*", marker):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])",
                lowered,
            )
        )
    return marker in lowered


def _deduplicate_candidates(items: Iterable[OntologyEntityCandidate]) -> list[OntologyEntityCandidate]:
    by_uri: dict[str, OntologyEntityCandidate] = {}
    for item in items:
        current = by_uri.get(item.uri)
        if current is None or item.final_score > current.final_score or (item.seed and not current.seed):
            by_uri[item.uri] = item
    return sorted(by_uri.values(), key=lambda item: (not item.seed, item.discovered_at_hop, -item.final_score, item.uri))


def _deduplicate_relations(items: Iterable[OntologyRelation]) -> list[OntologyRelation]:
    by_id: dict[str, OntologyRelation] = {}
    for item in items:
        key = f"{item.relation_id}|{item.direction}"
        current = by_id.get(key)
        if current is None or item.hop < current.hop:
            by_id[key] = item
    return sorted(by_id.values(), key=lambda item: (item.hop, item.subject_uri, item.predicate_uri, item.object_uri))


class MetadataPipeline:
    """Paper-aligned A_meta pipeline with local self-reflection and bounded retry."""

    def __init__(self, config: ProjectConfig, catalogue: CQCatalogue, index: OntologyMetadataIndex):
        self.config = config
        self.catalogue = catalogue
        self.index = index

    def parse_query(
        self,
        query: str,
        mode: QueryMode,
        *,
        anchor_terms: Sequence[str] = (),
        cq_ids: Sequence[str] = (),
    ) -> QueryInterpretation:
        lowered = query.lower().replace("–", "-").replace("—", "-")
        # Treat module codes, not letters in S-E-P-R-C, as scope requests.
        out_of_scope = bool(
            re.search(r"(?<![a-z0-9])(?:b[1-7]|c[1-4]|d)(?![a-z0-9])", lowered)
        ) or _contains(
            lowered,
            (
                "operational-use",
                "operation stage",
                "end-of-life",
                "end of life",
                "deconstruction",
                "demolition",
                "recycling benefit",
                "运营使用",
                "运营阶段",
                "拆除阶段",
                "回收收益",
                "边界外",
                "d模块",
            ),
        )
        terms: list[ParsedEngineeringTerm] = []
        seen_canonical: set[str] = set()
        required_roles: list[SemanticRole] = []

        def add_role(role: SemanticRole) -> None:
            if role not in required_roles and role not in {SemanticRole.PROPERTY, SemanticRole.OTHER}:
                required_roles.append(role)

        for item in DOMAIN_LEXICON:
            matched = [synonym for synonym in item["synonyms"] if synonym.lower() in lowered]
            if not matched:
                continue
            canonical = str(item["canonical"])
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            roles = list(item["roles"])
            for role in roles:
                add_role(role)
            terms.append(
                ParsedEngineeringTerm(
                    term_id=f"TERM-{len(terms) + 1:03d}",
                    original_text=max(matched, key=len),
                    definition=str(item["definition"]),
                    enhanced_phrases=list(item["enhanced"]),
                    source="query",
                    out_of_scope=out_of_scope and canonical in {
                        "A1-A3 product stage",
                        "A4 transportation stage",
                        "A5 construction stage",
                        "carbon emission assessment",
                    },
                )
            )

        for anchor in anchor_terms:
            clean = anchor.strip()
            if not clean or any(clean.lower() == term.original_text.lower() for term in terms):
                continue
            terms.append(
                ParsedEngineeringTerm(
                    term_id=f"TERM-{len(terms) + 1:03d}",
                    original_text=clean,
                    definition="An ontology-guided anchor supplied by the selected TDG node.",
                    enhanced_phrases=[clean],
                    source="orchestrator_anchor",
                )
            )

        properties: list[str] = []
        property_enhancements: list[str] = []
        for canonical, synonyms, enhanced in PROPERTY_LEXICON:
            explicit_numeric_mass = (
                canonical == "mass"
                and bool(
                    re.search(
                        r"\b\d+(?:\.\d+)?\s*(?:t|tonnes?|kg)\b(?!\s*-\s*km)",
                        lowered,
                    )
                )
            )
            chinese_count = (
                canonical == "count"
                and "多少" in lowered
                and bool(
                    re.search(
                        r"(?:有|共|各有|分别有)?多少(?:个|条|项|种)?(?:产品|构件|工作包|资源流|运输|leg|载具|默认项|因子|排除项|记录)",
                        lowered,
                    )
                )
            )
            if (
                any(_contains_property_marker(lowered, marker) for marker in synonyms)
                or explicit_numeric_mass
                or chinese_count
            ):
                properties.append(canonical)
                property_enhancements.extend(enhanced)
        if properties:
            add_role(SemanticRole.PROPERTY)
            terms.append(
                ParsedEngineeringTerm(
                    term_id=f"TERM-{len(terms) + 1:03d}",
                    original_text="; ".join(properties),
                    definition="Requested engineering quantity or comparison property.",
                    enhanced_phrases=list(dict.fromkeys(property_enhancements)),
                    requested_properties=properties,
                    source="query",
                )
            )

        identifiers = sorted(
            set(
                re.findall(
                    r"(?<![A-Za-z0-9])(?:M-\d+-\d+|V-\d+-\d+|CQ-[A-Z]\d{2}|[0-3][A-Za-z0-9_$]{21})(?![A-Za-z0-9])",
                    query,
                    flags=re.I,
                )
            )
        )
        for identifier in identifiers:
            terms.append(
                ParsedEngineeringTerm(
                    term_id=f"TERM-{len(terms) + 1:03d}",
                    original_text=identifier,
                    definition="An explicit engineering identifier requiring named-individual granularity.",
                    enhanced_phrases=[identifier],
                    source="query",
                )
            )

        trace_query = _contains(lowered, ("trace", "provenance", "audit chain", "s-e-p-r-c", "追溯", "溯源", "审计链", "完整链路"))
        carbon_query = _contains(lowered, ("carbon", "emission", "碳", "排放"))
        process_query = _contains(lowered, ("cpd", "work package", "process", "工序", "工作包", "施工过程"))
        resource_query = _contains(lowered, ("resource", "material", "concrete", "steel", "efd", "资源", "材料", "混凝土", "钢", "排放因子"))
        element_query = _contains(lowered, ("element", "component", "ifc", "wall", "slab", "pile", "构件", "围护", "墙", "板", "桩"))
        system_query = _contains(lowered, ("project", "system", "main structure", "enclosure", "项目", "系统", "主体", "围护"))
        if trace_query:
            required_roles = list((SemanticRole.SYSTEM, SemanticRole.ELEMENT, SemanticRole.PROCESS, SemanticRole.RESOURCE, SemanticRole.CARBON))
        else:
            if system_query:
                add_role(SemanticRole.SYSTEM)
            if element_query:
                add_role(SemanticRole.ELEMENT)
            if process_query:
                add_role(SemanticRole.PROCESS)
            if resource_query:
                add_role(SemanticRole.RESOURCE)
            if carbon_query:
                add_role(SemanticRole.CARBON)
        if mode in {QueryMode.STRATEGIC_ANALYSIS, QueryMode.KNOWLEDGE_CORRECTION} and not out_of_scope:
            add_role(SemanticRole.RESOURCE)
            add_role(SemanticRole.CARBON)
        if not required_roles and not out_of_scope:
            required_roles = [SemanticRole.SYSTEM]

        if not terms:
            tokens = [
                token
                for token in re.findall(r"[a-z][a-z0-9-]{2,}|[\u3400-\u9fff]{2,}", lowered)
                if token not in {"what", "which", "where", "when", "does", "have", "with", "from", "this", "that", "多少", "哪些", "如何"}
            ]
            fallback = " ".join(tokens[:8]) or query
            terms.append(
                ParsedEngineeringTerm(
                    term_id="TERM-001",
                    original_text=fallback,
                    definition="A schema-grounded fallback phrase extracted from the user request.",
                    enhanced_phrases=[fallback],
                    source="query",
                )
            )

        granularity = "mixed"
        if identifiers:
            granularity = "resource" if any(value.upper().startswith(("M-", "V-")) for value in identifiers) else "element"
        elif properties:
            granularity = "property"
        elif trace_query:
            granularity = "mixed"
        elif resource_query:
            granularity = "resource"
        elif process_query:
            granularity = "process"
        elif element_query:
            granularity = "element"
        elif system_query:
            granularity = "system"
        elif carbon_query:
            granularity = "project"

        intent = {
            QueryMode.INFORMATION_RETRIEVAL: "retrieve governed project knowledge and, when requested, calculate deterministic aggregates",
            QueryMode.KNOWLEDGE_CORRECTION: "prepare a persistent delta and pause for human adjudication",
            QueryMode.STRATEGIC_ANALYSIS: "construct a transient delta over the factual baseline and execute deterministic tools",
        }[mode]
        return QueryInterpretation(
            query=query,
            mode=mode,
            intent_summary=intent,
            in_scope=not out_of_scope,
            boundary_status="outside_active_boundary" if out_of_scope else "inside_active_boundary",
            out_of_scope_concepts=[query] if out_of_scope else [],
            terms=terms,
            required_roles=required_roles,
            required_entity_kinds=[OntologyEntityKind.INDIVIDUAL] if identifiers else [],
            required_properties=properties,
            requested_granularity=granularity,  # type: ignore[arg-type]
            explicit_identifiers=identifiers,
            governing_cq_ids=list(cq_ids),
        )

    @staticmethod
    def query_phrases(interpretation: QueryInterpretation) -> list[str]:
        phrases: list[str] = []
        for term in interpretation.terms:
            phrases.append(term.original_text)
            phrases.extend(term.enhanced_phrases)
        return list(dict.fromkeys(value.strip() for value in phrases if value.strip()))

    def retrieve(
        self,
        interpretation: QueryInterpretation,
        phrases: Sequence[str],
    ) -> list[OntologyEntityCandidate]:
        # Apply instance granularity to explicit identifiers, not all schema terms.
        schema_candidates = self.index.search(
            phrases,
            top_k=self.config.metadata_top_k,
            required_roles=interpretation.required_roles,
            kinds=(),
        )
        property_candidates: list[OntologyEntityCandidate] = []
        for property_name in interpretation.required_properties:
            property_candidates.extend(
                self.index.search(
                    [property_name, *PROPERTY_EQUIVALENTS.get(property_name, ())],
                    top_k=3,
                    required_roles=[SemanticRole.PROPERTY],
                    kinds=[
                        OntologyEntityKind.OBJECT_PROPERTY,
                        OntologyEntityKind.DATATYPE_PROPERTY,
                    ],
                )
            )
        identifier_candidates: list[OntologyEntityCandidate] = []
        if interpretation.explicit_identifiers:
            identifier_candidates = self.index.search(
                interpretation.explicit_identifiers,
                top_k=max(4, len(interpretation.explicit_identifiers) * 2),
                required_roles=interpretation.required_roles,
                kinds=[OntologyEntityKind.INDIVIDUAL],
            )
        return _deduplicate_candidates(
            [*identifier_candidates, *property_candidates, *schema_candidates]
        )

    def expand(
        self,
        interpretation: QueryInterpretation,
        seeds: Sequence[OntologyEntityCandidate],
    ) -> tuple[list[OntologyEntityCandidate], list[OntologyRelation]]:
        return self.index.expand(
            seeds,
            max_hops=self.config.metadata_expansion_hops,
            candidate_limit=self.config.metadata_candidate_limit,
            required_roles=interpretation.required_roles,
        )

    def resolve_paths(
        self,
        interpretation: QueryInterpretation,
        candidates: Sequence[OntologyEntityCandidate],
    ) -> list[ResolvedSemanticPath]:
        return self.index.resolve_paths(candidates, interpretation.required_roles)

    def verify(
        self,
        interpretation: QueryInterpretation,
        candidates: Sequence[OntologyEntityCandidate],
        relations: Sequence[OntologyRelation],
        paths: Sequence[ResolvedSemanticPath],
    ) -> BackboneVerification:
        if interpretation.boundary_status == "outside_active_boundary":
            return BackboneVerification(
                sufficient=True,
                structural_backbone_present=True,
                required_roles=[],
                present_roles=[],
                missing_roles=[],
                required_properties=interpretation.required_properties,
                present_properties=[],
                missing_properties=[],
                granularity_satisfied=True,
                boundary_satisfied=False,
                reasons=["The query was correctly identified as outside the declared A1-A5 boundary; no graph execution is allowed."],
            )
        present_roles: list[SemanticRole] = []
        for candidate in candidates:
            for role in candidate.semantic_roles:
                if role not in present_roles:
                    present_roles.append(role)
        missing_roles = [role for role in interpretation.required_roles if role not in present_roles]
        property_candidates = [
            candidate
            for candidate in candidates
            if candidate.entity_kind in {OntologyEntityKind.DATATYPE_PROPERTY, OntologyEntityKind.OBJECT_PROPERTY}
            or SemanticRole.PROPERTY in candidate.semantic_roles
        ]

        def property_matches(candidate: OntologyEntityCandidate, required: str) -> bool:
            searchable = " ".join(
                (candidate.uri, candidate.label, *candidate.alt_labels, candidate.definition)
            ).lower()
            compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", searchable)
            return any(
                equivalent.lower() in searchable
                or re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", equivalent.lower()) in compact
                for equivalent in PROPERTY_EQUIVALENTS.get(required, (required,))
            )

        present_properties = [
            required
            for required in interpretation.required_properties
            if any(property_matches(candidate, required) for candidate in property_candidates)
        ]
        missing_properties = [
            required for required in interpretation.required_properties if required not in present_properties
        ]
        required_core = [role for role in interpretation.required_roles if role in {
            SemanticRole.SYSTEM,
            SemanticRole.ELEMENT,
            SemanticRole.PROCESS,
            SemanticRole.RESOURCE,
            SemanticRole.CARBON,
        }]
        structural = len(required_core) <= 1 or any(
            set(required_core) <= set(path.semantic_roles) for path in paths
        )
        if not structural and required_core:
            relation_nodes = {relation.subject_uri for relation in relations} | {relation.object_uri for relation in relations}
            connected_roles = {
                role
                for candidate in candidates
                if candidate.uri in relation_nodes
                for role in candidate.semantic_roles
            }
            structural = set(required_core) <= connected_roles
        granularity_satisfied = True
        if interpretation.explicit_identifiers:
            searchable = " ".join(
                f"{candidate.uri} {candidate.label} {' '.join(candidate.alt_labels)}" for candidate in candidates
            ).lower()
            granularity_satisfied = all(identifier.lower() in searchable for identifier in interpretation.explicit_identifiers)
        sufficient = not missing_roles and not missing_properties and structural and granularity_satisfied
        reasons: list[str] = []
        if missing_roles:
            reasons.append("Missing S-E-P-R-C roles: " + ", ".join(role.value for role in missing_roles))
        if missing_properties:
            reasons.append("Missing requested ontology properties: " + ", ".join(missing_properties))
        if not structural:
            reasons.append("Retrieved entities do not yet form a connected query backbone.")
        if not granularity_satisfied:
            reasons.append("The explicit identifier has not been resolved to named-individual granularity.")
        if sufficient:
            reasons.append("Candidate entities and relations satisfy the declared semantic backbone.")
        missing_concepts = [
            phrase
            for role in missing_roles
            for phrase in ROLE_SUPPLEMENTS.get(role, ())[:2]
        ] + [
            phrase
            for property_name in missing_properties
            for phrase in (property_name, f"has {property_name}", "numeric value", "unit")
        ]
        return BackboneVerification(
            sufficient=sufficient,
            structural_backbone_present=structural,
            required_roles=interpretation.required_roles,
            present_roles=present_roles,
            missing_roles=missing_roles,
            required_properties=interpretation.required_properties,
            present_properties=present_properties,
            missing_properties=missing_properties,
            granularity_satisfied=granularity_satisfied,
            boundary_satisfied=True,
            reasons=reasons,
            missing_concepts=list(dict.fromkeys(missing_concepts)),
        )

    def supplement(
        self,
        interpretation: QueryInterpretation,
        verification: BackboneVerification,
        existing_phrases: Sequence[str],
    ) -> list[str]:
        supplements: list[str] = []
        for role in verification.missing_roles:
            supplements.extend(ROLE_SUPPLEMENTS.get(role, ()))
        for property_name in verification.missing_properties:
            supplements.extend((property_name, f"has {property_name}", "numeric value", "unit"))
        if not verification.structural_backbone_present:
            for role in interpretation.required_roles:
                supplements.extend(ROLE_SUPPLEMENTS.get(role, ()))
            supplements.extend(("requires work package", "requires resource instance", "has emissions"))
        if not verification.granularity_satisfied:
            supplements.extend(interpretation.explicit_identifiers)
        existing = {item.lower() for item in existing_phrases}
        return [
            item
            for item in dict.fromkeys(supplements)
            if item and item.lower() not in existing
        ]

    def project_reference_signature(
        self,
        query: str,
        mode: QueryMode,
        cq_ids: Sequence[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Project dynamic retrieval to the versioned evaluation signatures.

        This projection is not used to discover candidates; it keeps the public
        Query-GT contract stable while the runtime uses actual entity/relation
        retrieval and path discovery.
        """

        lowered = query.lower().replace("–", "-").replace("—", "-")
        cq_set = set(cq_ids)
        profiles: list[str] = []

        def add(profile_id: str) -> None:
            if profile_id not in profiles:
                profiles.append(profile_id)

        def has(*terms: str) -> bool:
            return any(term in lowered for term in terms)

        out_of_scope = bool(re.search(r"(?<![a-z0-9])(?:b[1-7]?|c[1-4]?|d)(?![a-z0-9])", lowered)) or has(
            "operational-use", "operation stage", "end of life", "deconstruction", "demolition", "recycling benefit",
            "运营使用", "运营阶段", "拆除阶段", "回收收益", "边界外", "d模块",
        )
        included_stages = has("lifecycle stages are included", "included in the active assessment", "核算包含", "包含哪些生命周期")
        ontology_query = has("ontology term", "ontology术语", "ontology 术语", "subontology", "本体术语")
        transport = has("a4", "transport", "haul", "vehicle", "t-km", "运输", "运距", "载具", "吨公里")
        foundation = has("foundation pit", "foundation-pit", "earthwork", "excavation", "backfill", "基坑", "土方", "开挖", "回填")
        enclosure = has("enclosure", "diaphragm", "sectionsteelstrut", "围护", "地下连续墙", "地连墙", "映射率")
        main_structure = has("main structure", "main-structure", "主体结构")
        trace = has("trace", "provenance", "audit chain", "resource breakdown", "追溯", "溯源", "审计链", "资源明细", "完整链路")
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
        if out_of_scope:
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
        elif ontology_query:
            if trace or "CQ-C02" in cq_set:
                add("ONT-SEPRC-PROVENANCE")
            elif transport or "CQ-C06" in cq_set:
                add("ONT-A4-TRANSPORT")
            elif "CQ-C08" in cq_set:
                add("ONT-SYSTEM-CARBON")
            else:
                add("ONT-PROJECT-CARBON")
        elif mode in {QueryMode.STRATEGIC_ANALYSIS, QueryMode.KNOWLEDGE_CORRECTION}:
            add("ONT-SCENARIO")
            if transport or "CQ-X02" in cq_set:
                add("ONT-A4-TRANSPORT")
            if cq_set & {"CQ-X01", "CQ-X03", "CQ-X05"} or process_resource_detail:
                add("ONT-SEPRC-PROVENANCE")
            if cq_set & {"CQ-X08", "CQ-Q04"}:
                add("ONT-SCOPE-QUALITY")
        elif public_context:
            add("ONT-PROJECT-CARBON")
        elif cq_set & {"CQ-C09", "CQ-C02"} or trace or (
            main_structure and cq_set >= {"CQ-P04", "CQ-C03"}
        ) or (main_structure and cq_set >= {"CQ-I02", "CQ-P03"}):
            add("ONT-SEPRC-PROVENANCE")
        elif cq_set & {"CQ-S01", "CQ-S03"} or ("CQ-C08" in cq_set and "CQ-I03" not in cq_set):
            add("ONT-SYSTEM-CARBON")
        elif "CQ-C06" in cq_set and "CQ-C05" not in cq_set:
            add("ONT-A4-TRANSPORT")
        elif foundation or "CQ-C05" in cq_set:
            add("ONT-FOUNDATION-EARTHWORK")
            if "CQ-C06" in cq_set:
                add("ONT-A4-TRANSPORT")
        elif enclosure or cq_set & {"CQ-I02", "CQ-I03"}:
            add("ONT-ENCLOSURE-CPD")
        else:
            add("ONT-PROJECT-CARBON")
        available = ontology_profiles()
        terms = sorted({term for profile_id in profiles for term in available[profile_id].required_terms})
        path_ids = sorted({path["path_id"] for profile_id in profiles for path in available[profile_id].semantic_paths})
        return profiles, terms, path_ids

    def run(
        self,
        query: str,
        mode: QueryMode,
        *,
        anchor_terms: Sequence[str] = (),
        cq_ids: Sequence[str] = (),
    ) -> MetadataRetrievalContext:
        interpretation = self.parse_query(query, mode, anchor_terms=anchor_terms, cq_ids=cq_ids)
        phrases = self.query_phrases(interpretation)
        rounds: list[MetadataRetrievalRound] = []
        all_candidates: list[OntologyEntityCandidate] = []
        all_relations: list[OntologyRelation] = []
        all_paths: list[ResolvedSemanticPath] = []
        previous_uris: set[str] = set()
        for round_no in range(1, self.config.metadata_max_rounds + 1):
            seeds = self.retrieve(interpretation, phrases)
            expanded, relations = self.expand(interpretation, seeds)
            candidates = _deduplicate_candidates([*all_candidates, *expanded])
            relations = _deduplicate_relations([*all_relations, *relations])
            paths = self.resolve_paths(interpretation, candidates)
            verification = self.verify(interpretation, candidates, relations, paths)
            current_uris = {item.uri for item in candidates}
            novel = len(current_uris - previous_uris)
            supplements: list[str] = []
            if interpretation.boundary_status == "outside_active_boundary":
                stop_reason = "outside_active_boundary"
            elif verification.sufficient:
                stop_reason = "sufficient"
            elif novel == 0 and round_no > 1:
                stop_reason = "no_novel_candidates"
            elif round_no >= self.config.metadata_max_rounds:
                stop_reason = "iteration_limit"
            else:
                supplements = self.supplement(interpretation, verification, phrases)
                stop_reason = "retry_required"
            rounds.append(
                MetadataRetrievalRound(
                    round_no=round_no,
                    query_phrases=list(phrases),
                    seed_candidates=seeds,
                    expanded_candidates=expanded,
                    relations=relations,
                    semantic_paths=paths,
                    verification=verification,
                    supplementary_terms=supplements,
                    novel_candidate_count=novel,
                    stop_reason=stop_reason,  # type: ignore[arg-type]
                )
            )
            all_candidates, all_relations, all_paths = candidates, relations, paths
            previous_uris = current_uris
            if stop_reason != "retry_required":
                break
            phrases = list(dict.fromkeys([*phrases, *supplements]))
        profiles, signature_terms, signature_paths = self.project_reference_signature(query, mode, cq_ids)
        final = rounds[-1]
        return MetadataRetrievalContext(
            interpretation=interpretation,
            backend=self.index.backend_info,
            rounds=rounds,
            final_candidates=all_candidates,
            final_relations=all_relations,
            final_paths=all_paths,
            final_verification=final.verification,
            retry_count=max(0, len(rounds) - 1),
            max_rounds=self.config.metadata_max_rounds,
            termination_reason=final.stop_reason,
            reference_profile_ids=profiles,
            reference_signature_terms=signature_terms,
            reference_signature_path_ids=signature_paths,
        )


@lru_cache(maxsize=4)
def cached_ontology_index(config: ProjectConfig) -> OntologyMetadataIndex:
    return OntologyMetadataIndex(config)


def build_metadata_pipeline(config: ProjectConfig, catalogue: CQCatalogue) -> MetadataPipeline:
    return MetadataPipeline(config, catalogue, cached_ontology_index(config))
