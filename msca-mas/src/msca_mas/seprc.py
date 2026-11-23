"""S-E-P-R-C role assignment derived from the TBox class hierarchy.

Each class inherits a role through the ``rdfs:subClassOf`` hierarchy:

    S  msca:EngineeringSystem
    E  msca:Element
    P  msca:ConstructionProcess
    R  msca:Resource and msca:ResourceDescriptor
    C  msca:CarbonEmission and msca:CarbonAssessment

Classes outside these trees use an explicit map or keyword matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace, RDF, RDFS, URIRef

from .config import ProjectConfig

MSCA = Namespace("http://bimlca.uni.edu/ontology#")

ROLE_ROOTS: dict[str, list[str]] = {
    "S": ["EngineeringSystem"],
    "E": ["Element"],
    "P": ["ConstructionProcess"],
    "R": ["Resource", "ResourceDescriptor"],
    "C": ["CarbonEmission", "CarbonAssessment"],
}

# Map classes outside the five domain hierarchies explicitly.
EXPLICIT_CLASS_ROLES: dict[str, list[str]] = {
    "Project": ["S"],
    "SiteContext": ["S"],
    "WorkPackageTemplate": ["P"],
    "CPD_Entry": ["P"],
    "EFD_Entry": ["R"],
    "EFD_Factor": ["R"],
    "ResourceSubstitution": ["R"],
    "AlternativeScenario": ["C"],
    "ScenarioResult": ["C"],
    "CarbonIntensityIndicator": ["C"],
    "CarbonPerformanceIndicator": ["C"],
    "CarbonAggregationActivity": ["C"],
    "CalculationReadinessAssessment": ["C"],
    "AssessmentScope": ["SCOPE"],
    "QuantityInterval": ["SCOPE"],
    "SemanticAssertion": ["SCOPE"],
    "ClassificationAssertion": ["SCOPE"],
    "FactorCandidateEvaluation": ["R"],
    "FactorSelectionDecision": ["R"],
    "FactorOverride": ["R"],
    "ParameterOverride": ["R"],
    "WastageOverride": ["R"],
    "ReuseCycleOverride": ["R"],
    "TransportLeg": ["R"],
    "CompetencyQuestion": ["SCOPE"],
    "ElementType": ["E"],
    "EarthworkZone": ["E"],
}

_ROLE_MAP_RESOURCE = Path(__file__).parent / "resources" / "seprc_role_map.json"


def derive_role_map(schema_paths: list[Path]) -> dict[str, Any]:
    graph = Graph()
    for path in schema_paths:
        graph.parse(path, format="turtle")

    parents: dict[URIRef, set[URIRef]] = {}
    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            parents.setdefault(child, set()).add(parent)

    def ancestors(node: URIRef, seen: set[URIRef] | None = None) -> set[URIRef]:
        seen = seen or set()
        result: set[URIRef] = set()
        for parent in parents.get(node, ()):  # type: ignore[arg-type]
            if parent in seen:
                continue
            seen.add(parent)
            result.add(parent)
            result |= ancestors(parent, seen)
        return result

    classes = sorted(set(graph.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))), key=str)
    role_map: dict[str, dict[str, Any]] = {}
    for cls in classes:
        closure = {cls} | ancestors(cls)  # type: ignore[arg-type]
        local_names = {str(node).rsplit("#", 1)[-1].rsplit("/", 1)[-1] for node in closure}
        roles: list[str] = []
        source = None
        for role, root_names in ROLE_ROOTS.items():
            if any(name in local_names for name in root_names):
                roles.append(role)
                source = "tbox_root"
        if not roles:
            explicit = EXPLICIT_CLASS_ROLES.get(str(cls).rsplit("#", 1)[-1].rsplit("/", 1)[-1], [])
            if explicit:
                roles = list(explicit)
                source = "explicit_map"
        if roles:
            role_map[str(cls)] = {"roles": roles, "source": source}

    return {
        "map_version": 1,
        "role_roots": {role: [f"{MSCA}{name}" for name in names] for role, names in ROLE_ROOTS.items()},
        "explicit_class_roles": {f"{MSCA}{name}": roles for name, roles in EXPLICIT_CLASS_ROLES.items()},
        "classes_total": len(classes),
        "classes_mapped": len(role_map),
        "entries": role_map,
    }


def write_role_map_resource(config: ProjectConfig, output: Path | None = None) -> Path:
    payload = derive_role_map([Path(config.tbox)])
    output = output or _ROLE_MAP_RESOURCE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return output


class RoleAssigner:
    """Look up a role in the schema map, then fall back to keyword matching."""

    def __init__(self, role_map: dict[str, Any] | None = None):
        if role_map is None:
            role_map = json.loads(_ROLE_MAP_RESOURCE.read_text(encoding="utf-8"))
        self.entries: dict[str, dict[str, Any]] = role_map["entries"]
        self.stats = {"map_hits": 0, "keyword_fallbacks": 0}

    def lookup(self, class_uri: str) -> dict[str, Any] | None:
        return self.entries.get(class_uri)

    def roles_for(self, uri: str, types: list[str] | tuple[str, ...]) -> list[str] | None:
        """Schema-derived roles for an entity, or None when unmapped.

        A class entity maps directly; an individual inherits the union of its
        asserted types' roles.
        """

        roles: list[str] = []
        entry = self.lookup(uri)
        if entry is not None:
            roles.extend(entry["roles"])
            self.stats["map_hits"] += 1
        for type_uri in types:
            entry = self.lookup(str(type_uri))
            if entry is not None:
                for role in entry["roles"]:
                    if role not in roles:
                        roles.append(role)
        if roles:
            self.stats["map_hits"] += 1
            return roles
        self.stats["keyword_fallbacks"] += 1
        return None
