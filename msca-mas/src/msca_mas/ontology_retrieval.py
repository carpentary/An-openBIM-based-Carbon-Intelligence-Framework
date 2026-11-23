from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from rdflib import BNode, Graph, Literal, OWL, RDF, RDFS, SKOS, URIRef

from .config import ProjectConfig
from .semantic_models import (
    OntologyEntityCandidate,
    OntologyEntityKind,
    OntologyRelation,
    ResolvedSemanticPath,
    RetrievalBackendInfo,
    SemanticRole,
)
from .util import stable_digest


_ROLE_ASSIGNER_SINGLETON: dict[str, Any] = {}


class _KeywordOnlyRoleAssigner:
    """Use keyword matching for semantic roles."""

    stats = {"map_hits": 0, "keyword_fallbacks": 0}

    def roles_for(self, uri: str, types: list[str] | tuple[str, ...]) -> list[str] | None:
        return None  # forces the keyword heuristic everywhere


def _role_assigner() -> Any:
    """Load schema-derived roles on demand."""

    import os

    if os.environ.get("MSCA_SEPRC_ROLES", "").lower() in {"off", "0", "false"}:
        return _KeywordOnlyRoleAssigner()
    if "assigner" not in _ROLE_ASSIGNER_SINGLETON:
        from .seprc import RoleAssigner

        _ROLE_ASSIGNER_SINGLETON["assigner"] = RoleAssigner()
    return _ROLE_ASSIGNER_SINGLETON["assigner"]


SCHEMA_EDGE_PREDICATES = {
    str(RDF.type),
    str(RDFS.subClassOf),
    str(RDFS.domain),
    str(RDFS.range),
    str(OWL.equivalentClass),
    str(OWL.equivalentProperty),
    str(OWL.inverseOf),
    str(SKOS.broader),
    str(SKOS.narrower),
    str(SKOS.related),
}

ROLE_ORDER = (
    SemanticRole.SYSTEM,
    SemanticRole.ELEMENT,
    SemanticRole.PROCESS,
    SemanticRole.RESOURCE,
    SemanticRole.CARBON,
)


def _local_name(uri: str) -> str:
    value = uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return value.strip()


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    output = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", lowered)
    for run in re.findall(r"[\u3400-\u9fff]+", lowered):
        output.extend(run)
        output.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return output


def _char_features(text: str) -> list[str]:
    compact = "".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", text.lower()))
    features: list[str] = []
    for size in (2, 3, 4):
        features.extend(compact[index : index + size] for index in range(max(0, len(compact) - size + 1)))
    return features


def _compact_text(text: str) -> str:
    """Normalise labels for exact ontology-name matching.

    Ontology local names frequently use CamelCase while labels use spaces.  A
    compact comparison prevents ``EngineeringSystem`` and
    ``Engineering System`` from being treated as unrelated strings.
    """

    return "".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", text.lower()))


def _schema_kind_priority(kind: OntologyEntityKind) -> int:
    """Prefer schema concepts over generic catalogue individuals.

    Named individuals remain available (and are mandatory for identifier
    queries), but a broad semantic query should first establish its class and
    property backbone before expanding to ABox instances.
    """

    return {
        OntologyEntityKind.CLASS: 0,
        OntologyEntityKind.OBJECT_PROPERTY: 1,
        OntologyEntityKind.DATATYPE_PROPERTY: 1,
        OntologyEntityKind.CONCEPT: 2,
        OntologyEntityKind.INDIVIDUAL: 3,
        OntologyEntityKind.ANNOTATION_PROPERTY: 4,
        OntologyEntityKind.OTHER: 5,
    }[kind]


def _normalise_score(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class SemanticBackend(Protocol):
    @property
    def info(self) -> RetrievalBackendInfo: ...

    def similarities(self, query: str, documents: Sequence[str]) -> list[float]: ...

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


class DeterministicSemanticBackend:
    """Offline dense-space and pairwise reranking backend for reproducible tests.

    This backend preserves the paper's lexical/dense/rerank stages without
    claiming to be BGE. Every trace identifies it explicitly. Production can
    switch to the paper-specified local BGE models without changing the MAS.
    """

    def __init__(self, config: ProjectConfig, fallback_used: bool = False, dimensions: int = 768):
        self.config = config
        self.dimensions = dimensions
        self._info = RetrievalBackendInfo(
            lexical_backend="BM25",
            dense_backend="deterministic_feature_hashing",
            reranker_backend="deterministic_pairwise_schema_reranker",
            dense_model_id=config.metadata_dense_model_id,
            reranker_model_id=config.metadata_reranker_model_id,
            model_revision="individually_pinned",
            dense_model_revision=config.metadata_dense_model_revision,
            reranker_model_revision=config.metadata_reranker_model_revision,
            offline=True,
            fallback_used=fallback_used,
        )

    @property
    def info(self) -> RetrievalBackendInfo:
        return self._info

    @lru_cache(maxsize=100_000)
    def _vector(self, text: str) -> dict[int, float]:
        vector: dict[int, float] = defaultdict(float)
        features = [f"w:{token}" for token in _tokens(text)] + [f"c:{item}" for item in _char_features(text)]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            return {key: value / norm for key, value in vector.items()}
        return {}

    @staticmethod
    def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        value = sum(weight * right.get(index, 0.0) for index, weight in left.items())
        return max(0.0, min(1.0, (value + 1.0) / 2.0))

    def similarities(self, query: str, documents: Sequence[str]) -> list[float]:
        query_vector = self._vector(query)
        return [self._cosine(query_vector, self._vector(document)) for document in documents]

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        query_lower = query.lower().strip()
        query_tokens = set(_tokens(query))
        scores: list[float] = []
        for document in documents:
            lower = document.lower()
            document_tokens = set(_tokens(document))
            overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            exact = 1.0 if query_lower and query_lower in lower else 0.0
            phrase_parts = [part.strip() for part in re.split(r"[;,|]", query_lower) if part.strip()]
            phrase_hit = max((1.0 if part in lower else 0.0 for part in phrase_parts), default=0.0)
            scores.append(min(1.0, 0.55 * overlap + 0.30 * exact + 0.15 * phrase_hit))
        return scores


class BGESemanticBackend:
    """Local BGE-M3 retrieval and BGE-reranker-v2-m3 reranking."""

    def __init__(self, config: ProjectConfig):
        from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore

        self.encoder = SentenceTransformer(
            config.metadata_dense_model_id,
            revision=config.metadata_dense_model_revision,
            local_files_only=config.metadata_offline,
        )
        self.reranker = CrossEncoder(
            config.metadata_reranker_model_id,
            revision=config.metadata_reranker_model_revision,
            local_files_only=config.metadata_offline,
        )
        # Cache corpus embeddings across queries.
        self._document_embedding_cache: OrderedDict[str, object] = OrderedDict()
        self._document_embedding_cache_limit = 4
        self._info = RetrievalBackendInfo(
            lexical_backend="BM25",
            dense_backend="BGE-M3",
            reranker_backend="BGE-reranker-v2-m3",
            dense_model_id=config.metadata_dense_model_id,
            reranker_model_id=config.metadata_reranker_model_id,
            model_revision="individually_pinned",
            dense_model_revision=config.metadata_dense_model_revision,
            reranker_model_revision=config.metadata_reranker_model_revision,
            offline=config.metadata_offline,
            fallback_used=False,
        )

    @property
    def info(self) -> RetrievalBackendInfo:
        return self._info

    def similarities(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        cache_key = stable_digest(list(documents))
        document_embeddings = self._document_embedding_cache.get(cache_key)
        if document_embeddings is None:
            document_embeddings = self.encoder.encode(
                list(documents),
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self._document_embedding_cache[cache_key] = document_embeddings
            self._document_embedding_cache.move_to_end(cache_key)
            while len(self._document_embedding_cache) > self._document_embedding_cache_limit:
                self._document_embedding_cache.popitem(last=False)
        else:
            self._document_embedding_cache.move_to_end(cache_key)
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        values = [float(query_embedding @ embedding) for embedding in document_embeddings]
        return [max(0.0, min(1.0, (value + 1.0) / 2.0)) for value in values]

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        values = self.reranker.predict(
            [(query, document) for document in documents],
            batch_size=32,
            show_progress_bar=False,
        )
        raw = [float(value) for value in values]
        return _normalise_score(raw)


def build_semantic_backend(config: ProjectConfig) -> SemanticBackend:
    backend = config.metadata_backend.lower()
    if backend not in {"auto", "bge", "deterministic"}:
        raise ValueError(f"unknown metadata backend: {config.metadata_backend}")
    if backend in {"auto", "bge"}:
        try:
            return BGESemanticBackend(config)
        except Exception:
            if backend == "bge":
                raise
    return DeterministicSemanticBackend(config, fallback_used=backend == "auto")


@dataclass
class OntologyEntityRecord:
    uri: str
    labels: list[str]
    alt_labels: list[str]
    identifiers: list[str]
    definition: str
    kind: OntologyEntityKind
    roles: list[SemanticRole]
    source_graphs: list[str]
    types: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.labels[0] if self.labels else _local_name(self.uri)

    @property
    def searchable_text(self) -> str:
        return " | ".join(
            part
            for part in (
                self.label,
                *self.labels[1:],
                *self.alt_labels,
                *self.identifiers,
                self.definition,
                _local_name(self.uri),
                *(_local_name(item) for item in self.types),
                *(role.value for role in self.roles),
            )
            if part
        )


@dataclass(frozen=True)
class GraphEdge:
    subject: str
    predicate: str
    object: str
    source: str
    inferred_from_schema: bool = False


class OntologyMetadataIndex:
    """Unified class/property/individual index and relation graph for A_meta."""

    def __init__(self, config: ProjectConfig, backend: SemanticBackend | None = None):
        self.config = config
        self.backend = backend or build_semantic_backend(config)
        self.records: dict[str, OntologyEntityRecord] = {}
        self.edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[tuple[str, GraphEdge, str]]] = defaultdict(list)
        self._documents: list[str] = []
        self._uris: list[str] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_document_length = 0.0
        self._source_digests: dict[str, str] = {}
        self._load()

    @property
    def backend_info(self) -> RetrievalBackendInfo:
        return self.backend.info

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "entity_count": len(self.records),
            "relation_count": len(self.edges),
            "source_digests": self._source_digests,
            "backend": self.backend.info.model_dump(mode="json"),
            "index_digest": stable_digest(
                {
                    "entities": sorted(self.records),
                    "edges": sorted((edge.subject, edge.predicate, edge.object) for edge in self.edges),
                    "sources": self._source_digests,
                }
            ),
        }

    def _source_paths(self) -> list[Path]:
        paths = [
            self.config.complete_schema,
            self.config.catalogue,
            self.config.cpd_reference,
            self.config.efd_reference,
        ]
        reference_abox = self.config.artifact_dir / "reference_case" / "reference_case_abox.ttl"
        if reference_abox.exists():
            paths.append(reference_abox)
        return [path for path in paths if path.exists()]

    def _load(self) -> None:
        combined = Graph()
        subject_sources: dict[str, set[str]] = defaultdict(set)
        triple_sources: dict[tuple[str, str, str], str] = {}
        for path in self._source_paths():
            graph = Graph().parse(path)
            source_id = path.stem
            self._source_digests[source_id] = hashlib.sha256(path.read_bytes()).hexdigest()
            for subject, predicate, obj in graph:
                combined.add((subject, predicate, obj))
                if isinstance(subject, URIRef):
                    subject_sources[str(subject)].add(source_id)
                if isinstance(subject, URIRef) and isinstance(predicate, URIRef) and isinstance(obj, URIRef):
                    triple_sources[(str(subject), str(predicate), str(obj))] = source_id

        candidate_uris: set[str] = set()
        for subject in combined.subjects():
            if isinstance(subject, URIRef):
                candidate_uris.add(str(subject))
        for obj in combined.objects():
            if isinstance(obj, URIRef) and any(combined.triples((obj, None, None))):
                candidate_uris.add(str(obj))

        for uri in sorted(candidate_uris):
            subject = URIRef(uri)
            labels = self._literal_values(combined, subject, (RDFS.label, SKOS.prefLabel))
            alt_labels = self._literal_values(combined, subject, (SKOS.altLabel,))
            domain_names: list[str] = []
            identifiers: list[str] = []
            for predicate, value in combined.predicate_objects(subject):
                if not isinstance(predicate, URIRef) or not isinstance(value, Literal):
                    continue
                predicate_name = _compact_text(_local_name(str(predicate)))
                text = str(value).strip()
                if not text:
                    continue
                if predicate_name == "identifier" or predicate_name.endswith("code") or predicate_name == "hasglobalid":
                    identifiers.append(text)
                elif predicate_name.endswith("name"):
                    domain_names.append(text)
            labels = list(
                dict.fromkeys(
                    [
                        *labels,
                        *sorted(domain_names, key=lambda value: (value.casefold(), value)),
                    ]
                )
            )
            identifiers = sorted(set(identifiers), key=lambda value: (value.casefold(), value))
            definitions = self._literal_values(combined, subject, (SKOS.definition, RDFS.comment))
            types = sorted(str(value) for value in combined.objects(subject, RDF.type) if isinstance(value, URIRef))
            kind = self._kind(types)
            if not labels:
                labels = [_local_name(uri)]
            roles = self._roles(uri, labels, alt_labels, definitions, types, kind)
            self.records[uri] = OntologyEntityRecord(
                uri=uri,
                labels=labels,
                alt_labels=alt_labels,
                identifiers=identifiers,
                definition=definitions[0] if definitions else "",
                kind=kind,
                roles=roles,
                source_graphs=sorted(subject_sources.get(uri, set())),
                types=types,
            )

        seen_edges: set[tuple[str, str, str, bool]] = set()
        for subject, predicate, obj in combined:
            if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef) or not isinstance(obj, URIRef):
                continue
            subject_uri, predicate_uri, object_uri = str(subject), str(predicate), str(obj)
            predicate_types = {str(value) for value in combined.objects(predicate, RDF.type)}
            is_object_property = str(OWL.ObjectProperty) in predicate_types
            if predicate_uri not in SCHEMA_EDGE_PREDICATES and not is_object_property:
                continue
            if subject_uri not in self.records or object_uri not in self.records:
                continue
            key = (subject_uri, predicate_uri, object_uri, False)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            self.edges.append(
                GraphEdge(
                    subject=subject_uri,
                    predicate=predicate_uri,
                    object=object_uri,
                    source=triple_sources.get((subject_uri, predicate_uri, object_uri), "ontology_graph"),
                )
            )

        for predicate in combined.subjects(RDF.type, OWL.ObjectProperty):
            if not isinstance(predicate, URIRef):
                continue
            domains = [value for value in combined.objects(predicate, RDFS.domain) if isinstance(value, URIRef)]
            ranges = [value for value in combined.objects(predicate, RDFS.range) if isinstance(value, URIRef)]
            for domain in domains:
                for range_value in ranges:
                    key = (str(domain), str(predicate), str(range_value), True)
                    if key in seen_edges or str(domain) not in self.records or str(range_value) not in self.records:
                        continue
                    seen_edges.add(key)
                    self.edges.append(
                        GraphEdge(
                            subject=str(domain),
                            predicate=str(predicate),
                            object=str(range_value),
                            source="schema_domain_range",
                            inferred_from_schema=True,
                        )
                    )

        self.edges.sort(key=lambda item: (item.subject, item.predicate, item.object, item.source))
        for edge in self.edges:
            self._adjacency[edge.subject].append((edge.object, edge, "forward"))
            self._adjacency[edge.object].append((edge.subject, edge, "inverse"))

        self._uris = sorted(self.records)
        self._documents = [self.records[uri].searchable_text for uri in self._uris]
        self._term_frequencies = [Counter(_tokens(document)) for document in self._documents]
        for frequencies in self._term_frequencies:
            self._document_frequency.update(frequencies.keys())
        self._average_document_length = (
            sum(sum(frequencies.values()) for frequencies in self._term_frequencies) / len(self._term_frequencies)
            if self._term_frequencies
            else 0.0
        )

    @staticmethod
    def _literal_values(graph: Graph, subject: URIRef, predicates: Iterable[URIRef]) -> list[str]:
        values: list[str] = []
        for predicate in predicates:
            for value in graph.objects(subject, predicate):
                text = str(value).strip()
                if text and text not in values:
                    values.append(text)
        # Sort RDF labels before creating retrieval text and digests.
        return sorted(values, key=lambda value: (value.casefold(), value))

    @staticmethod
    def _kind(types: Sequence[str]) -> OntologyEntityKind:
        type_set = set(types)
        if str(OWL.Class) in type_set:
            return OntologyEntityKind.CLASS
        if str(OWL.ObjectProperty) in type_set:
            return OntologyEntityKind.OBJECT_PROPERTY
        if str(OWL.DatatypeProperty) in type_set:
            return OntologyEntityKind.DATATYPE_PROPERTY
        if str(OWL.AnnotationProperty) in type_set:
            return OntologyEntityKind.ANNOTATION_PROPERTY
        if str(SKOS.Concept) in type_set:
            return OntologyEntityKind.CONCEPT
        if str(OWL.NamedIndividual) in type_set or types:
            return OntologyEntityKind.INDIVIDUAL
        return OntologyEntityKind.OTHER

    @staticmethod
    def _roles(
        uri: str,
        labels: Sequence[str],
        alt_labels: Sequence[str],
        definitions: Sequence[str],
        types: Sequence[str],
        kind: OntologyEntityKind,
    ) -> list[SemanticRole]:
        # Use class-derived roles first; use keywords only as a fallback.
        if kind not in {
            OntologyEntityKind.OBJECT_PROPERTY,
            OntologyEntityKind.DATATYPE_PROPERTY,
            OntologyEntityKind.ANNOTATION_PROPERTY,
        }:
            mapped = _role_assigner().roles_for(uri, types)
            if mapped is not None:
                return [SemanticRole(role) for role in mapped]
        text = " ".join((_local_name(uri), *labels, *alt_labels, *definitions, *(_local_name(item) for item in types))).lower()
        roles: list[SemanticRole] = []

        def add(role: SemanticRole) -> None:
            if role not in roles:
                roles.append(role)

        if kind in {
            OntologyEntityKind.OBJECT_PROPERTY,
            OntologyEntityKind.DATATYPE_PROPERTY,
            OntologyEntityKind.ANNOTATION_PROPERTY,
        }:
            add(SemanticRole.PROPERTY)
        if any(term in text for term in ("engineering system", "subsystem", "construction project", "metro station", "project")):
            add(SemanticRole.SYSTEM)
        if any(
            term in text
            for term in (
                "element",
                "wall",
                "slab",
                "beam",
                "column",
                "pile",
                "bracing",
                "strut",
                "spatial work",
                "excavation zone",
                "foundation pit",
            )
        ):
            add(SemanticRole.ELEMENT)
        if any(
            term in text
            for term in (
                "work package",
                "process",
                "construction engineering",
                "excavation engineering",
                "transportation engineering",
                "backfill",
                "formwork engineering",
                "cpd entry",
                "quota",
            )
        ):
            add(SemanticRole.PROCESS)
        if any(
            term in text
            for term in (
                "resource",
                "material",
                "equipment",
                "energy",
                "concrete",
                "steel",
                "formwork",
                "emission factor",
                "efd factor",
                "vehicle",
                "fuel",
                "electricity",
            )
        ):
            add(SemanticRole.RESOURCE)
        if any(
            term in text
            for term in (
                "carbon",
                "emission",
                "assessment",
                "life cycle stage",
                "product stage",
                "transportation stage",
                "construction stage",
                "scenario result",
                "kgco2e",
            )
        ):
            add(SemanticRole.CARBON)
        if any(term in text for term in ("assessment scope", "coverage gap", "boundary", "included stage", "excluded")):
            add(SemanticRole.SCOPE)
        return roles or [SemanticRole.OTHER]

    def _bm25(self, query_tokens: Sequence[str], document_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        frequencies = self._term_frequencies[document_index]
        document_length = sum(frequencies.values())
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            inverse_document_frequency = math.log(
                1 + (len(self._documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(self._average_document_length, 1.0)
            )
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        return score

    def search(
        self,
        phrases: Sequence[str],
        *,
        top_k: int,
        required_roles: Sequence[SemanticRole] = (),
        kinds: Sequence[OntologyEntityKind] = (),
    ) -> list[OntologyEntityCandidate]:
        clean_phrases = list(dict.fromkeys(phrase.strip() for phrase in phrases if phrase.strip()))
        if not clean_phrases:
            return []
        allowed_kinds = set(kinds)
        role_set = set(required_roles)
        aggregate: dict[str, dict[str, object]] = {}
        for phrase in clean_phrases:
            lexical_raw = [self._bm25(_tokens(phrase), index) for index in range(len(self._documents))]
            lexical = _normalise_score(lexical_raw)
            dense = self.backend.similarities(phrase, self._documents)
            code_sensitive = bool(re.search(r"(?:\b[A-Z]?\d+[A-Z0-9.-]*\b|\b[A-Z]-\d+-\d+\b)", phrase, re.I))
            lexical_weight, dense_weight = ((0.65, 0.35) if code_sensitive else (0.45, 0.55))
            phrase_compact = _compact_text(phrase)
            for index, uri in enumerate(self._uris):
                record = self.records[uri]
                if allowed_kinds and record.kind not in allowed_kinds:
                    continue
                label_text = " | ".join((*record.labels, *record.alt_labels)).lower()
                label_forms = {
                    _compact_text(value)
                    for value in (
                        *record.labels,
                        *record.alt_labels,
                        *record.identifiers,
                        _local_name(uri),
                    )
                    if value
                }
                exact = 1.0 if (
                    phrase.lower() in label_text
                    or (phrase_compact and phrase_compact in label_forms)
                ) else 0.0
                role_bonus = 0.04 if role_set & set(record.roles) else 0.0
                kind_bonus = 0.0
                if not allowed_kinds:
                    if record.kind == OntologyEntityKind.CLASS:
                        kind_bonus = 0.05
                    elif record.kind in {OntologyEntityKind.OBJECT_PROPERTY, OntologyEntityKind.DATATYPE_PROPERTY}:
                        kind_bonus = 0.03
                    elif record.kind == OntologyEntityKind.INDIVIDUAL and not exact:
                        kind_bonus = -0.03
                fused = max(
                    0.0,
                    min(
                        1.0,
                        lexical_weight * lexical[index]
                        + dense_weight * dense[index]
                        + 0.22 * exact
                        + role_bonus
                        + kind_bonus,
                    ),
                )
                current = aggregate.get(uri)
                if current is None or fused > float(current["fused"]):
                    aggregate[uri] = {
                        "lexical": lexical[index],
                        "dense": dense[index],
                        "fused": fused,
                        "phrases": [phrase],
                        "exact": exact,
                    }
                elif math.isclose(fused, float(current["fused"]), abs_tol=1e-12):
                    current["phrases"] = list(dict.fromkeys([*current["phrases"], phrase]))  # type: ignore[index]
                    current["exact"] = max(float(current.get("exact", 0.0)), exact)

        ranked = sorted(
            aggregate.items(),
            key=lambda item: (
                -float(item[1].get("exact", 0.0)),
                -float(item[1]["fused"]),
                _schema_kind_priority(self.records[item[0]].kind),
                item[0],
            ),
        )
        # Retain candidates for each required role before filling the shortlist globally.
        coarse: list[tuple[str, dict[str, object]]] = []
        coarse_uris: set[str] = set()

        def add_coarse(item: tuple[str, dict[str, object]]) -> None:
            if item[0] not in coarse_uris:
                coarse.append(item)
                coarse_uris.add(item[0])

        for item in ranked:
            if float(item[1].get("exact", 0.0)):
                add_coarse(item)
        per_role_quota = max(4, top_k)
        for role in required_roles:
            role_ranked = sorted(
                (item for item in ranked if role in self.records[item[0]].roles),
                key=lambda item: (
                    -float(item[1].get("exact", 0.0)),
                    _schema_kind_priority(self.records[item[0]].kind),
                    -float(item[1]["fused"]),
                    item[0],
                ),
            )
            for item in role_ranked[:per_role_quota]:
                add_coarse(item)
        coarse_limit = max(top_k * 8, 60)
        for item in ranked:
            if len(coarse) >= coarse_limit:
                break
            add_coarse(item)
        coarse = coarse[:coarse_limit]
        rerank_query = " ; ".join(clean_phrases)
        rerank_documents = [self.records[uri].searchable_text for uri, _ in coarse]
        rerank_scores = self.backend.rerank(rerank_query, rerank_documents)
        candidates: list[OntologyEntityCandidate] = []
        for (uri, score), rerank_score in zip(coarse, rerank_scores, strict=True):
            record = self.records[uri]
            final = min(1.0, 0.64 * float(score["fused"]) + 0.36 * rerank_score)
            candidates.append(
                OntologyEntityCandidate(
                    uri=uri,
                    label=record.label,
                    alt_labels=list(dict.fromkeys([*record.alt_labels, *record.identifiers])),
                    definition=record.definition,
                    entity_kind=record.kind,
                    semantic_roles=record.roles,
                    source_graphs=record.source_graphs,
                    matched_phrases=list(score["phrases"]),
                    lexical_score=float(score["lexical"]),
                    dense_score=float(score["dense"]),
                    fused_score=float(score["fused"]),
                    rerank_score=rerank_score,
                    final_score=final,
                    seed=True,
                    evidence=[
                        f"BM25={float(score['lexical']):.6f}",
                        f"dense={float(score['dense']):.6f}",
                        f"rerank={rerank_score:.6f}",
                        f"exact_label={bool(score.get('exact', 0.0))}",
                    ],
                )
            )
        candidates.sort(key=lambda item: (-item.final_score, item.uri))
        selected: list[OntologyEntityCandidate] = []
        selected_uris: set[str] = set()

        def add_candidate(candidate: OntologyEntityCandidate) -> None:
            if candidate.uri not in selected_uris and len(selected) < top_k:
                selected.append(candidate)
                selected_uris.add(candidate.uri)

        # Seed expansion with one candidate per requested role.
        for role in required_roles:
            role_candidates = sorted(
                (candidate for candidate in candidates if role in candidate.semantic_roles),
                key=lambda candidate: (
                    0 if "exact_label=True" in candidate.evidence else 1,
                    _schema_kind_priority(candidate.entity_kind),
                    -candidate.final_score,
                    candidate.uri,
                ),
            )
            if role_candidates:
                add_candidate(role_candidates[0])
        # Keep exact name or identifier matches after reranking.
        for candidate in candidates:
            if "exact_label=True" in candidate.evidence:
                add_candidate(candidate)
        for candidate in candidates:
            add_candidate(candidate)
        return sorted(selected, key=lambda item: (-item.final_score, item.uri))

    def expand(
        self,
        seeds: Sequence[OntologyEntityCandidate],
        *,
        max_hops: int,
        candidate_limit: int,
        required_roles: Sequence[SemanticRole] = (),
    ) -> tuple[list[OntologyEntityCandidate], list[OntologyRelation]]:
        if not seeds:
            return [], []
        required = set(required_roles)
        candidate_map = {item.uri: item for item in seeds}
        frontier: deque[tuple[str, int, float]] = deque(
            (item.uri, 0, item.final_score) for item in sorted(seeds, key=lambda value: (-value.final_score, value.uri))
        )
        visited_hop = {item.uri: 0 for item in seeds}
        relation_map: dict[str, OntologyRelation] = {}
        while frontier and len(candidate_map) < candidate_limit:
            current_uri, hop, seed_score = frontier.popleft()
            if hop >= max_hops:
                continue
            neighbours = sorted(
                self._adjacency.get(current_uri, []),
                key=lambda item: (
                    0 if required & set(self.records[item[0]].roles) else 1,
                    item[1].predicate,
                    item[0],
                ),
            )
            for neighbour_uri, edge, direction in neighbours:
                if neighbour_uri not in self.records:
                    continue
                next_hop = hop + 1
                if visited_hop.get(neighbour_uri, next_hop + 1) <= next_hop:
                    continue
                visited_hop[neighbour_uri] = next_hop
                neighbour = self.records[neighbour_uri]
                relation = self._relation(edge, next_hop, direction)
                relation_map[relation.relation_id + f"|{direction}"] = relation
                role_bonus = 0.10 if required & set(neighbour.roles) else 0.0
                path_score = min(1.0, seed_score * (0.78**next_hop) + role_bonus)
                if neighbour_uri not in candidate_map:
                    candidate_map[neighbour_uri] = OntologyEntityCandidate(
                        uri=neighbour_uri,
                        label=neighbour.label,
                        alt_labels=neighbour.alt_labels,
                        definition=neighbour.definition,
                        entity_kind=neighbour.kind,
                        semantic_roles=neighbour.roles,
                        source_graphs=neighbour.source_graphs,
                        matched_phrases=[],
                        fused_score=path_score,
                        rerank_score=path_score,
                        final_score=path_score,
                        seed=False,
                        discovered_at_hop=next_hop,
                        evidence=[
                            f"graph expansion from {current_uri}",
                            f"via {edge.predicate}",
                            f"direction={direction}",
                        ],
                    )
                frontier.append((neighbour_uri, next_hop, seed_score))
                if len(candidate_map) >= candidate_limit:
                    break
        expanded = sorted(candidate_map.values(), key=lambda item: (not item.seed, item.discovered_at_hop, -item.final_score, item.uri))
        relations = sorted(
            relation_map.values(),
            key=lambda item: (item.hop, item.subject_uri, item.predicate_uri, item.object_uri, item.direction),
        )
        return expanded, relations

    def _relation(self, edge: GraphEdge, hop: int, direction: str) -> OntologyRelation:
        subject = self.records.get(edge.subject)
        predicate = self.records.get(edge.predicate)
        obj = self.records.get(edge.object)
        return OntologyRelation(
            subject_uri=edge.subject,
            predicate_uri=edge.predicate,
            object_uri=edge.object,
            subject_label=subject.label if subject else _local_name(edge.subject),
            predicate_label=predicate.label if predicate else _local_name(edge.predicate),
            object_label=obj.label if obj else _local_name(edge.object),
            source=edge.source,
            hop=hop,
            inferred_from_schema=edge.inferred_from_schema,
            direction=direction,  # type: ignore[arg-type]
            evidence=(
                "OWL object-property domain/range connection"
                if edge.inferred_from_schema
                else "explicit ontology graph relation"
            ),
        )

    def resolve_paths(
        self,
        candidates: Sequence[OntologyEntityCandidate],
        required_roles: Sequence[SemanticRole],
        *,
        max_hops: int = 5,
        max_paths: int = 20,
    ) -> list[ResolvedSemanticPath]:
        candidate_by_uri = {item.uri: item for item in candidates}
        role_candidates: dict[SemanticRole, list[OntologyEntityCandidate]] = defaultdict(list)
        for candidate in candidates:
            for role in candidate.semantic_roles:
                role_candidates[role].append(candidate)
        for values in role_candidates.values():
            values.sort(key=lambda item: (-item.final_score, item.discovered_at_hop, item.uri))

        required_order = [role for role in ROLE_ORDER if role in set(required_roles)]
        paths: list[ResolvedSemanticPath] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

        if len(required_order) >= 2:
            stitched_nodes: list[str] = []
            stitched_predicates: list[str] = []
            stitch_ok = True
            for left_role, right_role in zip(required_order, required_order[1:]):
                left_items = role_candidates.get(left_role, [])[:4]
                right_items = role_candidates.get(right_role, [])[:4]
                segment = self._best_path(left_items, right_items, max_hops)
                if segment is None:
                    stitch_ok = False
                    break
                nodes, predicates = segment
                if not stitched_nodes:
                    stitched_nodes.extend(nodes)
                    stitched_predicates.extend(predicates)
                elif stitched_nodes[-1] == nodes[0]:
                    stitched_nodes.extend(nodes[1:])
                    stitched_predicates.extend(predicates)
                else:
                    # Link adjacent path segments with an explicit relation between their nodes.
                    bridge = self._shortest_path(
                        stitched_nodes[-1],
                        nodes[0],
                        max_hops,
                    )
                    if bridge is None:
                        stitch_ok = False
                        break
                    bridge_nodes, bridge_predicates = bridge
                    stitched_nodes.extend(bridge_nodes[1:])
                    stitched_predicates.extend(bridge_predicates)
                    stitched_nodes.extend(nodes[1:])
                    stitched_predicates.extend(predicates)
            if stitch_ok and stitched_nodes:
                roles = self._path_roles(stitched_nodes)
                paths.append(
                    ResolvedSemanticPath(
                        path_id="PATH-DYNAMIC-SEPRC-BACKBONE",
                        node_uris=stitched_nodes,
                        predicate_uris=stitched_predicates,
                        semantic_roles=roles,
                        score=max(0.0, 1.0 - 0.04 * len(stitched_predicates)),
                        source="graph_path_finder",
                        complete_backbone=set(required_order) <= set(roles),
                    )
                )
                seen.add((tuple(stitched_nodes), tuple(stitched_predicates)))

        seeds = sorted((item for item in candidates if item.seed), key=lambda item: (-item.final_score, item.uri))[:8]
        for left_index, left in enumerate(seeds):
            for right in seeds[left_index + 1 :]:
                result = self._shortest_path(left.uri, right.uri, max_hops)
                if result is None:
                    continue
                nodes, predicates = result
                key = (tuple(nodes), tuple(predicates))
                if key in seen:
                    continue
                seen.add(key)
                roles = self._path_roles(nodes)
                paths.append(
                    ResolvedSemanticPath(
                        path_id=f"PATH-DYNAMIC-{len(paths) + 1:03d}",
                        node_uris=nodes,
                        predicate_uris=predicates,
                        semantic_roles=roles,
                        score=max(0.0, 1.0 - 0.05 * len(predicates)),
                        source="schema_domain_range" if any(
                            edge.inferred_from_schema
                            for node in nodes
                            for _, edge, _ in self._adjacency.get(node, [])
                            if edge.predicate in predicates
                        ) else "graph_path_finder",
                        complete_backbone=set(required_order) <= set(roles),
                    )
                )
                if len(paths) >= max_paths:
                    break
            if len(paths) >= max_paths:
                break
        paths.sort(key=lambda item: (not item.complete_backbone, -item.score, len(item.predicate_uris), item.path_id))
        return paths[:max_paths]

    def _best_path(
        self,
        starts: Sequence[OntologyEntityCandidate],
        ends: Sequence[OntologyEntityCandidate],
        max_hops: int,
    ) -> tuple[list[str], list[str]] | None:
        best: tuple[list[str], list[str]] | None = None
        for start in starts:
            for end in ends:
                result = self._shortest_path(start.uri, end.uri, max_hops)
                if result is None:
                    continue
                if best is None or len(result[1]) < len(best[1]):
                    best = result
        return best

    def _shortest_path(self, start: str, end: str, max_hops: int) -> tuple[list[str], list[str]] | None:
        if start == end:
            return [start], []
        queue: deque[tuple[str, list[str], list[str]]] = deque([(start, [start], [])])
        visited = {start}
        while queue:
            current, nodes, predicates = queue.popleft()
            if len(predicates) >= max_hops:
                continue
            for neighbour, edge, _direction in sorted(
                self._adjacency.get(current, []), key=lambda item: (item[1].predicate, item[0])
            ):
                if neighbour in visited:
                    continue
                next_nodes = [*nodes, neighbour]
                next_predicates = [*predicates, edge.predicate]
                if neighbour == end:
                    return next_nodes, next_predicates
                visited.add(neighbour)
                queue.append((neighbour, next_nodes, next_predicates))
        return None

    def _path_roles(self, nodes: Sequence[str]) -> list[SemanticRole]:
        roles: list[SemanticRole] = []
        for uri in nodes:
            record = self.records.get(uri)
            if not record:
                continue
            for role in record.roles:
                if role in ROLE_ORDER and role not in roles:
                    roles.append(role)
        return roles

    def relation_predicates(self, relations: Sequence[OntologyRelation]) -> list[str]:
        return sorted({relation.predicate_uri for relation in relations})
