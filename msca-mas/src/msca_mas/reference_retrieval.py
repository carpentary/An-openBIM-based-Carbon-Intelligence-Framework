from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdflib import Graph, RDF, URIRef

from .config import ProjectConfig
from .ontology_retrieval import SemanticBackend
from .util import local_name


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    values = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", lowered)
    for run in re.findall(r"[\u3400-\u9fff]+", lowered):
        values.append(run)
        values.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return values


def _normalise(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _literal_values(graph: Graph, subject: Any, predicate_name: str) -> list[str]:
    values = [
        str(value)
        for predicate, value in graph.predicate_objects(subject)
        if local_name(str(predicate)) == predicate_name
    ]
    return sorted(dict.fromkeys(values), key=lambda value: (value.casefold(), value))


def _first(graph: Graph, subject: Any, predicate_name: str) -> str | None:
    values = _literal_values(graph, subject, predicate_name)
    return values[0] if values else None


def _first_term(graph: Graph, subject: Any, predicate_name: str) -> Any | None:
    return next(
        (
            value
            for predicate, value in graph.predicate_objects(subject)
            if local_name(str(predicate)) == predicate_name
        ),
        None,
    )


@dataclass
class ReferenceRecord:
    dataset: str
    uri: str
    code: str
    names: list[str]
    category: str = ""
    description: str = ""
    region: str | None = None
    reference_year: int | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return " | ".join(
            value
            for value in (
                self.dataset,
                self.code,
                *self.names,
                self.category,
                self.description,
                self.region or "",
                str(self.reference_year or ""),
                " ".join(f"{key} {value}" for key, value in self.fields.items()),
            )
            if value
        )

    def as_result(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "uri": self.uri,
            "code": self.code,
            "names": self.names,
            "category": self.category,
            "description": self.description,
            "region": self.region,
            "reference_year": self.reference_year,
            **self.fields,
        }


class ReferenceDatabaseRetriever:
    """Reader-side CPD/EFD/transport hybrid retrieval with hard filters.

    This index is deliberately separate from the ontology metadata index: the
    Metadata Agent resolves schema and entity pointers, while the Reader Agent
    resolves governed parameter records and values.
    """

    def __init__(self, config: ProjectConfig, backend: SemanticBackend):
        self.config = config
        self.backend = backend
        self.records = [
            *self._load_cpd(config.cpd_reference),
            *self._load_efd(config.efd_reference),
            *self._load_transport(config.reference_case_dir),
        ]

    @staticmethod
    def predict_categories(query: str) -> list[str]:
        lowered = query.lower()
        categories: list[str] = []

        def add(category: str) -> None:
            if category not in categories:
                categories.append(category)

        if any(term in lowered for term in ("cpd", "work package", "quota", "process", "工序", "工作包", "定额")):
            add("CPD")
        if any(term in lowered for term in ("efd", "emission factor", "factor", "排放因子", "因子")):
            add("EFD")
        if any(term in lowered for term in ("a4", "transport", "vehicle", "distance", "haul", "运输", "运距", "载具")):
            add("TRANSPORT")
        if not categories:
            if any(term in lowered for term in ("material", "equipment", "energy", "resource", "材料", "设备", "能源", "资源")):
                categories.extend(["EFD", "CPD"])
            else:
                categories.extend(["CPD", "EFD"])
        return categories

    @staticmethod
    def _load_cpd(path: Path) -> list[ReferenceRecord]:
        graph = Graph().parse(path)
        records: list[ReferenceRecord] = []
        for subject, object_type in graph.subject_objects(RDF.type):
            if not isinstance(subject, URIRef) or local_name(str(object_type)) != "CPD_Entry":
                continue
            names = _literal_values(graph, subject, "hasEntryName")
            quota_uris = _literal_values(graph, subject, "hasQuota")
            records.append(
                ReferenceRecord(
                    dataset="CPD",
                    uri=str(subject),
                    code=_first(graph, subject, "hasEntryCode") or local_name(str(subject)),
                    names=names or [local_name(str(subject))],
                    category=local_name(_first(graph, subject, "belongsToSubdivision") or ""),
                    description=_first(graph, subject, "hasTechnicalDescription") or "",
                    fields={
                        "quota_count": len(quota_uris),
                        "quota_uris": quota_uris,
                        "source": str(path),
                    },
                )
            )
        return records

    @staticmethod
    def _load_efd(path: Path) -> list[ReferenceRecord]:
        graph = Graph().parse(path)
        records: list[ReferenceRecord] = []
        for subject, object_type in graph.subject_objects(RDF.type):
            if not isinstance(subject, URIRef) or local_name(str(object_type)) != "EFD_Entry":
                continue
            factor_uri = _first_term(graph, subject, "hasFactor")
            factor_uri_text = str(factor_uri) if factor_uri is not None else None
            quantity_uri = _first_term(graph, factor_uri, "factorValue") if factor_uri is not None else None
            numeric_value = _first(graph, quantity_uri, "numericValue") if quantity_uri is not None else None
            unit_uri = _first(graph, quantity_uri, "unit") if quantity_uri is not None else None
            year_text = _first(graph, factor_uri, "hasReferenceYear") if factor_uri else None
            region = _first(graph, factor_uri, "hasGeographicScope") if factor_uri else None
            try:
                reference_year = int(float(year_text)) if year_text else None
            except ValueError:
                reference_year = None
            records.append(
                ReferenceRecord(
                    dataset="EFD",
                    uri=str(subject),
                    code=_first(graph, subject, "hasEFDEntryCode") or local_name(str(subject)),
                    names=_literal_values(graph, subject, "hasEFDEntryName") or [local_name(str(subject))],
                    category=local_name(_first(graph, subject, "belongsToSubcategory") or ""),
                    region=region,
                    reference_year=reference_year,
                    fields={
                        "factor_uri": factor_uri_text,
                        "factor_resource_uri": _first(graph, factor_uri, "factorResource") if factor_uri else None,
                        "factor_value": float(numeric_value) if numeric_value is not None else None,
                        "factor_unit_uri": unit_uri,
                        "data_tier": local_name(_first(graph, factor_uri, "hasDataTier") or "") if factor_uri else None,
                        "temporal_scope": _first(graph, factor_uri, "hasTemporalScope") if factor_uri else None,
                        "source": str(path),
                    },
                )
            )
        return records

    @staticmethod
    def _load_transport(directory: Path) -> list[ReferenceRecord]:
        records: list[ReferenceRecord] = []
        sources = (
            (directory / "material_transport_defaults.csv", "material_code", "material_name"),
            (directory / "transport_vehicles.csv", "vehicle_code", "vehicle_name"),
        )
        for path, code_field, name_field in sources:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    code = str(row.get(code_field, ""))
                    name = str(row.get(name_field, ""))
                    records.append(
                        ReferenceRecord(
                            dataset="TRANSPORT",
                            uri=f"reference://transport/{code}",
                            code=code,
                            names=[name] if name else [code],
                            category="vehicle" if code_field == "vehicle_code" else "material_default",
                            region="China",
                            fields={**row, "source": str(path)},
                        )
                    )
        return records

    @staticmethod
    def _bm25(
        query_tokens: Sequence[str],
        frequencies: Counter[str],
        document_frequency: Counter[str],
        document_count: int,
        document_length: int,
        average_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * document_length / max(average_length, 1.0))
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        return score

    @staticmethod
    def _passes_filters(record: ReferenceRecord, filters: dict[str, Any]) -> bool:
        region = filters.get("region")
        if region and record.region:
            record_region = str(record.region).casefold()
            expected_region = str(region).casefold()
            if expected_region not in record_region and record_region not in expected_region:
                return False
        minimum_year = filters.get("reference_year_min")
        maximum_year = filters.get("reference_year_max")
        if minimum_year is not None and record.reference_year is not None and record.reference_year < int(minimum_year):
            return False
        if maximum_year is not None and record.reference_year is not None and record.reference_year > int(maximum_year):
            return False
        code = filters.get("code") or filters.get("material_code") or filters.get("vehicle_code")
        if code and record.code.casefold() != str(code).casefold():
            return False
        category = filters.get("category")
        if category and str(category).casefold() not in record.category.casefold():
            return False
        return True

    def search(
        self,
        query: str,
        *,
        categories: Iterable[str] = (),
        hard_filters: dict[str, Any] | None = None,
        top_k: int = 8,
        lexical_only: bool = False,
    ) -> dict[str, Any]:
        predicted = list(dict.fromkeys(value.upper() for value in categories if value)) or self.predict_categories(query)
        filters = dict(hard_filters or {})
        candidates = [
            record
            for record in self.records
            if record.dataset in predicted and self._passes_filters(record, filters)
        ]
        if not candidates:
            return {
                "category_prediction": predicted,
                "hard_filters": filters,
                "backend": self.backend.info.model_dump(mode="json"),
                "retrieval_mode": "lexical_only" if lexical_only else "hybrid",
                "candidate_count_after_hard_filters": 0,
                "results": [],
            }
        documents = [record.searchable_text for record in candidates]
        frequencies = [Counter(_tokens(document)) for document in documents]
        document_frequency: Counter[str] = Counter()
        for value in frequencies:
            document_frequency.update(value.keys())
        average_length = sum(sum(value.values()) for value in frequencies) / len(frequencies)
        lexical = _normalise(
            [
                self._bm25(
                    _tokens(query),
                    value,
                    document_frequency,
                    len(frequencies),
                    sum(value.values()),
                    average_length,
                )
                for value in frequencies
            ]
        )
        if lexical_only:
            # In lexical-only mode, skip vector similarity and cross-encoder reranking.
            dense = [0.0] * len(candidates)
        else:
            dense = self.backend.similarities(query, documents)
        coarse: list[tuple[int, float, float, float]] = []
        query_compact = re.sub(r"[^a-z0-9]+", "", query.lower())
        for index, record in enumerate(candidates):
            exact = 1.0 if record.code.lower() in query.lower() or (
                query_compact and query_compact in re.sub(r"[^a-z0-9]+", "", record.searchable_text.lower())
            ) else 0.0
            fused = min(1.0, 0.45 * lexical[index] + 0.55 * dense[index] + 0.20 * exact)
            coarse.append((index, fused, lexical[index], dense[index]))
        coarse.sort(key=lambda item: (-item[1], candidates[item[0]].code))
        coarse = coarse[: max(30, top_k * 5)]
        if lexical_only:
            rerank = [fused for _index, fused, *_lexical, _dense in coarse]
        else:
            rerank = self.backend.rerank(query, [documents[index] for index, *_ in coarse])
        ranked: list[dict[str, Any]] = []
        for (index, fused, lexical_score, dense_score), rerank_score in zip(coarse, rerank, strict=True):
            record = candidates[index]
            ranked.append(
                {
                    **record.as_result(),
                    "retrieval_scores": {
                        "bm25": round(lexical_score, 8),
                        "dense": round(dense_score, 8),
                        "fused": round(fused, 8),
                        "rerank": round(float(rerank_score), 8),
                        "final": round(0.64 * fused + 0.36 * float(rerank_score), 8),
                    },
                }
            )
        ranked.sort(key=lambda item: (-item["retrieval_scores"]["final"], item["code"]))
        return {
            "category_prediction": predicted,
            "hard_filters": filters,
            "backend": self.backend.info.model_dump(mode="json"),
            "retrieval_mode": "lexical_only" if lexical_only else "hybrid",
            "candidate_count_after_hard_filters": len(candidates),
            "results": ranked[:top_k],
        }
