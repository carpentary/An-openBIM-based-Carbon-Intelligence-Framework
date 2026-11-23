from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph, Namespace, RDF, RDFS, SKOS

from .config import ProjectConfig
from .util import local_name, sha256_file, stable_digest


MSCA = Namespace("http://bimlca.uni.edu/ontology#")
INT = Namespace("http://bimlca.uni.edu/integration/schema#")
CPDS = Namespace("http://bimlca.uni.edu/cpd/schema#")
EFDS = Namespace("http://bimlca.uni.edu/efd/schema#")
QUDT = Namespace("http://qudt.org/schema/qudt/")

INDEX_SCHEMA_VERSION = "2"


def _text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class KnowledgeIndex:
    """Disk-backed, provenance-preserving index over the 18 RC4 CKG shards."""

    def __init__(self, config: ProjectConfig, database_path: Path | None = None):
        self.config = config
        self.database_path = database_path or (config.cache_dir / "knowledge.sqlite3")

    @property
    def source_signature(self) -> str:
        return stable_digest(
            {
                "manifest": sha256_file(self.config.ckg_manifest),
                "summary": sha256_file(self.config.ckg_summary),
                "domain_abox": sha256_file(self.config.ckg_domain_abox),
            }
        )

    def is_current(self) -> bool:
        if not self.database_path.exists():
            return False
        try:
            with self._connect() as connection:
                values = dict(connection.execute("SELECT key, value FROM meta"))
            return (
                values.get("schema_version") == INDEX_SCHEMA_VERSION
                and values.get("source_signature") == self.source_signature
            )
        except sqlite3.Error:
            return False

    def build(self, force: bool = False) -> dict[str, Any]:
        if self.is_current() and not force:
            return {"rebuilt": False, **self.stats()}
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="knowledge.", suffix=".sqlite3.tmp", dir=self.database_path.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            connection = sqlite3.connect(temporary_path)
            try:
                self._create_schema(connection)
                shard_paths = sorted(self.config.ckg_shard_dir.glob(self.config.ckg_shard_glob))
                if not shard_paths:
                    raise FileNotFoundError(f"No CKG shards match {self.config.ckg_shard_glob}")
                for shard in shard_paths:
                    self._ingest_pwp_shard(connection, shard)
                    connection.commit()
                self._ingest_domain_abox(connection, self.config.ckg_domain_abox)
                self._ingest_summary(connection, self.config.ckg_summary)
                connection.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [
                        ("schema_version", INDEX_SCHEMA_VERSION),
                        ("source_signature", self.source_signature),
                        ("ckg_release", "6.1.0-RC4"),
                        ("shard_count", str(len(shard_paths))),
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            os.replace(temporary_path, self.database_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return {"rebuilt": True, **self.stats()}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE elements (
                uri TEXT PRIMARY KEY,
                type_uri TEXT,
                system_uri TEXT,
                global_id TEXT,
                construction_method_uri TEXT,
                functional_position_uri TEXT,
                primary_material_family_uri TEXT,
                storey_uri TEXT,
                label TEXT,
                source_file TEXT NOT NULL
            );
            CREATE TABLE work_packages (
                uri TEXT PRIMARY KEY,
                element_uri TEXT,
                template_uri TEXT,
                quantity_value REAL,
                quantity_unit TEXT,
                cpd_entry_uri TEXT,
                qto_method TEXT,
                source_file TEXT NOT NULL
            );
            CREATE TABLE resources (
                uri TEXT PRIMARY KEY,
                work_package_uri TEXT,
                descriptor_uri TEXT,
                quantity_value REAL,
                quantity_unit TEXT,
                factor_uri TEXT,
                match_quality_uri TEXT,
                transport_status TEXT,
                source_file TEXT NOT NULL
            );
            CREATE TABLE emissions (
                uri TEXT PRIMARY KEY,
                resource_uri TEXT,
                factor_uri TEXT,
                stage_uri TEXT,
                value REAL,
                unit_uri TEXT,
                status_uri TEXT,
                source_file TEXT NOT NULL
            );
            CREATE TABLE summary_emissions (
                owner_uri TEXT NOT NULL,
                owner_kind TEXT NOT NULL,
                emission_uri TEXT NOT NULL,
                stage_uri TEXT,
                value REAL,
                unit_uri TEXT,
                coverage_status TEXT,
                PRIMARY KEY(owner_uri, emission_uri)
            );
            CREATE TABLE coverage_gaps (
                uri TEXT PRIMARY KEY,
                label TEXT,
                status TEXT
            );
            CREATE INDEX idx_pwp_element ON work_packages(element_uri);
            CREATE INDEX idx_element_system ON elements(system_uri);
            CREATE INDEX idx_element_global_id ON elements(global_id);
            CREATE INDEX idx_pwp_template ON work_packages(template_uri);
            CREATE INDEX idx_resource_pwp ON resources(work_package_uri);
            CREATE INDEX idx_resource_descriptor ON resources(descriptor_uri);
            CREATE INDEX idx_emission_resource ON emissions(resource_uri);
            CREATE INDEX idx_emission_stage ON emissions(stage_uri);
            CREATE INDEX idx_summary_owner ON summary_emissions(owner_uri);
            CREATE INDEX idx_summary_stage ON summary_emissions(stage_uri);
            """
        )

    @staticmethod
    def _quantity(graph: Graph, quantity_uri: Any) -> tuple[float | None, str | None]:
        if quantity_uri is None:
            return None, None
        return _number(graph.value(quantity_uri, QUDT.numericValue)), _text(graph.value(quantity_uri, QUDT.unit))

    def _ingest_pwp_shard(self, connection: sqlite3.Connection, path: Path) -> None:
        graph = Graph()
        graph.parse(path, format="turtle")
        pwp_rows: list[tuple[Any, ...]] = []
        for pwp in graph.subjects(RDF.type, MSCA.ProjectWorkPackage):
            quantity_uri = graph.value(pwp, MSCA.hasQuantity)
            quantity, unit = self._quantity(graph, quantity_uri)
            decision = graph.value(pwp, MSCA.hasCPDSelectionDecision)
            pwp_rows.append(
                (
                    str(pwp),
                    _text(graph.value(pwp, MSCA.constructsElement)),
                    _text(graph.value(pwp, MSCA.instantiatedFrom)),
                    quantity,
                    unit,
                    _text(graph.value(decision, CPDS.selectedOperationalEntry)) if decision else None,
                    _text(graph.value(pwp, INT.qtoMethod)),
                    path.name,
                )
            )
        connection.executemany(
            "INSERT OR REPLACE INTO work_packages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pwp_rows
        )

        resource_rows: list[tuple[Any, ...]] = []
        for resource in graph.subjects(RDF.type, MSCA.ResourceInstance):
            quantity_uri = graph.value(resource, MSCA.hasActualQuantity)
            quantity, unit = self._quantity(graph, quantity_uri)
            decision = graph.value(resource, MSCA.hasFactorSelectionDecision)
            resource_rows.append(
                (
                    str(resource),
                    _text(graph.value(resource, MSCA.consumedByWorkPackage)),
                    _text(graph.value(resource, MSCA.describedBy)),
                    quantity,
                    unit,
                    _text(graph.value(decision, EFDS.selectedFactor)) if decision else None,
                    _text(graph.value(resource, INT.hasMatchQuality)),
                    _text(graph.value(resource, INT.transportStatus)),
                    path.name,
                )
            )
        connection.executemany(
            "INSERT OR REPLACE INTO resources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", resource_rows
        )

        emission_rows: list[tuple[Any, ...]] = []
        for emission in graph.subjects(RDF.type, MSCA.CarbonEmission):
            quantity_uri = graph.value(emission, MSCA.hasEmissionQuantity)
            value, unit = self._quantity(graph, quantity_uri)
            emission_rows.append(
                (
                    str(emission),
                    _text(graph.value(emission, MSCA.calculatedFromResource)),
                    _text(graph.value(emission, MSCA.calculatedUsingFactor)),
                    _text(graph.value(emission, MSCA.belongsToStage)),
                    value,
                    unit,
                    _text(graph.value(emission, INT.hasCalculationStatus)),
                    path.name,
                )
            )
        connection.executemany(
            "INSERT OR REPLACE INTO emissions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", emission_rows
        )

    def _ingest_domain_abox(self, connection: sqlite3.Connection, path: Path) -> None:
        graph = Graph()
        graph.parse(path, format="turtle")
        element_rows: list[tuple[Any, ...]] = []
        candidates = set(graph.subjects(MSCA.belongsToSystem, None))
        for element in candidates:
            local_types = [
                value
                for value in graph.objects(element, RDF.type)
                if str(value).startswith(str(MSCA))
            ]
            element_rows.append(
                (
                    str(element),
                    _text(local_types[0]) if local_types else None,
                    _text(graph.value(element, MSCA.belongsToSystem)),
                    _text(graph.value(element, MSCA.hasGlobalID)),
                    _text(graph.value(element, MSCA.hasConstructionMethod)),
                    _text(graph.value(element, MSCA.hasFunctionalPosition)),
                    _text(graph.value(element, MSCA.hasPrimaryMaterialFamily)),
                    _text(graph.value(element, MSCA.isLocatedIn)),
                    _text(graph.value(element, RDFS.label)) or _text(graph.value(element, SKOS.prefLabel)),
                    path.name,
                )
            )
        connection.executemany(
            "INSERT OR REPLACE INTO elements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            element_rows,
        )

    def _ingest_summary(self, connection: sqlite3.Connection, path: Path) -> None:
        graph = Graph()
        graph.parse(path, format="turtle")
        assessments = {str(value) for value in graph.subjects(RDF.type, MSCA.CarbonAssessment)}
        rows: list[tuple[Any, ...]] = []
        for owner, emission in graph.subject_objects(MSCA.hasEmissions):
            quantity_uri = graph.value(emission, MSCA.hasEmissionQuantity)
            value, unit = self._quantity(graph, quantity_uri)
            rows.append(
                (
                    str(owner),
                    "assessment" if str(owner) in assessments else "element",
                    str(emission),
                    _text(graph.value(emission, MSCA.belongsToStage)),
                    value,
                    unit,
                    _text(graph.value(emission, INT.coverageStatus)),
                )
            )
        connection.executemany(
            "INSERT OR REPLACE INTO summary_emissions VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        gap_rows: list[tuple[str, str | None, str | None]] = []
        for assessment in graph.subjects(RDF.type, MSCA.CarbonAssessment):
            for gap in graph.objects(assessment, INT.hasCoverageGap):
                gap_rows.append(
                    (
                        str(gap),
                        _text(graph.value(gap, RDFS.label)) or local_name(str(gap)),
                        _text(graph.value(gap, INT.coverageStatus)),
                    )
                )
        connection.executemany("INSERT OR REPLACE INTO coverage_gaps VALUES (?, ?, ?)", gap_rows)

    def stats(self) -> dict[str, Any]:
        if not self.database_path.exists():
            return {"database": str(self.database_path), "ready": False}
        with self._connect() as connection:
            available = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("elements", "work_packages", "resources", "emissions", "summary_emissions", "coverage_gaps")
                if table in available
            }
            meta = dict(connection.execute("SELECT key, value FROM meta")) if "meta" in available else {}
        return {"database": str(self.database_path), "ready": self.is_current(), "counts": counts, "meta": meta}

    def project_overview(self) -> dict[str, Any]:
        self._require_ready()
        with self._connect() as connection:
            stage_rows = connection.execute(
                """
                SELECT stage_uri, SUM(value) AS value, MIN(unit_uri) AS unit_uri
                FROM summary_emissions
                WHERE owner_kind='assessment'
                GROUP BY stage_uri
                ORDER BY stage_uri
                """
            ).fetchall()
            gaps = [dict(row) for row in connection.execute("SELECT uri, label, status FROM coverage_gaps ORDER BY uri")]
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("elements", "work_packages", "resources", "emissions")
            }
        baseline_totals = [
            {
                "stage": local_name(row["stage_uri"]),
                "value": row["value"],
                "unit": local_name(row["unit_uri"]),
            }
            for row in stage_rows
        ]
        reference_path = self.config.artifact_dir / "reference_case" / "reference_case_summary.json"
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            totals = [
                {"stage": stage, "value": value, "unit": "Unit_KgCO2e"}
                for stage, value in reference["combined_stage_totals_kgco2e"].items()
            ]
            main_stage_totals = {
                "A1-A3_ProductStage": float(
                    reference["main_structure"]["baseline_stage_totals_kgco2e"].get(
                        "A1-A3_ProductStage", 0.0
                    )
                ),
                "A4_TransportationStage": float(reference["main_structure"]["A4_kgco2e"]),
                "A5_ConstructionStage": float(
                    reference["main_structure"]["baseline_stage_totals_kgco2e"].get(
                        "A5_ConstructionStage", 0.0
                    )
                ),
            }
            scope_stage_totals = {
                "main_structure": main_stage_totals,
                "enclosure_structure": {
                    stage: float(value)
                    for stage, value in reference["enclosure_structure"]["stage_totals_kgco2e"].items()
                },
                "foundation_pit": {
                    stage: float(value)
                    for stage, value in reference["foundation_pit"]["stage_totals_kgco2e"].items()
                },
            }
            return {
                "assessment_status": reference["calculation_status"],
                "applicability_scope": reference["profile"]["applicability_scope"],
                "profile": reference.get("profile", {}),
                "formulae": reference.get("formulae", {}),
                "model_coverage": {
                    "main_structure": "3,550 IFC products in the governed RC4 semantic baseline plus default A4 transport",
                    "enclosure_structure": "4,024 applicable IFC products mapped to CPD/EFD; one display-only surface is not applicable",
                    "foundation_pit": "formula-driven reference profile from public station geometry and declared parameters",
                },
                "totals": totals,
                "combined_total_kgco2e": reference["combined_total_kgco2e"],
                "scope_stage_totals_kgco2e": scope_stage_totals,
                "main_structure": reference["main_structure"],
                "main_structure_baseline_totals": baseline_totals,
                "foundation_pit": reference["foundation_pit"],
                "enclosure_structure": reference["enclosure_structure"],
                "transport": reference["transport"],
                "public_geometry": reference.get("public_geometry", {}),
                "public_construction_context": reference.get("public_construction_context", {}),
                "provenance_vocabulary": reference.get("provenance_vocabulary", {}),
                "sources": reference.get("sources", []),
                "parameters": reference.get("parameters", {}),
                "counts": counts,
                "coverage_gaps": gaps,
                "source": str(reference_path),
                "reference_case_artifacts": reference["artifacts"],
            }
        return {
            "assessment_status": "REFERENCE_CASE_NOT_BUILT",
            "model_coverage": {
                "main_structure": "indexed in the governed RC4 carbon baseline (3,550 IFC products)",
                "enclosure_structure": "input ABox and QTO are available; run case-build for the combined calculation",
            },
            "totals": baseline_totals,
            "counts": counts,
            "coverage_gaps": gaps,
            "source": str(self.config.ckg_summary),
        }

    def top_elements(self, limit: int = 10, stage: str | None = None) -> list[dict[str, Any]]:
        self._require_ready()
        parameters: list[Any] = []
        stage_clause = ""
        if stage:
            stage_clause = " AND e.stage_uri LIKE ?"
            parameters.append(f"%{stage}%")
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT p.element_uri AS owner_uri, SUM(e.value) AS value,
                       MIN(e.unit_uri) AS unit_uri,
                       GROUP_CONCAT(DISTINCT e.stage_uri) AS stages
                FROM work_packages p
                JOIN resources r ON r.work_package_uri=p.uri
                JOIN emissions e ON e.resource_uri=r.uri
                WHERE p.element_uri IS NOT NULL {stage_clause}
                GROUP BY p.element_uri
                ORDER BY value DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [
            {
                "element_uri": row["owner_uri"],
                "element": local_name(row["owner_uri"]),
                "value": row["value"],
                "unit": local_name(row["unit_uri"]),
                "stages": [local_name(item) for item in (row["stages"] or "").split(",") if item],
            }
            for row in rows
        ]

    def resource_breakdown(self, term: str, stage: str | None = None, limit: int = 30) -> dict[str, Any]:
        self._require_ready()
        like = f"%{term}%"
        parameters: list[Any] = [like]
        stage_clause = ""
        if stage:
            stage_clause = " AND e.stage_uri LIKE ?"
            parameters.append(f"%{stage}%")
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT r.descriptor_uri, e.stage_uri, SUM(e.value) AS emission_value,
                       MIN(e.unit_uri) AS emission_unit, COUNT(DISTINCT r.uri) AS resource_count
                FROM resources r
                JOIN emissions e ON e.resource_uri=r.uri
                WHERE r.descriptor_uri LIKE ? {stage_clause}
                GROUP BY r.descriptor_uri, e.stage_uri
                ORDER BY emission_value DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        items = [
            {
                "descriptor_uri": row["descriptor_uri"],
                "descriptor": local_name(row["descriptor_uri"]),
                "stage": local_name(row["stage_uri"]),
                "emission_value": row["emission_value"],
                "emission_unit": local_name(row["emission_unit"]),
                "resource_count": row["resource_count"],
            }
            for row in rows
        ]
        return {
            "term": term,
            "items": items,
            "total_emission": sum(item["emission_value"] or 0.0 for item in items),
            "unit": items[0]["emission_unit"] if items else None,
        }

    def work_package_breakdown(self, term: str, limit: int = 30) -> list[dict[str, Any]]:
        self._require_ready()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT template_uri, COUNT(*) AS work_package_count,
                       SUM(quantity_value) AS quantity_value, MIN(quantity_unit) AS quantity_unit
                FROM work_packages
                WHERE template_uri LIKE ?
                GROUP BY template_uri
                ORDER BY work_package_count DESC
                LIMIT ?
                """,
                (f"%{term}%", limit),
            ).fetchall()
        return [
            {
                "template_uri": row["template_uri"],
                "template": local_name(row["template_uri"]),
                "work_package_count": row["work_package_count"],
                "quantity_value": row["quantity_value"],
                "quantity_unit": local_name(row["quantity_unit"]),
            }
            for row in rows
        ]

    def element_trace(self, element_uri: str, limit: int = 250) -> dict[str, Any]:
        """Return an auditable Element -> PWP -> Resource -> Carbon chain."""
        self._require_ready()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.element_uri, d.type_uri AS element_type_uri, d.system_uri,
                       d.global_id, d.construction_method_uri, d.functional_position_uri,
                       d.primary_material_family_uri, d.storey_uri, d.label AS element_label,
                       p.uri AS work_package_uri, p.template_uri,
                       p.quantity_value AS work_quantity_value, p.quantity_unit AS work_quantity_unit,
                       p.cpd_entry_uri, p.qto_method, p.source_file,
                       r.uri AS resource_uri, r.descriptor_uri,
                       r.quantity_value AS resource_quantity_value, r.quantity_unit AS resource_quantity_unit,
                       r.factor_uri, r.match_quality_uri, r.transport_status,
                       e.uri AS emission_uri, e.stage_uri, e.value AS emission_value,
                       e.unit_uri AS emission_unit, e.status_uri AS calculation_status
                FROM work_packages p
                LEFT JOIN elements d ON d.uri=p.element_uri
                LEFT JOIN resources r ON r.work_package_uri=p.uri
                LEFT JOIN emissions e ON e.resource_uri=r.uri
                WHERE p.element_uri=?
                ORDER BY p.uri, r.uri, e.stage_uri
                LIMIT ?
                """,
                (element_uri, limit),
            ).fetchall()
        chains = [dict(row) for row in rows]
        return {
            "semantic_route": "System -> Element -> ProjectWorkPackage -> ResourceInstance -> CarbonEmission",
            "system_uri": chains[0]["system_uri"] if chains else None,
            "system": local_name(chains[0]["system_uri"]) if chains else None,
            "element_uri": element_uri,
            "element": local_name(element_uri),
            "element_metadata": {
                key: chains[0].get(key) if chains else None
                for key in (
                    "element_type_uri",
                    "global_id",
                    "construction_method_uri",
                    "functional_position_uri",
                    "primary_material_family_uri",
                    "storey_uri",
                    "element_label",
                )
            },
            "chain_rows": chains,
            "chain_row_count": len(chains),
            "distinct_work_packages": len({row["work_package_uri"] for row in chains if row["work_package_uri"]}),
            "distinct_resources": len({row["resource_uri"] for row in chains if row["resource_uri"]}),
            "distinct_emissions": len({row["emission_uri"] for row in chains if row["emission_uri"]}),
            "emission_sum_in_returned_rows": sum(float(row["emission_value"] or 0.0) for row in chains),
            "truncated": len(chains) == limit,
            "boundary": "Trace belongs to the governed main-structure RC4 baseline; enclosure is not included.",
        }

    def _require_ready(self) -> None:
        if not self.is_current():
            raise RuntimeError("Knowledge index is absent or stale; run `msca-mas prepare` first")
