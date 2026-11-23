from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IfcSource:
    path: Path
    role: str


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved, immutable paths and declared assessment boundaries."""

    config_path: Path
    project_root: Path
    project_id: str
    project_name: str
    ifc_sources: tuple[IfcSource, ...]
    tbox: Path
    complete_schema: Path
    catalogue: Path
    swrl: Path
    shacl: Path
    cpd_reference: Path
    efd_reference: Path
    ontology_release_report: Path
    swrl_regression_input: Path
    swrl_chainer: Path
    cq_catalogue: Path
    cq_manifest: Path
    cq_query_dir: Path
    cq_fixture: Path
    ckg_root: Path
    ckg_manifest: Path
    ckg_summary: Path
    ckg_shard_dir: Path
    ckg_shard_glob: str
    ckg_domain_abox: Path
    topology_facts: Path
    unresolved_topology_register: Path
    topology_coordinate_unit: str
    topology_connection_epsilon: float
    topology_containment_epsilon: float
    included_stages: tuple[str, ...]
    known_pending: tuple[str, ...]
    reference_case_dir: Path
    cache_dir: Path
    artifact_dir: Path
    transaction_dir: Path
    metadata_backend: str
    metadata_dense_model_id: str
    metadata_reranker_model_id: str
    metadata_dense_model_revision: str
    metadata_reranker_model_revision: str
    metadata_offline: bool
    metadata_max_rounds: int
    metadata_top_k: int
    metadata_expansion_hops: int
    metadata_candidate_limit: int
    structured_output_max_retries: int

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ProjectConfig":
        if config_path is None:
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / "config" / "project.toml"
        else:
            path = Path(config_path).expanduser().resolve()
            project_root = path.parent.parent
        with path.open("rb") as handle:
            data = tomllib.load(handle)

        mas = data.get("mas", {})
        metadata = mas.get("metadata", {})
        structured_output = mas.get("structured_output", {})

        def resolved(value: str) -> Path:
            return (path.parent / value).resolve()

        ifc_sources = tuple(
            IfcSource(resolved(data["ifc"][key]["path"]), data["ifc"][key]["role"])
            for key in ("main", "enclosure")
        )
        return cls(
            config_path=path,
            project_root=project_root,
            project_id=data["project"]["id"],
            project_name=data["project"]["name"],
            ifc_sources=ifc_sources,
            tbox=resolved(data["ontology"]["tbox"]),
            complete_schema=resolved(data["ontology"]["complete_schema"]),
            catalogue=resolved(data["ontology"]["catalogue"]),
            swrl=resolved(data["ontology"]["swrl"]),
            shacl=resolved(data["ontology"]["shacl"]),
            cpd_reference=resolved(data["ontology"]["cpd_reference"]),
            efd_reference=resolved(data["ontology"]["efd_reference"]),
            ontology_release_report=resolved(data["ontology"]["release_report"]),
            swrl_regression_input=resolved(data["ontology"]["swrl_regression_input"]),
            swrl_chainer=resolved(data["ontology"]["swrl_chainer"]),
            cq_catalogue=resolved(data["ontology"]["cq_catalogue"]),
            cq_manifest=resolved(data["ontology"]["cq_manifest"]),
            cq_query_dir=resolved(data["ontology"]["cq_query_dir"]),
            cq_fixture=resolved(data["ontology"]["cq_fixture"]),
            ckg_root=resolved(data["ckg"]["root"]),
            ckg_manifest=resolved(data["ckg"]["manifest"]),
            ckg_summary=resolved(data["ckg"]["summary"]),
            ckg_shard_dir=resolved(data["ckg"]["shard_dir"]),
            ckg_shard_glob=data["ckg"]["shard_glob"],
            ckg_domain_abox=resolved(data["ckg"]["domain_abox"]),
            topology_facts=resolved(data["topology"]["governed_facts"]),
            unresolved_topology_register=resolved(data["topology"]["unresolved_register"]),
            topology_coordinate_unit=data["topology"]["coordinate_unit"],
            topology_connection_epsilon=float(data["topology"]["reconstructed_connection_epsilon"]),
            topology_containment_epsilon=float(data["topology"]["reconstructed_containment_epsilon"]),
            included_stages=tuple(data["assessment"]["included_stages"]),
            known_pending=tuple(data["assessment"]["known_pending"]),
            reference_case_dir=resolved(data["reference_case"]["directory"]),
            cache_dir=resolved(data["runtime"]["cache_dir"]),
            artifact_dir=resolved(data["runtime"]["artifact_dir"]),
            transaction_dir=resolved(data["runtime"]["transaction_dir"]),
            metadata_backend=str(metadata.get("backend", "auto")),
            metadata_dense_model_id=str(metadata.get("dense_model_id", "BAAI/bge-m3")),
            metadata_reranker_model_id=str(metadata.get("reranker_model_id", "BAAI/bge-reranker-v2-m3")),
            metadata_dense_model_revision=str(
                metadata.get(
                    "dense_model_revision",
                    "5617a9f61b028005a4858fdac845db406aefb181",
                )
            ),
            metadata_reranker_model_revision=str(
                metadata.get(
                    "reranker_model_revision",
                    "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                )
            ),
            metadata_offline=bool(metadata.get("offline", True)),
            metadata_max_rounds=int(metadata.get("max_rounds", 3)),
            metadata_top_k=int(metadata.get("top_k", 8)),
            metadata_expansion_hops=int(metadata.get("expansion_hops", 2)),
            metadata_candidate_limit=int(metadata.get("candidate_limit", 80)),
            structured_output_max_retries=int(structured_output.get("max_retries", 2)),
        )

    def required_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {
            "tbox": self.tbox,
            "complete_schema": self.complete_schema,
            "catalogue": self.catalogue,
            "swrl": self.swrl,
            "shacl": self.shacl,
            "cpd_reference": self.cpd_reference,
            "efd_reference": self.efd_reference,
            "ontology_release_report": self.ontology_release_report,
            "swrl_regression_input": self.swrl_regression_input,
            "swrl_chainer": self.swrl_chainer,
            "cq_catalogue": self.cq_catalogue,
            "cq_manifest": self.cq_manifest,
            "cq_query_dir": self.cq_query_dir,
            "cq_fixture": self.cq_fixture,
            "ckg_root": self.ckg_root,
            "ckg_manifest": self.ckg_manifest,
            "ckg_summary": self.ckg_summary,
            "ckg_shard_dir": self.ckg_shard_dir,
            "ckg_domain_abox": self.ckg_domain_abox,
            "topology_facts": self.topology_facts,
            "unresolved_topology_register": self.unresolved_topology_register,
            "reference_case_dir": self.reference_case_dir,
        }
        paths.update({f"ifc_{source.role}": source.path for source in self.ifc_sources})
        return paths

    def doctor(self) -> dict[str, Any]:
        checks = {
            name: {"path": str(path), "exists": path.exists(), "kind": "directory" if path.is_dir() else "file"}
            for name, path in self.required_paths().items()
        }
        shards = sorted(self.ckg_shard_dir.glob(self.ckg_shard_glob)) if self.ckg_shard_dir.exists() else []
        return {
            "project_id": self.project_id,
            "config": str(self.config_path),
            "ok": all(item["exists"] for item in checks.values()) and bool(shards),
            "paths": checks,
            "ckg_shard_count": len(shards),
            "included_stages": list(self.included_stages),
            "known_pending": list(self.known_pending),
            "mas_architecture": {
                "metadata_backend": self.metadata_backend,
                "dense_model_id": self.metadata_dense_model_id,
                "reranker_model_id": self.metadata_reranker_model_id,
                "dense_model_revision": self.metadata_dense_model_revision,
                "reranker_model_revision": self.metadata_reranker_model_revision,
                "offline": self.metadata_offline,
                "max_metadata_rounds": self.metadata_max_rounds,
                "structured_output_max_retries": self.structured_output_max_retries,
            },
            "topology_tolerances": {
                "coordinate_unit": self.topology_coordinate_unit,
                "connection_epsilon": self.topology_connection_epsilon,
                "containment_epsilon": self.topology_containment_epsilon,
                "status": "controlled runtime parameter profile",
            },
        }
