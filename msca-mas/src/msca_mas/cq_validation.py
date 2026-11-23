from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdflib import Graph

from .config import ProjectConfig


def run_cq_checks(config: ProjectConfig, cq_ids: list[str] | None = None) -> dict[str, Any]:
    """Run the configured CQ queries against the validation graph."""
    graph = Graph()
    source_files = [
        config.complete_schema,
        config.cpd_reference,
        config.efd_reference,
        config.cq_fixture,
    ]
    for path in source_files:
        graph.parse(path, format="turtle")
    manifest = json.loads(config.cq_manifest.read_text(encoding="utf-8"))
    selected = sorted(cq_ids or manifest)
    results: list[dict[str, Any]] = []
    for cq_id in selected:
        if cq_id not in manifest:
            results.append({"cq_id": cq_id, "status": "ERROR", "error": "CQ ID is absent from manifest"})
            continue
        entry = manifest[cq_id]
        query_path = config.cq_query_dir / Path(entry["query_file"]).name
        minimum = int(entry["expected_minimum_rows_in_fixture"])
        try:
            rows = list(graph.query(query_path.read_text(encoding="utf-8")))
            status = "PASS" if len(rows) >= minimum else "FAIL"
            results.append(
                {
                    "cq_id": cq_id,
                    "status": status,
                    "rows": len(rows),
                    "minimum": minimum,
                    "query_file": str(query_path),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "cq_id": cq_id,
                    "status": "ERROR",
                    "rows": None,
                    "minimum": minimum,
                    "query_file": str(query_path),
                    "error": str(exc),
                }
            )
    passed = sum(result["status"] == "PASS" for result in results)
    return {
        "status": "PASS" if passed == len(results) else "FAIL",
        "passed": passed,
        "total": len(results),
        "graph_triples": len(graph),
        "source_files": [str(path) for path in source_files],
        "results": results,
    }
