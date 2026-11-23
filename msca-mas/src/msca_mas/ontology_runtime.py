"""SHACL checks, SWRL inference, and SPARQL queries for the project graph.

Release validation tracks new or increased violation groups. Correction
validation checks the proposed change separately.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from rdflib import BNode, Graph, URIRef

from .config import ProjectConfig
from .util import atomic_write_json, sha256_file, stable_digest

_SHACL = "http://www.w3.org/ns/shacl#"

# Replace affected release shapes with the bundled overrides.
SPEC_ILLEGAL_SHAPE_PATCHES: dict[str, str] = {
    "http://bimlca.uni.edu/ontology#ExecutableElementMappingShape": (
        "VALUES clause in sh:sparql violates the SHACL SPARQL constraint "
        "grammar (SHACL Spec §5.3.2 forbids pre-bound-variable interaction); "
        "pyshacl aborts the entire validation once the constraint triggers. "
        "Replaced with the semantically identical FILTER(?status IN (...)) form."
    ),
}

_PATCH_RESOURCE = Path(__file__).parent / "resources" / "shacl_overrides.ttl"

_SHAPE_FILES = (
    "ontology/MSCA-Shapes-v6.1.0-RC4.ttl",
    "ontology/MSCA-Shapes-RC4-Additions.ttl",
)

# Input graphs are relative to the CKG release root.
_RELEASE_DATA_DIRS = (
    "data/project/ckg",
    "data/project/source_abox",
    "data/cpd",
    "data/efd",
    "data/catalogue",
)

_RELEASE_SCHEMA_FILES = (
    "ontology/MSCA-Core-v6.0.0.ttl",
    "ontology/MSCA-CPD-Schema-v6.1.0.ttl",
    "ontology/MSCA-EFD-Schema-v6.1.0.ttl",
    "ontology/MSCA-Catalogue-v6.0.0.ttl",
    "ontology/MSCA-IFC-Bridge-v6.0.0.ttl",
    "ontology/MSCA-Integration-QTO-v6.1.0.ttl",
)

_RELEASE_SUMMARY = "data/project/ckg/Station01-CKG-Summary-RC4.ttl"


def _remove_shape(shapes: Graph, node: URIRef) -> int:
    """Remove a shape and its reachable blank-node subtrees (constraint bodies)."""

    from rdflib import BNode

    removed = 0
    frontier = [node]
    seen: set[Any] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        triples = list(shapes.triples((current, None, None)))
        for triple in triples:
            shapes.remove(triple)
            removed += 1
            if isinstance(triple[2], BNode):
                frontier.append(triple[2])
    return removed


def load_shapes(config: ProjectConfig) -> tuple[Graph, list[dict[str, Any]]]:
    """Load release shapes and apply the bundled overrides."""

    shapes = Graph()
    for relative in _SHAPE_FILES:
        shapes.parse(config.ckg_root / relative, format="turtle")
    patches: list[dict[str, Any]] = []
    for iri, reason in SPEC_ILLEGAL_SHAPE_PATCHES.items():
        node = URIRef(iri)
        removed = _remove_shape(shapes, node)
        if removed:
            shapes.parse(_PATCH_RESOURCE, format="turtle")
            patches.append(
                {
                    "shape": iri,
                    "removed_triples": removed,
                    "reason": reason,
                    "patch_resource": str(_PATCH_RESOURCE.name),
                    "patch_sha256": sha256_file(_PATCH_RESOURCE),
                }
            )
    return shapes, patches


def release_data_files(config: ProjectConfig, shard_limit: int | None = None) -> list[Path]:
    """Ordered corpus of the governed release (excluding the derived summary)."""

    files: list[Path] = []
    for relative_dir in _RELEASE_DATA_DIRS:
        directory = config.ckg_root / relative_dir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.ttl")):
            if path.name == Path(_RELEASE_SUMMARY).name:
                continue
            files.append(path)
    if shard_limit is not None:
        shards = [p for p in files if "Station01-CKG-PWP-" in p.name]
        shard_names = {p.name for p in shards[:shard_limit]}
        files = [p for p in files if "Station01-CKG-PWP-" not in p.name or p.name in shard_names]
    return files


def build_release_graph(
    config: ProjectConfig,
    *,
    include_schema: bool = True,
    shard_limit: int | None = None,
) -> tuple[Graph, list[str]]:
    graph = Graph()
    sources: list[str] = []
    for path in release_data_files(config, shard_limit=shard_limit):
        graph.parse(path, format="turtle")
        sources.append(str(path))
    if include_schema:
        for relative in _RELEASE_SCHEMA_FILES:
            path = config.ckg_root / relative
            if path.is_file():
                graph.parse(path, format="turtle")
                sources.append(str(path))
    return graph, sources


def _violation_groups(results_graph: Graph) -> dict[str, int]:
    query = f"""PREFIX sh: <{_SHACL}>
SELECT ?shape ?path ?sev (COUNT(?v) AS ?n) WHERE {{
  ?v a sh:ValidationResult ; sh:resultSeverity ?sev .
  OPTIONAL {{ ?v sh:sourceShape ?shape }}
  OPTIONAL {{ ?v sh:resultPath ?path }}
}} GROUP BY ?shape ?path ?sev"""

    def _local(value: Any) -> str:
        text = str(value) if value else "-"
        return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    groups: dict[str, int] = {}
    for row in results_graph.query(query):
        key = f"{_local(row.sev)}|{_local(row.shape)}|{_local(row.path)}"
        groups[key] = groups.get(key, 0) + int(row.n.toPython())
    return groups


def run_shacl(data_graph: Graph, shapes_graph: Graph) -> dict[str, Any]:
    """Execute pyshacl once and summarise the result graph."""

    from pyshacl import validate  # deferred: heavy import, optional dependency

    started = time.perf_counter()
    conforms, results, _text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        advanced=False,
        meta_shacl=False,
    )
    duration = round(time.perf_counter() - started, 1)
    if type(results).__name__ == "ValidationFailure":
        return {
            "engine": "pyshacl",
            "execution": "failed",
            "conforms": False,
            "error": str(results),
            "duration_seconds": duration,
        }
    groups = _violation_groups(results)
    violations = sum(count for key, count in groups.items() if key.startswith("Violation|"))
    warnings = sum(count for key, count in groups.items() if key.startswith("Warning|"))
    return {
        "engine": "pyshacl",
        "execution": "complete",
        "conforms": bool(conforms),
        "violation_count": violations,
        "warning_count": warnings,
        "violation_groups": dict(sorted(groups.items(), key=lambda item: -item[1])),
        "duration_seconds": duration,
    }


def validate_release(config: ProjectConfig, shard_limit: int | None = None) -> dict[str, Any]:
    """Full SHACL conformance report for the governed CKG release."""

    shapes, patches = load_shapes(config)
    data, sources = build_release_graph(config, shard_limit=shard_limit)
    result = run_shacl(data, shapes)
    return {
        "release_root": str(config.ckg_root),
        "release_manifest_sha256": sha256_file(config.ckg_manifest),
        "shapes_files": [str(config.ckg_root / relative) for relative in _SHAPE_FILES],
        "shapes_triples": len(shapes),
        "shape_patches": patches,
        "data_triples": len(data),
        "data_sources": [str(path) for path in sources],
        "data_source_count": len(sources),
        "shard_limit": shard_limit,
        **result,
    }


def register_path(config: ProjectConfig) -> Path:
    return config.cache_dir / "shacl_known_violations.json"


def establish_register(config: ProjectConfig, report: dict[str, Any]) -> Path:
    payload = {
        "register_version": 1,
        "established_from": {
            "release_manifest_sha256": report.get("release_manifest_sha256"),
            "shard_limit": report.get("shard_limit"),
            "violation_groups": report.get("violation_groups", {}),
        },
    }
    path = register_path(config)
    atomic_write_json(path, payload)
    return path


def register_status(config: ProjectConfig, report: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed regression check of a report against the register."""

    path = register_path(config)
    if not path.exists():
        return {
            "ok": False,
            "reason": "no known-violation register; run validate-ckg --establish-register first",
        }
    register = json.loads(path.read_text(encoding="utf-8"))
    known: dict[str, int] = register["established_from"]["violation_groups"]
    current: dict[str, int] = report.get("violation_groups", {})
    new_groups = sorted(set(current) - set(known))
    increased = sorted(
        key for key in set(current) & set(known) if current[key] > known[key]
    )
    resolved = sorted(key for key in set(known) - set(current))
    decreased = sorted(
        key for key in set(known) & set(current) if current[key] < known[key]
    )
    return {
        "ok": not new_groups and not increased,
        "new_violation_groups": new_groups,
        "increased_groups": increased,
        "resolved_groups": resolved,
        "decreased_groups": decreased,
    }


def validate_delta(delta_graph: Graph, config: ProjectConfig) -> dict[str, Any]:
    """Check a correction graph without the release's violation register."""

    shapes, _patches = load_shapes(config)
    result = run_shacl(delta_graph, shapes)
    result["policy"] = "zero_tolerance"
    if result.get("execution") == "complete":
        result["ok"] = result["conforms"]
    else:
        result["ok"] = False
    return result


_PATCH_VOCABULARY_CACHE: dict[str, Graph] = {}


def _patch_vocabulary() -> Graph:
    """Read the vocabulary declarations in the shape overrides."""

    graph = _PATCH_VOCABULARY_CACHE.get("graph")
    if graph is None:
        graph = Graph()
        graph.parse(_PATCH_RESOURCE, format="turtle")
        _PATCH_VOCABULARY_CACHE["graph"] = graph
    return graph


def validate_delta_in_context(
    config: ProjectConfig,
    target_uri: str,
    delta_triples: list[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Contextual zero-tolerance gate for a KC correction preview.

    A transaction delta carries no rdf:type, so validating it in isolation
    would target nothing.  The governed check instead validates the target
    node's post-state (existing release assertions about the target plus the
    delta) and fails closed only on violations the delta itself introduces,
    measured against the same node's pre-state.
    """

    from rdflib import RDF

    shapes, patches = load_shapes(config)
    release = load_release_graph_cached(config)
    target = URIRef(target_uri)

    pre = Graph()
    for triple in release.triples((target, None, None)):
        pre.add(triple)
    # Use the same pre-existing type declarations in both validation graphs.
    referenced: set[Any] = set()
    for _s, _p, obj in delta_triples:
        if isinstance(obj, URIRef):
            referenced.add(obj)
    for obj in referenced:
        for type_triple in release.triples((obj, RDF.type, None)):
            pre.add(type_triple)
        if not any(pre.triples((obj, RDF.type, None))):
            # Use the unit declarations from the shape patch in both graphs.
            for type_triple in _patch_vocabulary().triples((obj, RDF.type, None)):
                pre.add(type_triple)
    post = Graph()
    for triple in pre:
        post.add(triple)
    for triple in delta_triples:
        post.add(triple)
    if not any(triple[0] == target and str(triple[1]).endswith("#type") for triple in post):
        post.add((target, RDF.type, URIRef("http://bimlca.uni.edu/ontology#UserCorrection")))

    pre_result = run_shacl(pre, shapes)
    post_result = run_shacl(post, shapes)
    if pre_result["execution"] != "complete" or post_result["execution"] != "complete":
        return {
            "policy": "contextual_zero_tolerance",
            "ok": False,
            "error": "SHACL engine did not complete for the transaction subgraphs",
            "pre": pre_result,
            "post": post_result,
        }

    introduced = sorted(
        set(post_result.get("violation_groups", {})) - set(pre_result.get("violation_groups", {}))
    )
    return {
        "policy": "contextual_zero_tolerance",
        "ok": not introduced,
        "target_uri": target_uri,
        "pre_violation_count": pre_result.get("violation_count", 0),
        "post_violation_count": post_result.get("violation_count", 0),
        "introduced_violation_groups": introduced,
        "shape_patches": [patch["shape"] for patch in patches],
    }


def release_digest(config: ProjectConfig) -> str:
    return stable_digest(
        {
            "manifest": sha256_file(config.ckg_manifest),
            "shapes": [sha256_file(config.ckg_root / relative) for relative in _SHAPE_FILES],
            "patch": sha256_file(_PATCH_RESOURCE),
        }
    )




def swrl_inference_path(config: ProjectConfig) -> Path:
    """Materialised inferences live OUTSIDE knowledge.sqlite3: its SHA-256 is
    pinned by every ground-truth suite's source snapshot and must not change."""

    return config.cache_dir / "swrl_inferences.sqlite3"


def swrl_materialize(
    config: ProjectConfig,
    *,
    include_shards: bool = True,
    shard_limit: int | None = None,
) -> dict[str, Any]:
    """Run the governed SWRL rules over the release graph and materialise.

    Inferred triples are stored with their producing rule as provenance in a
    dedicated SQLite file that the Reader can query as a governed source.
    """

    import sqlite3

    from .swrl_engine import IndexedSWRLForwardChainer

    rules = Graph()
    rules.parse(config.swrl, format="turtle")

    if include_shards:
        graph, sources = build_release_graph(config, shard_limit=shard_limit)
    else:
        graph, sources = _build_release_graph_without_shards(config)

    started = time.perf_counter()
    engine = IndexedSWRLForwardChainer(graph, rules)
    stats = engine.run()
    duration = round(time.perf_counter() - started, 1)

    target = swrl_inference_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    try:
        connection.execute("DROP TABLE IF EXISTS inferences")
        connection.execute("DROP TABLE IF EXISTS meta")
        connection.execute(
            """CREATE TABLE inferences (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                rule TEXT NOT NULL,
                object_is_literal INTEGER NOT NULL DEFAULT 0
            )"""
        )
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        rows = [
            (
                str(inference.triple[0]),
                str(inference.triple[1]),
                str(inference.triple[2]),
                inference.rule,
                0 if isinstance(inference.triple[2], URIRef) or isinstance(inference.triple[2], BNode) else 1,
            )
            for inference in engine.inferences
        ]
        connection.executemany("INSERT INTO inferences VALUES (?, ?, ?, ?, ?)", rows)
        connection.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [
                ("rule_source", str(config.swrl)),
                ("rule_source_sha256", sha256_file(Path(config.swrl))),
                ("engine", "msca_mas.swrl_engine.IndexedSWRLForwardChainer"),
                ("new_triples", str(stats["new_triples"])),
                ("rule_firings", str(stats["rule_firings"])),
                ("iterations", str(stats["iterations"])),
                ("data_source_count", str(len(sources))),
                ("include_shards", "1" if include_shards else "0"),
                ("shard_limit", str(shard_limit) if shard_limit else ""),
                ("duration_seconds", str(duration)),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    return {
        "rule_source": str(config.swrl),
        "rule_source_sha256": sha256_file(Path(config.swrl)),
        "rule_count": len(engine.rules),
        "data_triples": len(graph),
        "inference_count": len(engine.inferences),
        "statistics": stats,
        "duration_seconds": duration,
        "output": str(target),
        "data_sources": sources,
    }


def _build_release_graph_without_shards(config: ProjectConfig) -> tuple[Graph, list[str]]:
    graph = Graph()
    sources: list[str] = []
    for relative_dir in _RELEASE_DATA_DIRS:
        if relative_dir.endswith("/ckg"):
            continue
        directory = config.ckg_root / relative_dir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.ttl")):
            graph.parse(path, format="turtle")
            sources.append(str(path))
    for relative in _RELEASE_SCHEMA_FILES:
        path = config.ckg_root / relative
        if path.is_file():
            graph.parse(path, format="turtle")
            sources.append(str(path))
    return graph, sources



_SPARQL_TEMPLATES = {
    "CKG-PROJECT-OVERVIEW": "CQ-C04.rq",
    "CKG-TOP-ELEMENTS": "CQ-C09.rq",
    "CKG-RESOURCE-BREAKDOWN": "CQ-P04.rq",
    "CKG-WORK-PACKAGES": "CQ-P03.rq",
    "CKG-ELEMENT-TRACE": "CQ-C02.rq",
}

_TEMPLATE_DIR = Path(__file__).parent / "resources" / "sparql_templates"

_RELEASE_GRAPH_CACHE: dict[str, Graph] = {}


def load_release_graph_cached(config: ProjectConfig) -> Graph:
    """Parse the release graph once per process (read-only, memory cached)."""

    digest = release_digest(config)
    graph = _RELEASE_GRAPH_CACHE.get(digest)
    if graph is None:
        graph, _ = build_release_graph(config)
        _RELEASE_GRAPH_CACHE.clear()  # one release per process
        _RELEASE_GRAPH_CACHE[digest] = graph
    return graph


def _sparql_cache_path(config: ProjectConfig) -> Path:
    return config.cache_dir / "sparql_results_cache.sqlite3"


def _local_name(value: Any) -> str:
    text = str(value)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rsplit("/", 1)[-1]


def execute_template_sparql(
    config: ProjectConfig,
    template_id: str,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a SPARQL template, caching by release, template and bindings."""

    import sqlite3

    from rdflib import Literal, URIRef

    filename = _SPARQL_TEMPLATES.get(template_id)
    if filename is None:
        raise ValueError(f"unknown SPARQL template: {template_id}")
    # Include the row limit in the cache key as well as the other query bindings.
    cache_bindings = dict(bindings or {})
    bindings = dict(bindings or {})
    limit = bindings.pop("limit", None)
    query_path = _TEMPLATE_DIR / filename
    query_text = query_path.read_text(encoding="utf-8")
    # Exclude comments when checking whether the query can change the graph.
    query_body = "\n".join(
        line for line in query_text.splitlines() if not line.lstrip().startswith("#")
    )
    if not re.search(r"\b(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\b", query_body, re.I):
        raise ValueError(f"template {filename} is not a read query")
    if re.search(r"\b(?:INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|MOVE|COPY|ADD)\b", query_body, re.I):
        raise ValueError(f"template {filename} contains a mutating SPARQL operation")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit binding must be a positive integer")
        if not re.search(r"\bLIMIT\s+\d+\b", query_body):
            raise ValueError(f"template {filename} has no integer LIMIT to substitute")
        query_text = re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {limit}", query_text, count=1)

    init_bindings: dict[str, Any] = {}
    # RDFLib can miss initBindings in grouped filters; use VALUES instead.
    values_clauses: list[str] = []

    def _escape_literal(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    if "term" in bindings:
        values_clauses.append(f'VALUES ?term {{ "{_escape_literal(bindings["term"])}" }}')
    if bindings.get("stage"):
        values_clauses.append(f'VALUES ?stageFilter {{ "{_escape_literal(bindings["stage"])}" }}')
    if bindings.get("element_uri"):
        element_iri = str(bindings["element_uri"])
        if re.search(r"[\s<>\"{}|^`]", element_iri):
            raise ValueError("element_uri binding contains unsafe IRI characters")
        values_clauses.append(f"VALUES ?element {{ <{element_iri}> }}")
    if values_clauses:
        # Bind inside WHERE so the values are available to its filters.
        where_open = re.search(r"WHERE\s*\{", query_text)
        if where_open is None:
            raise ValueError(f"template {filename} has no WHERE group for bindings")
        insert_at = where_open.end()
        query_text = query_text[:insert_at] + "\n" + "\n".join(values_clauses) + "\n" + query_text[insert_at:]

    cache_key = stable_digest(
        [release_digest(config), template_id, sha256_file(query_path), cache_bindings]
    )
    cache = _sparql_cache_path(config)
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS sparql_results (key TEXT PRIMARY KEY, template_id TEXT NOT NULL, rows_json TEXT NOT NULL, duration_seconds REAL NOT NULL, created_epoch REAL NOT NULL)"
        )
        connection.commit()
        row = connection.execute(
            "SELECT rows_json, duration_seconds FROM sparql_results WHERE key=?", (cache_key,)
        ).fetchone()
    finally:
        connection.close()
    if row is not None:
        return {
            "template_id": template_id,
            "sparql_file": str(query_path),
            "sparql_sha256": sha256_file(query_path),
            "bindings": bindings,
            "rows": json.loads(row[0]),
            "duration_seconds": row[1],
            "cache": "disk_hit",
        }

    graph = load_release_graph_cached(config)
    started = time.perf_counter()
    result = graph.query(query_text)
    rows = [
        {str(var): (str(value) if value is not None else None) for var, value in zip(result.vars, row)}
        for row in result
    ]
    duration = round(time.perf_counter() - started, 3)
    connection = sqlite3.connect(cache)
    try:
        connection.execute(
            "INSERT OR REPLACE INTO sparql_results VALUES (?, ?, ?, ?, ?)",
            (cache_key, template_id, json.dumps(rows), duration, time.time()),
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "template_id": template_id,
        "sparql_file": str(query_path),
        "sparql_sha256": sha256_file(query_path),
        "bindings": bindings,
        "rows": rows,
        "duration_seconds": duration,
        "cache": "miss",
    }


def _close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tolerance


def _stage_sums_from_index(config: ProjectConfig) -> dict[str, float]:
    import sqlite3

    connection = sqlite3.connect(config.cache_dir / "knowledge.sqlite3")
    try:
        rows = connection.execute(
            "SELECT stage_uri, SUM(value) FROM emissions GROUP BY stage_uri"
        ).fetchall()
    finally:
        connection.close()
    return {_local_name(stage): float(total or 0.0) for stage, total in rows}


def compare_template_parity(
    config: ProjectConfig,
    knowledge: Any,
    template_id: str,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assert the SPARQL template and the materialised index answer alike."""

    bindings = dict(bindings or {})
    sparql = execute_template_sparql(config, template_id, bindings)
    rows = sparql["rows"]

    def verdict(ok: bool, detail: dict[str, Any]) -> dict[str, Any]:
        return {
            "template_id": template_id,
            "bindings": bindings,
            "match": ok,
            "compared_rows": detail.get("compared_rows", 0),
            "detail": detail,
        }

    if template_id == "CKG-PROJECT-OVERVIEW":
        expected = _stage_sums_from_index(config)
        actual = {_local_name(row.get("stage")): float(row.get("stageCarbon") or 0.0) for row in rows}
        ok = set(expected) == set(actual) and all(_close(actual[stage], expected[stage]) for stage in expected)
        return verdict(ok, {"index_stage_sums": expected, "sparql_stage_sums": actual, "compared_rows": len(expected)})

    if template_id == "CKG-TOP-ELEMENTS":
        expected = knowledge.top_elements(int(bindings.get("limit", 10)), bindings.get("stage"))
        expected_pairs = [(item["element_uri"], float(item["value"] or 0.0)) for item in expected]
        actual_pairs = [
            (row.get("element"), float(row.get("carbonValue") or 0.0))
            for row in rows
        ]
        ok = len(expected_pairs) == len(actual_pairs) and all(
            str(exp[0]) == str(act[0]) and _close(exp[1], act[1])
            for exp, act in zip(expected_pairs, actual_pairs)
        )
        return verdict(ok, {"compared_rows": len(expected_pairs), "index_top": expected_pairs[:3], "sparql_top": actual_pairs[:3]})

    if template_id == "CKG-RESOURCE-BREAKDOWN":
        expected = knowledge.resource_breakdown(str(bindings.get("term", "")), bindings.get("stage"), int(bindings.get("limit", 30)))
        expected_map = {
            (item["descriptor_uri"], item["stage"]): (float(item["emission_value"] or 0.0), int(item["resource_count"]))
            for item in expected["items"]
        }
        actual_map = {
            (row.get("descriptor"), _local_name(row.get("stage"))): (float(row.get("emissionValue") or 0.0), int(row.get("resourceCount") or 0))
            for row in rows
        }
        ok = len(expected_map) == len(actual_map) and all(
            key in actual_map
            and _close(expected_map[key][0], actual_map[key][0])
            and expected_map[key][1] == actual_map[key][1]
            for key in expected_map
        )
        return verdict(ok, {"compared_rows": len(expected_map), "index_rows": len(expected["items"]), "sparql_rows": len(rows)})

    if template_id == "CKG-WORK-PACKAGES":
        expected = knowledge.work_package_breakdown(str(bindings.get("term", "")), int(bindings.get("limit", 30)))
        expected_map = {
            item["template_uri"]: (int(item["work_package_count"]), float(item["quantity_value"] or 0.0))
            for item in expected
        }
        actual_map = {
            row.get("template"): (int(row.get("workPackageCount") or 0), float(row.get("quantityValue") or 0.0))
            for row in rows
        }
        ok = len(expected_map) == len(actual_map) and all(
            key in actual_map
            and expected_map[key][0] == actual_map[key][0]
            and _close(expected_map[key][1], actual_map[key][1])
            for key in expected_map
        )
        return verdict(ok, {"compared_rows": len(expected_map), "index_rows": len(expected), "sparql_rows": len(rows)})

    if template_id == "CKG-ELEMENT-TRACE":
        element_uri = str(bindings.get("element_uri", ""))
        expected = knowledge.element_trace(element_uri, int(bindings.get("limit", 250)))
        # Exclude index rows that have no resource link from path parity checks.
        expected_keys = {
            (row["work_package_uri"], row["resource_uri"], _local_name(row["stage_uri"]), round(float(row["emission_value"] or 0.0), 4))
            for row in expected["chain_rows"]
            if row["resource_uri"] is not None
        }
        actual_keys = {
            (row.get("workPackage"), row.get("resource"), _local_name(row.get("stage")), round(float(row.get("emissionValue") or 0.0), 4))
            for row in rows
        }
        ok = expected_keys == actual_keys
        return verdict(ok, {"compared_rows": len(expected_keys), "index_rows": len(expected["chain_rows"]), "sparql_rows": len(rows)})

    raise ValueError(f"no parity comparator for template {template_id}")


def assert_template_parity(
    config: ProjectConfig,
    knowledge: Any,
    template_id: str,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runtime gate used by the Reader tool; raises on any divergence."""

    report = compare_template_parity(config, knowledge, template_id, bindings)
    if not report["match"]:
        raise ValueError(
            "SPARQL_INDEX_PARITY_MISMATCH: "
            f"template {template_id} diverged from the materialised index; "
            f"detail={json.dumps(report['detail'], ensure_ascii=False)[:400]}"
        )
    return report
