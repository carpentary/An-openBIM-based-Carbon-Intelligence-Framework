from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace, OWL, RDF

from .config import ProjectConfig
from .cq_validation import run_cq_checks


SWRL = Namespace("http://www.w3.org/2003/11/swrl#")
SH = Namespace("http://www.w3.org/ns/shacl#")
MSCA = Namespace("http://bimlca.uni.edu/ontology#")


def _parse(path: Path) -> tuple[Graph, dict[str, Any]]:
    graph = Graph()
    try:
        graph.parse(path, format="turtle")
        return graph, {"path": str(path), "status": "PASS", "triples": len(graph)}
    except Exception as exc:
        return graph, {"path": str(path), "status": "FAIL", "triples": 0, "error": str(exc)}


def _check_swrl_inference(config: ProjectConfig) -> dict[str, Any]:
    specification = importlib.util.spec_from_file_location("msca_governed_swrl_chainer", config.swrl_chainer)
    if specification is None or specification.loader is None:
        return {"status": "FAIL", "error": f"cannot load {config.swrl_chainer}"}
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(specification.name, None)

    test = Namespace("http://bimlca.uni.edu/test-swrl#")
    data = Graph()
    for path in (config.tbox, config.catalogue, config.swrl_regression_input):
        data.parse(path, format="turtle")
    rules = Graph().parse(config.swrl, format="turtle")
    statistics = module.RestrictedSWRLForwardChainer(data, rules).run()
    expected = (
        (test.BottomSlabCandidate, MSCA.hasElementType, MSCA.Type_BottomSlab),
        (test.BottomSlabCandidate, MSCA.requiresWorkPackageTemplate, MSCA.WPT_SLAB_BOTTOM),
        (test.SideWallCandidate, MSCA.hasElementType, MSCA.Type_SideWall),
        (test.SideWallCandidate, MSCA.requiresWorkPackageTemplate, MSCA.WPT_WALL_STRAIGHT),
        (test.SteelBraceCandidate, MSCA.hasConstructionRole, MSCA.Role_TemporaryWork),
        (test.ExcavationZoneCandidate, MSCA.hasConstructionRole, MSCA.Role_ImplicitSpatialWork),
        (test.ExcavationZoneCandidate, MSCA.hasRepresentationMode, MSCA.Representation_DerivedGeometry),
        (test.ProjectWP, MSCA.hasProcessFamily, MSCA.PF_Waterproofing),
    )
    missing = [tuple(map(str, triple)) for triple in expected if triple not in data]
    return {
        "status": "PASS" if not missing else "FAIL",
        "expected_entailments": len(expected),
        "found_entailments": len(expected) - len(missing),
        "missing": missing,
        "engine_statistics": statistics,
        "output_written_to_source_release": False,
    }


def _meta_validate_shapes(config: ProjectConfig) -> dict[str, Any]:
    try:
        from pyshacl import validate
    except ImportError:
        return {"status": "SKIPPED", "reason": "pySHACL is not installed"}
    shapes = Graph().parse(config.shacl, format="turtle")
    conforms, report_graph, report_text = validate(
        data_graph=Graph(),
        shacl_graph=shapes,
        advanced=True,
        meta_shacl=True,
        allow_infos=True,
        allow_warnings=True,
    )
    return {
        "status": "PASS" if conforms else "FAIL",
        "conforms": bool(conforms),
        "report_triples": len(report_graph),
        "purpose": "SHACL profile meta-validation only; empty data graph does not establish production CKG conformance.",
        "summary": str(report_text)[:2000],
    }


def run_validation(config: ProjectConfig) -> dict[str, Any]:
    parse_targets = {
        "tbox": config.tbox,
        "complete_schema": config.complete_schema,
        "catalogue": config.catalogue,
        "swrl": config.swrl,
        "shacl": config.shacl,
        "cpd_reference": config.cpd_reference,
        "efd_reference": config.efd_reference,
        "ckg_domain_abox": config.ckg_domain_abox,
        "ckg_summary": config.ckg_summary,
        "topology_facts": config.topology_facts,
    }
    graphs: dict[str, Graph] = {}
    parse_results: dict[str, dict[str, Any]] = {}
    for name, path in parse_targets.items():
        graphs[name], parse_results[name] = _parse(path)

    generated_abox = config.artifact_dir / "abox" / "station01_enclosure_input_abox.ttl"
    if generated_abox.exists():
        _, parse_results["generated_enclosure_abox"] = _parse(generated_abox)
    reference_abox = config.artifact_dir / "reference_case" / "station01_reference_case_abox.ttl"
    if reference_abox.exists():
        _, parse_results["generated_reference_case_abox"] = _parse(reference_abox)

    complete = graphs["complete_schema"]
    structural_counts = {
        "owl_classes": len(set(complete.subjects(RDF.type, OWL.Class))),
        "object_properties": len(set(complete.subjects(RDF.type, OWL.ObjectProperty))),
        "datatype_properties": len(set(complete.subjects(RDF.type, OWL.DatatypeProperty))),
        "swrl_rules": len(set(complete.subjects(RDF.type, SWRL.Imp))),
        "shacl_node_shapes": len(set(complete.subjects(RDF.type, SH.NodeShape))),
        "competency_questions": len(set(complete.subjects(RDF.type, MSCA.CompetencyQuestion))),
    }
    cq_report = run_cq_checks(config)
    swrl_report = _check_swrl_inference(config)
    shacl_report = _meta_validate_shapes(config)
    bundled_report = json.loads(config.ontology_release_report.read_text(encoding="utf-8"))
    critical_pass = (
        all(item["status"] == "PASS" for item in parse_results.values())
        and cq_report["status"] == "PASS"
        and swrl_report["status"] == "PASS"
        and shacl_report["status"] == "PASS"
    )
    return {
        "status": "PASS" if critical_pass else "FAIL",
        "project_id": config.project_id,
        "rdf_parse": parse_results,
        "structural_counts": structural_counts,
        "cq_checks": cq_report,
        "swrl_checks": swrl_report,
        "shacl_profile_meta_validation": shacl_report,
        "bundled_release_verification": {
            "path": str(config.ontology_release_report),
            "status": bundled_report.get("status"),
            "counts": bundled_report.get("counts", {}),
            "warnings": bundled_report.get("warnings", []),
        },
        "data_conformity_boundary": (
            "This report covers ontology syntax, CQ and SWRL checks, and SHACL shape validation. "
            "Use validate-ckg to check the project graph."
        ),
    }
