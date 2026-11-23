"""Command-line interface for the MAS."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from .abox import build_enclosure_abox
from .agents import AgentServices
from .config import ProjectConfig
from .cq import CQCatalogue
from .cq_validation import run_cq_checks
from .evaluation import score_predictions
from .evaluation_runner import load_query_suite, run_query_suite
from .ifc import build_inventory, write_inventory
from .ifc_properties import extract_properties, write_property_report
from .knowledge import KnowledgeIndex
from .models import AgentName, QueryMode
from .reasoning import DeepSeekPlanningReasoner, planning_reasoner_from_env
from .reference_case import build_reference_case
from .runtime import invoke_query
from .transactions import TransactionPreview, TransactionService
from .util import atomic_write_json, portable_path
from .validation import run_validation


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _config(arguments: argparse.Namespace) -> ProjectConfig:
    return ProjectConfig.load(arguments.config)


def command_doctor(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    report = config.doctor()
    packages = {}
    for package in ("msca-mas", "langgraph", "rdflib", "pydantic", "pyshacl"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    report["packages"] = packages
    report["python"] = sys.version
    report["knowledge_index"] = KnowledgeIndex(config).stats()
    print(_json(report))
    return 0 if report["ok"] else 1


def _topology_unresolved_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def command_prepare(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    doctor = config.doctor()
    if not doctor["ok"]:
        print(_json(doctor))
        return 1
    artifact_dir = config.artifact_dir
    inventory = build_inventory(config.ifc_sources)
    write_inventory(inventory, artifact_dir / "ifc")
    abox_report = build_enclosure_abox(inventory, artifact_dir / "abox")

    property_reports: list[dict[str, Any]] = []
    if not arguments.skip_properties:
        roles = {"enclosure_structure"}
        if arguments.full_properties:
            roles.add("main_structure")
        for source in config.ifc_sources:
            if source.role not in roles:
                continue
            ids = [product.step_id for product in inventory.products if product.source_role == source.role]
            report = extract_properties(source.path, source.role, ids)
            write_property_report(report, artifact_dir / "ifc")
            property_reports.append(report.compact())

    knowledge = KnowledgeIndex(config)
    index_report = knowledge.build(force=arguments.force_index)
    reference_case_report: dict[str, Any] | None = None
    if not arguments.skip_properties:
        reference_case_report = build_reference_case(config, knowledge=knowledge)
    representative = knowledge.top_elements(limit=1, stage="A1-A3")
    representative_trace: dict[str, Any] = {}
    if representative:
        representative_trace = knowledge.element_trace(representative[0]["element_uri"])
        representative_trace["ranked_element_summary"] = representative[0]
        atomic_write_json(artifact_dir / "representative_seprc_trace.json", representative_trace)
    catalogue = CQCatalogue.load(config.cq_catalogue)
    report = {
        "project_id": config.project_id,
        "status": "READY_REFERENCE_CASE" if reference_case_report else "READY_WITH_REFERENCE_CASE_SKIPPED",
        "stages": [
            {
                "stage": "raw_ifc_input",
                "status": "READY",
                "evidence": inventory.compact(),
            },
            {
                "stage": "ifc_qto_extraction",
                "status": "READY" if property_reports else "SKIPPED",
                "evidence": property_reports,
            },
            {
                "stage": "main_structure_semantic_baseline",
                "status": "READY",
                "evidence": index_report,
            },
            {
                "stage": "representative_s_e_p_r_c_trace",
                "status": "READY" if representative_trace else "BLOCKED_NO_TRACEABLE_ELEMENT",
                "evidence": {
                    "artifact": str(artifact_dir / "representative_seprc_trace.json"),
                    "element_uri": representative_trace.get("element_uri"),
                    "work_packages": representative_trace.get("distinct_work_packages", 0),
                    "resources": representative_trace.get("distinct_resources", 0),
                    "emissions": representative_trace.get("distinct_emissions", 0),
                },
            },
            {
                "stage": "enclosure_input_abox",
                "status": "READY",
                "evidence": abox_report.model_dump(mode="json"),
            },
            {
                "stage": "reference_case_calculation",
                "status": "READY" if reference_case_report else "SKIPPED_WITH_PROPERTY_EXTRACTION",
                "evidence": {
                    "summary": str(artifact_dir / "reference_case" / "reference_case_summary.json"),
                    "combined_stage_totals_kgco2e": (
                        reference_case_report.get("combined_stage_totals_kgco2e", {})
                        if reference_case_report
                        else {}
                    ),
                },
            },
            {
                "stage": "topology",
                "status": "READY_FROM_GOVERNED_RELEASE",
                "evidence": {
                    "facts": str(config.topology_facts),
                    "unresolved_rows": _topology_unresolved_count(config.unresolved_topology_register),
                    "runtime_parameter_profile": {
                        "coordinate_unit": config.topology_coordinate_unit,
                        "connection_epsilon": config.topology_connection_epsilon,
                        "containment_epsilon": config.topology_containment_epsilon,
                        "basis": "controlled_configuration",
                    },
                },
            },
            {
                "stage": "competency_questions",
                "status": "READY",
                "evidence": {"count": len(catalogue.entries), "groups": catalogue.group_counts()},
            },
            {
                "stage": "mas_runtime",
                "status": "READY",
                "evidence": {
                    "agents": ["orchestrator", "metadata", "extractor", "reader", "calculator", "synthesis"],
                    "modes": ["IR", "KC", "SA"],
                },
            },
        ],
        "important_boundary": (
            "The combined reference-case result joins the released main-structure inventory with IFC-derived "
            "enclosure quantities, governed CPD/EFD mappings, material-default A4 transport and the declared "
            "foundation-pit reference profile. Its applicability scope is methodological demonstration."
        ),
    }
    atomic_write_json(artifact_dir / "prepare_report.json", report)
    print(_json(report))
    return 0


def command_case_build(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    doctor = config.doctor()
    if not doctor["ok"]:
        print(_json(doctor))
        return 1
    inventory = build_inventory(config.ifc_sources)
    write_inventory(inventory, config.artifact_dir / "ifc")
    build_enclosure_abox(inventory, config.artifact_dir / "abox")
    enclosure_source = next(source for source in config.ifc_sources if source.role == "enclosure_structure")
    enclosure_ids = [
        product.step_id for product in inventory.products if product.source_role == "enclosure_structure"
    ]
    property_report = extract_properties(enclosure_source.path, enclosure_source.role, enclosure_ids)
    write_property_report(property_report, config.artifact_dir / "ifc")
    knowledge = KnowledgeIndex(config)
    knowledge.build(force=arguments.force_index)
    report = build_reference_case(config, knowledge=knowledge)
    print(
        _json(
            {
                "status": report["calculation_status"],
                "combined_stage_totals_kgco2e": report["combined_stage_totals_kgco2e"],
                "combined_total_kgco2e": report["combined_total_kgco2e"],
                "summary": report["artifacts"]["summary"],
            }
        )
    )
    return 0


def command_ask(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    knowledge = KnowledgeIndex(config)
    if not knowledge.is_current():
        if arguments.no_auto_index:
            print("Knowledge index is absent or stale. Run `msca-mas prepare`.", file=sys.stderr)
            return 2
        knowledge.build()
    planning_reasoner = planning_reasoner_from_env(
        max_retries=config.structured_output_max_retries
    )
    services = AgentServices.create(
        config,
        knowledge=knowledge,
        planning_reasoner=planning_reasoner,
    )
    state = invoke_query(arguments.query, config=config, services=services)
    report = state["final_report"]
    if arguments.json:
        print(_json(report))
    else:
        print(report["answer"])
        print(f"status={report['status']} mode={report['mode']}")
        if report.get("issues"):
            for issue in report["issues"]:
                print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")
        if report.get("transaction_preview"):
            preview_path = services.transactions.save_preview(
                TransactionPreview.model_validate(report["transaction_preview"])
            )
            print(f"transaction_preview={preview_path}")
            print(f"confirmation_digest={report['transaction_preview']['confirmation_digest']}")
    return 0 if report["status"] != "blocked" else 3


def command_llm_check(arguments: argparse.Namespace) -> int:
    """Run one bounded structured planning call without exposing credentials."""

    config = _config(arguments)
    try:
        reasoner = planning_reasoner_from_env(
            max_retries=config.structured_output_max_retries
        )
        if reasoner is None:
            print(
                _json(
                    {
                        "status": "DISABLED",
                        "message": "Set MSCA_LLM_PROVIDER=deepseek and DEEPSEEK_API_KEY in the environment.",
                    }
                )
            )
            return 2
        decision = reasoner.decide(
            "Report the governed total project carbon.",
            QueryMode.INFORMATION_RETRIEVAL,
        )
    except Exception as exc:
        print(_json({"status": "FAIL", "error_type": type(exc).__name__, "message": str(exc)}))
        return 1
    summary = (
        reasoner.configuration_summary()
        if isinstance(reasoner, DeepSeekPlanningReasoner)
        else {"provider": type(reasoner).__name__}
    )
    print(
        _json(
            {
                "status": "PASS",
                "configuration": summary,
                "decision": decision.model_dump(mode="json"),
            }
        )
    )
    return 0


def command_cq_check(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    report = run_cq_checks(config, cq_ids=arguments.cq_id or None)
    output = config.artifact_dir / "cq_validation_report.json"
    atomic_write_json(output, report)
    print(_json({"status": report["status"], "passed": report["passed"], "total": report["total"], "output": str(output)}))
    return 0 if report["status"] == "PASS" else 1


def command_validate(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    report = run_validation(config)
    output = config.artifact_dir / "validation_report.json"
    atomic_write_json(output, report)
    print(
        _json(
            {
                "status": report["status"],
                "structural_counts": report["structural_counts"],
                "cq": {
                    "passed": report["cq_checks"]["passed"],
                    "total": report["cq_checks"]["total"],
                },
                "swrl": report["swrl_checks"],
                "shacl_profile": report["shacl_profile_meta_validation"]["status"],
                "output": str(output),
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


def command_validate_ckg(arguments: argparse.Namespace) -> int:
    from .ontology_runtime import (
        establish_register,
        register_path,
        register_status,
        release_digest,
        validate_release,
    )

    config = _config(arguments)
    report = validate_release(config, shard_limit=arguments.shard_limit)
    report["release_digest"] = release_digest(config)
    output = config.cache_dir / "shacl_conformance_report.json"
    atomic_write_json(output, report)
    if arguments.establish_register:
        register_file = establish_register(config, report)
        summary = {
            "established_register": str(register_file),
            "execution": report["execution"],
            "conforms": report.get("conforms"),
            "violation_count": report.get("violation_count"),
            "violation_group_count": len(report.get("violation_groups", {})),
            "report": str(output),
        }
    else:
        status = register_status(config, report)
        summary = {
            "execution": report["execution"],
            "conforms": report.get("conforms"),
            "violation_count": report.get("violation_count"),
            "violation_group_count": len(report.get("violation_groups", {})),
            "register": str(register_path(config)),
            "register_status": status,
            "report": str(output),
        }
    print(_json(summary))
    if report["execution"] != "complete":
        return 1
    if arguments.establish_register:
        return 0
    status = register_status(config, report)
    return 0 if status["ok"] else 1


def command_swrl_materialize(arguments: argparse.Namespace) -> int:
    from .ontology_runtime import swrl_materialize

    config = _config(arguments)
    report = swrl_materialize(
        config,
        include_shards=not arguments.no_shards,
        shard_limit=arguments.shard_limit,
    )
    print(_json({key: value for key, value in report.items() if key != "data_sources"}))
    return 0 if report["statistics"]["new_triples"] >= 0 else 1


def command_tool_catalogue(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    services = AgentServices.create(config, knowledge=KnowledgeIndex(config))
    owner = AgentName(arguments.agent) if arguments.agent else None
    tools = services.tool_registry().manifest(owner)
    report = {
        "project_id": config.project_id,
        "agent": owner.value if owner else None,
        "tool_count": len(tools),
        "tools": tools,
    }
    if arguments.output:
        output = Path(arguments.output).resolve()
        atomic_write_json(output, report)
        print(_json({"tool_count": len(tools), "output": str(output)}))
    else:
        print(_json(report))
    return 0


def command_gt_run(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    dataset_path = Path(arguments.dataset).resolve()
    suite = load_query_suite(dataset_path)
    knowledge = KnowledgeIndex(config)
    if not knowledge.is_current():
        knowledge.build()
    services = AgentServices.create(
        config,
        knowledge=knowledge,
        planning_reasoner=planning_reasoner_from_env(
            max_retries=config.structured_output_max_retries
        ),
    )
    output_dir = Path(arguments.output_dir).resolve() if arguments.output_dir else (
        config.artifact_dir / "evaluation"
    )
    languages = ("en", "zh") if arguments.language == "both" else (arguments.language,)
    summaries = []
    for language in languages:
        predictions, execution = run_query_suite(config, services, suite, language)
        prediction_path = output_dir / f"predictions_{language}.json"
        execution_path = output_dir / f"execution_{language}.json"
        score_path = output_dir / f"score_{language}.json"
        atomic_write_json(prediction_path, predictions)
        atomic_write_json(execution_path, execution)
        score = score_predictions(
            dataset_path,
            prediction_path,
            path_root=config.project_root,
        )
        atomic_write_json(score_path, score)
        summaries.append(
            {
                "language": language,
                "execution": {key: value for key, value in execution.items() if key not in {"failures", "per_case"}},
                "score": score["metrics"],
                "prediction": portable_path(prediction_path, config.project_root),
                "execution_report": portable_path(execution_path, config.project_root),
                "score_report": portable_path(score_path, config.project_root),
            }
        )
    report = {"dataset_id": suite.dataset_id, "languages": summaries}
    atomic_write_json(output_dir / "summary.json", report)
    print(_json({"output": str(output_dir / "summary.json"), **report}))
    return 0


def command_gt_score(arguments: argparse.Namespace) -> int:
    dataset = Path(arguments.dataset).resolve()
    predictions = Path(arguments.predictions).resolve()
    report = score_predictions(dataset, predictions, path_root=Path.cwd())
    output = Path(arguments.output).resolve() if arguments.output else predictions.with_name(
        predictions.stem + "_score.json"
    )
    atomic_write_json(output, report)
    print(_json({"dataset_id": report["dataset_id"], "metrics": report["metrics"], "output": str(output)}))
    return 0


def command_commit(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    service = TransactionService(config.transaction_dir, config=config)
    output = service.commit(Path(arguments.preview).resolve(), arguments.confirm)
    print(_json({"status": "committed", "delta_graph": str(output), "source_graph_mutated": False}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msca-mas", description="Ontology-constrained multi-agent carbon assessment"
    )
    parser.add_argument("--config", help="Path to the project configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check input paths and the graph index")
    doctor.set_defaults(handler=command_doctor)

    prepare = subparsers.add_parser("prepare", help="Index the CKG and prepare project inputs")
    prepare.add_argument("--force-index", action="store_true", help="Rebuild an existing index")
    prepare.add_argument("--skip-properties", action="store_true", help="Skip IFC property and quantity extraction")
    prepare.add_argument("--full-properties", action="store_true", help="Extract properties from both IFC models")
    prepare.set_defaults(handler=command_prepare)

    case_build = subparsers.add_parser("case-build", help="Build the reference case inventory")
    case_build.add_argument("--force-index", action="store_true", help="Rebuild an existing index")
    case_build.set_defaults(handler=command_case_build)

    ask = subparsers.add_parser("ask", help="Answer a project query")
    ask.add_argument("query")
    ask.add_argument("--json", action="store_true", help="Print a structured report")
    ask.add_argument("--no-auto-index", action="store_true", help="Require a current CKG index")
    ask.set_defaults(handler=command_ask)

    llm_check = subparsers.add_parser("llm-check", help="Check the configured planning provider")
    llm_check.set_defaults(handler=command_llm_check)

    cq_check = subparsers.add_parser("cq-check", help="Run the competency question checks")
    cq_check.add_argument("--cq-id", action="append", help="Limit checks to the supplied IDs")
    cq_check.set_defaults(handler=command_cq_check)

    validate = subparsers.add_parser("validate", help="Check ontology, CQ, SWRL and SHACL inputs")
    validate.set_defaults(handler=command_validate)

    validate_ckg = subparsers.add_parser("validate-ckg", help="Check the CKG against its SHACL shapes")
    validate_ckg.add_argument("--shard-limit", type=int, help="Only load the first N work package shards")
    validate_ckg.add_argument(
        "--establish-register", action="store_true", help="Record the release's known validation issues"
    )
    validate_ckg.set_defaults(handler=command_validate_ckg)

    swrl = subparsers.add_parser("swrl-materialize", help="Run SWRL rules and record inferred facts")
    swrl.add_argument("--no-shards", action="store_true", help="Exclude work package shards")
    swrl.add_argument("--shard-limit", type=int, help="Only load the first N work package shards")
    swrl.set_defaults(handler=command_swrl_materialize)

    tools = subparsers.add_parser("tool-catalogue", help="List agent tools and input schemas")
    tools.add_argument("--agent", choices=[name.value for name in AgentName], help="Limit the list to one agent")
    tools.add_argument("--output", help="Write the list to a JSON file")
    tools.set_defaults(handler=command_tool_catalogue)

    run_queries = subparsers.add_parser("gt-run", help="Evaluate the MAS on a supplied query suite")
    run_queries.add_argument("--dataset", required=True, help="Query suite JSON")
    run_queries.add_argument("--language", choices=("en", "zh", "both"), default="en")
    run_queries.add_argument("--output-dir", help="Output directory for predictions and reports")
    run_queries.set_defaults(handler=command_gt_run)

    score = subparsers.add_parser("gt-score", help="Score saved predictions against a supplied query suite")
    score.add_argument("--dataset", required=True, help="Query suite JSON")
    score.add_argument("--predictions", required=True, help="Predictions JSON")
    score.add_argument("--output", help="Write the score report to a JSON file")
    score.set_defaults(handler=command_gt_score)

    commit = subparsers.add_parser("commit", help="Write a reviewed correction as a separate RDF delta")
    commit.add_argument("--preview", required=True, help="Pending correction JSON")
    commit.add_argument("--confirm", required=True, help="Confirmation digest from the reviewed preview")
    commit.set_defaults(handler=command_commit)

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
