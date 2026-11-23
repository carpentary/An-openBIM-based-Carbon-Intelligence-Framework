from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .entity_resolution import (
    TRANSPORT_TARGET_PREFIX,
    matching_transport_defaults,
    transport_material_labels,
)
from .models import ScenarioOverlay
from .util import atomic_write_json, stable_digest


MSCA = "http://bimlca.uni.edu/ontology#"
CKG = "http://bimlca.uni.edu/ckg/station01/rc4#"
UNIT = "http://qudt.org/vocab/unit/"
IRI_RE = re.compile(r"^https?://[^\s<>\"{}|\\^`]+$")


class TransactionPreview(BaseModel):
    transaction_id: str
    overlay_id: str
    operation: str
    target_uri: str
    predicate_uri: str
    new_value: float
    unit_uri: str
    created_at_utc: str
    confirmation_digest: str
    source_graph_mutated: bool = False


class TransactionService:
    """Append-only correction deltas; source ontology/CKG files are never edited."""

    def __init__(self, transaction_dir: Path, config: Any | None = None):
        self.transaction_dir = transaction_dir
        # Use contextual SHACL when a project configuration is supplied.
        self.config = config

    def preview(
        self,
        overlay: ScenarioOverlay,
        *,
        query_contract: dict[str, Any] | None = None,
        material_transport_defaults: list[dict[str, Any]] | None = None,
    ) -> TransactionPreview | None:
        if not overlay.ready or not overlay.changes:
            return None
        change = overlay.changes[0]
        if change.change_type != "transport_override" or "distance_km" not in change.parameters:
            return None
        material_prefix = TRANSPORT_TARGET_PREFIX
        direct_material_code = ""
        if change.target.startswith(material_prefix):
            direct_material_code = change.target[len(material_prefix):]
        if re.fullmatch(r"M-\d+-\d+", direct_material_code):
            material_code = direct_material_code.upper()
            operation = "upsert_material_transport_parameter"
            target_uri = CKG + "MaterialTransportDefault_" + material_code.replace("-", "_")
        else:
            labels = transport_material_labels(change.target, query_contract)
            matching = matching_transport_defaults(
                list(material_transport_defaults or []),
                labels,
                query=str((query_contract or {}).get("query", "")),
            )
            material_intent = bool(labels) or change.target.startswith(material_prefix)
            if len(matching) == 1:
                material_code = str(matching[0]["material_code"]).upper()
                operation = "upsert_material_transport_parameter"
                target_uri = CKG + "MaterialTransportDefault_" + material_code.replace("-", "_")
            elif material_intent:
                return None
            else:
                operation = "upsert_project_parameter"
                target_uri = CKG + "ProjectTransportDistance_UserCorrection"
        core = {
            "overlay_id": overlay.overlay_id,
            "operation": operation,
            "target_uri": target_uri,
            "predicate_uri": MSCA + "hasTransportDistance",
            "new_value": float(change.parameters["distance_km"]),
            "unit_uri": UNIT + "KiloM",
        }
        self._shacl_gate(target_uri, float(change.parameters["distance_km"]))
        digest = stable_digest(core)
        return TransactionPreview(
            transaction_id=f"txn-{digest[:16]}",
            created_at_utc=datetime.now(UTC).isoformat(),
            confirmation_digest=digest,
            source_graph_mutated=False,
            **core,
        )

    def _shacl_gate(self, target_uri: str, new_value: float) -> None:
        """Reject a preview if it introduces a SHACL violation at the target."""

        import os

        if self.config is None:
            return
        if os.environ.get("MSCA_TX_SHACL", "").lower() in {"off", "0", "false"}:
            return
        try:
            from .ontology_runtime import validate_delta_in_context
        except ImportError:
            raise ValueError("TRANSACTION_SHACL_GATE_UNAVAILABLE: pyshacl is required for governed previews")

        from rdflib import URIRef
        from rdflib.namespace import PROV
        from rdflib import Literal

        now = datetime.now(UTC).isoformat()
        target = URIRef(target_uri)
        decimal_type = URIRef("http://www.w3.org/2001/XMLSchema#decimal")
        date_time_type = URIRef("http://www.w3.org/2001/XMLSchema#dateTime")
        generator = URIRef(CKG + f"txn-{stable_digest([target_uri, new_value, now])[:16]}")
        delta_triples = [
            (target, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef(MSCA + "UserCorrection")),
            (target, URIRef(MSCA + "hasTransportDistance"), Literal(f"{new_value}", datatype=decimal_type)),
            (target, URIRef(MSCA + "hasParameterUnit"), URIRef(UNIT + "KiloM")),
            (target, PROV.generatedAtTime, Literal(now, datatype=date_time_type)),
            (target, PROV.wasGeneratedBy, generator),
        ]
        report = validate_delta_in_context(self.config, target_uri, delta_triples)
        if not report.get("ok"):
            introduced = report.get("introduced_violation_groups") or report.get("error") or "engine failure"
            raise ValueError(
                "TRANSACTION_SHACL_VIOLATION: the correction preview would introduce "
                f"new SHACL violations for {target_uri}: {introduced}"
            )

    def save_preview(self, preview: TransactionPreview) -> Path:
        path = self.transaction_dir / "pending" / f"{preview.transaction_id}.json"
        atomic_write_json(path, preview.model_dump(mode="json"))
        return path

    def commit(self, preview_path: Path, confirmation_digest: str) -> Path:
        data = json.loads(preview_path.read_text(encoding="utf-8"))
        preview = TransactionPreview.model_validate(data)
        if confirmation_digest != preview.confirmation_digest:
            raise PermissionError("confirmation digest does not match the reviewed transaction")
        for uri in (preview.target_uri, preview.predicate_uri, preview.unit_uri):
            if not IRI_RE.fullmatch(uri):
                raise ValueError(f"unsafe IRI in transaction: {uri}")
        committed_dir = self.transaction_dir / "committed"
        committed_dir.mkdir(parents=True, exist_ok=True)
        output = committed_dir / f"{preview.transaction_id}.ttl"
        if output.exists():
            raise FileExistsError(f"transaction already committed: {output}")
        timestamp = preview.created_at_utc.replace('"', "")
        ttl = (
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
            "@prefix prov: <http://www.w3.org/ns/prov#> .\n\n"
            f"<{preview.target_uri}> a <{MSCA}UserCorrection> ;\n"
            f"    <{preview.predicate_uri}> "
            f"\"{preview.new_value}\"^^xsd:decimal ;\n"
            f"    <{MSCA}hasParameterUnit> <{preview.unit_uri}> ;\n"
            f"    prov:generatedAtTime \"{timestamp}\"^^xsd:dateTime ;\n"
            f"    prov:wasGeneratedBy <{CKG}{preview.transaction_id}> .\n"
        )
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(ttl)
        return output
