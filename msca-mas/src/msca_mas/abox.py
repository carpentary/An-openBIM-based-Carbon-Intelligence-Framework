from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel

from .ifc import IfcInventoryReport, IfcProduct
from .util import atomic_write_json


MSCA = "http://bimlca.uni.edu/ontology#"
RECON = "http://bimlca.uni.edu/reconstructed/station01#"


class ClassificationDecision(BaseModel):
    base_class: str
    element_type: str | None = None
    candidate_type: str | None = None
    system: str
    construction_role: str | None = None
    status: Literal["confirmed", "candidate", "base_only"]
    confidence: float
    evidence: str


class ABoxBuildReport(BaseModel):
    output_ttl: str
    classification_register: str
    source_role: str
    element_count: int
    status_counts: dict[str, int]
    confirmed_type_counts: dict[str, int]
    candidate_type_counts: dict[str, int]
    downstream_status: str


def classify_enclosure(product: IfcProduct) -> ClassificationDecision:
    text = " ".join((product.name or "", product.object_type or "")).lower()
    horizontal = RECON + "System_HorizontalBracing"
    vertical = RECON + "System_VerticalRetaining"
    temporary = MSCA + "Role_TemporaryWork"
    dual = MSCA + "Role_DualPurposeElement"

    if "underground diaphragm wall" in text or "continuous wall" in text:
        return ClassificationDecision(
            base_class=MSCA + "Wall",
            element_type=MSCA + "Type_DiaphragmWall",
            system=vertical,
            construction_role=dual,
            status="confirmed",
            confidence=0.99,
            evidence="explicit project object/type name: Underground Diaphragm Wall",
        )
    if "steel pipe support" in text:
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_SteelPipeStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.99,
            evidence="explicit project type name: Steel Pipe Support",
        )
    if "steel connecting beam" in text:
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_SectionSteelStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.96,
            evidence="explicit project type name: Steel Connecting Beam; mapped to the steel-beam installation route",
        )
    if "steel enclosure beam" in text:
        return ClassificationDecision(
            base_class=MSCA + "Beam",
            element_type=MSCA + "Type_Waling",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.96,
            evidence="explicit project type name: Double-Span Steel Enclosure Beam; mapped to steel waling",
        )
    if "top support" in text or "mid support" in text:
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_ConcreteStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.95,
            evidence="explicit project type name identifies a concrete top/mid support member",
        )
    if "scissor bracing" in text:
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_SectionSteelStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.97,
            evidence="explicit project type name: Scissor Bracing",
        )
    if "braced column" in text:
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_LatticeColumn",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.95,
            evidence="explicit project type name: Braced Column",
        )
    if "grout mixing pile" in text or "jet grouting pile" in text:
        return ClassificationDecision(
            base_class=MSCA + "Pile",
            element_type=MSCA + "Type_MixedInPlacePile",
            system=vertical,
            status="confirmed",
            confidence=0.94,
            evidence="explicit project type name identifies in-situ soil/grout mixing pile method",
        )
    if "pillar pile" in text:
        return ClassificationDecision(
            base_class=MSCA + "SupportColumnPile",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.96,
            evidence="explicit project type name: Pillar Pile; mapped to SupportColumnPile class",
        )
    if "concrete circular pile" in text:
        return ClassificationDecision(
            base_class=MSCA + "SupportColumnPile",
            system=vertical,
            construction_role=dual,
            status="confirmed",
            confidence=0.96,
            evidence="explicit project type name: Concrete Circular Pile; mapped to SupportColumnPile class",
        )
    if "crown beam" in text:
        return ClassificationDecision(
            base_class=MSCA + "Beam",
            element_type=MSCA + "Type_CappingBeam",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.96,
            evidence="explicit project type name: Auxiliary Structure Crown Beam",
        )
    if "auxiliary structure support" in text or re_search_code(text, "ll"):
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_ConcreteStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.94,
            evidence="project support/LL naming identifies an auxiliary concrete bracing member",
        )
    if "plate" in text and "support" in text:
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_ConcreteStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.92,
            evidence="project plate-support naming identifies an auxiliary concrete bracing member",
        )
    if "underwater c35 plain concrete" in text:
        return ClassificationDecision(
            base_class=MSCA + "Wall",
            element_type=RECON + "Type_PlainConcreteRetainingWall",
            system=vertical,
            construction_role=temporary,
            status="confirmed",
            confidence=0.95,
            evidence="explicit project name identifies underwater C35 plain-concrete retaining work",
        )
    if product.ifc_entity == "IFCBUILDINGELEMENTPROXY" and "surface" in text:
        return ClassificationDecision(
            base_class=MSCA + "PhysicalElement",
            element_type=RECON + "Type_SiteSurfaceReference",
            system=horizontal,
            status="confirmed",
            confidence=0.99,
            evidence="IFC object is a site-surface display reference; carbon mapping is not applicable",
        )
    if re_search_code(text, "gl"):
        return ClassificationDecision(
            base_class=MSCA + "Beam",
            element_type=MSCA + "Type_CappingBeam",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.93,
            evidence="project naming code GL (capping/crown beam) plus concrete rectangular beam",
        )
    if re_search_code(text, "zc"):
        return ClassificationDecision(
            base_class=MSCA + "Bracing",
            element_type=MSCA + "Type_ConcreteStrut",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.93,
            evidence="project naming code ZC (support/strut) plus concrete rectangular beam",
        )
    if re_search_code(text, "yl"):
        return ClassificationDecision(
            base_class=MSCA + "Beam",
            element_type=MSCA + "Type_Waling",
            system=horizontal,
            construction_role=temporary,
            status="confirmed",
            confidence=0.93,
            evidence="project naming code YL (waist/waling beam) plus concrete rectangular beam",
        )
    base_map = {
        "IFCBEAM": MSCA + "Beam",
        "IFCCOLUMN": MSCA + "Column",
        "IFCWALLSTANDARDCASE": MSCA + "Wall",
        "IFCWALL": MSCA + "Wall",
        "IFCFOOTING": MSCA + "FoundationElement",
        "IFCSLAB": MSCA + "Slab",
        "IFCBUILDINGELEMENTPROXY": MSCA + "PhysicalElement",
    }
    return ClassificationDecision(
        base_class=base_map.get(product.ifc_entity, MSCA + "PhysicalElement"),
        system=vertical if product.ifc_entity in {"IFCWALL", "IFCWALLSTANDARDCASE", "IFCFOOTING"} else horizontal,
        status="base_only",
        confidence=0.50,
        evidence="IFC entity class only; no governed project naming rule matched",
    )


def re_search_code(text: str, code: str) -> bool:
    import re

    return bool(re.search(rf":{code}(?:-|\b)", text, re.IGNORECASE))


def _literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _product_uri(product: IfcProduct) -> str:
    identity = product.global_id or f"STEP-{product.step_id}"
    return RECON + "Element_" + quote(identity, safe="")


def build_enclosure_abox(report: IfcInventoryReport, output_dir: Path) -> ABoxBuildReport:
    products = [product for product in report.products if product.source_role == "enclosure_structure"]
    if not products:
        raise ValueError("inventory does not contain the enclosure_structure source")
    source_summary = next(source for source in report.sources if source.role == "enclosure_structure")
    output_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = output_dir / "station01_enclosure_input_abox.ttl"
    csv_path = output_dir / "station01_enclosure_classification_register.csv"
    decisions: list[tuple[IfcProduct, ClassificationDecision]] = [
        (product, classify_enclosure(product)) for product in products
    ]

    with ttl_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "@prefix msca: <http://bimlca.uni.edu/ontology#> .\n"
            "@prefix recon: <http://bimlca.uni.edu/reconstructed/station01#> .\n"
            "@prefix prov: <http://www.w3.org/ns/prov#> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
            "recon:EnclosureInputABox a owl:Ontology ;\n"
            "    rdfs:comment \"Reconstructed raw-IFC input overlay; not a completed carbon result graph.\"@en .\n\n"
            "recon:classificationStatus a rdf:Property .\n"
            "recon:confidenceScore a rdf:Property .\n"
            "recon:classificationEvidence a rdf:Property .\n"
            "recon:candidateElementType a rdf:Property .\n"
            "recon:ifcEntity a rdf:Property .\n"
            "recon:ifcStepId a rdf:Property .\n"
            "recon:ifcObjectType a rdf:Property .\n\n"
            "recon:System_Enclosure a msca:EnclosureStructureSystem .\n"
            "recon:System_HorizontalBracing a msca:HorizontalBracingSystem ; msca:subsystemOf recon:System_Enclosure .\n"
            "recon:System_VerticalRetaining a msca:VerticalRetainingSystem ; msca:subsystemOf recon:System_Enclosure .\n\n"
            f"recon:Source_EnclosureIFC a prov:Entity ; rdfs:label {_literal(source_summary.path)} ;\n"
            f"    recon:sha256 {_literal(source_summary.sha256)} ; recon:fileBytes \"{source_summary.bytes}\"^^xsd:integer .\n\n"
        )
        for product, decision in decisions:
            uri = _product_uri(product)
            predicates = [
                f"a <{decision.base_class}>",
                f"<http://www.w3.org/2000/01/rdf-schema#label> {_literal(product.name or product.global_id or str(product.step_id))}",
                f"<{MSCA}hasGlobalID> {_literal(product.global_id or '')}",
                f"<{MSCA}belongsToSystem> <{decision.system}>",
                f"<{MSCA}hasRepresentationMode> <{MSCA}Representation_ExplicitBIM>",
                "<http://www.w3.org/ns/prov#wasDerivedFrom> <http://bimlca.uni.edu/reconstructed/station01#Source_EnclosureIFC>",
                f"<{RECON}ifcEntity> {_literal(product.ifc_entity)}",
                f"<{RECON}ifcStepId> \"{product.step_id}\"^^<http://www.w3.org/2001/XMLSchema#integer>",
                f"<{RECON}classificationStatus> {_literal(decision.status)}",
                f"<{RECON}confidenceScore> \"{decision.confidence}\"^^<http://www.w3.org/2001/XMLSchema#decimal>",
                f"<{RECON}classificationEvidence> {_literal(decision.evidence)}",
            ]
            if product.object_type:
                predicates.append(f"<{RECON}ifcObjectType> {_literal(product.object_type)}")
            if decision.element_type:
                predicates.append(f"<{MSCA}hasElementType> <{decision.element_type}>")
            if decision.candidate_type:
                predicates.append(f"<{RECON}candidateElementType> <{decision.candidate_type}>")
            if decision.construction_role:
                predicates.append(f"<{MSCA}hasConstructionRole> <{decision.construction_role}>")
            handle.write(f"<{uri}>\n    " + " ;\n    ".join(predicates) + " .\n\n")

    fields = [
        "global_id",
        "step_id",
        "ifc_entity",
        "name",
        "object_type",
        "base_class",
        "element_type",
        "candidate_type",
        "system",
        "construction_role",
        "status",
        "confidence",
        "evidence",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for product, decision in decisions:
            writer.writerow(
                {
                    "global_id": product.global_id,
                    "step_id": product.step_id,
                    "ifc_entity": product.ifc_entity,
                    "name": product.name,
                    "object_type": product.object_type,
                    **decision.model_dump(mode="json"),
                }
            )
    status_counts = Counter(decision.status for _, decision in decisions)
    confirmed = Counter(decision.element_type for _, decision in decisions if decision.element_type)
    candidates = Counter(decision.candidate_type for _, decision in decisions if decision.candidate_type)
    build_report = ABoxBuildReport(
        output_ttl=str(ttl_path),
        classification_register=str(csv_path),
        source_role="enclosure_structure",
        element_count=len(products),
        status_counts=dict(status_counts),
        confirmed_type_counts={key: value for key, value in confirmed.items() if key},
        candidate_type_counts={key: value for key, value in candidates.items() if key},
        downstream_status=(
            "INPUT_ABOX_READY; classification rules are project-specific; complete quantity, CPD/EFD mapping "
            "and carbon results are published by the reference-case build."
        ),
    )
    atomic_write_json(output_dir / "station01_enclosure_abox_report.json", build_report.model_dump(mode="json"))
    return build_report
