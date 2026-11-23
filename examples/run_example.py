"""Run the restricted synthetic geometry-to-carbon example without an API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD

from msca_mas.calculations import transport_emission
from msca_mas.config import IfcSource
from msca_mas.ifc import ENTITY_RE, build_inventory, iter_step_records, parse_ref, split_step_args
from msca_mas.ifc_properties import extract_properties
from msca_mas.swrl_engine import IndexedSWRLForwardChainer

HERE = Path(__file__).resolve().parent
MSCA = Namespace("http://bimlca.uni.edu/ontology#")
BRIDGE = Namespace("http://bimlca.uni.edu/evidence#")
IFCB = Namespace("http://bimlca.uni.edu/ifc/bridge#")
EX = Namespace("https://example.org/msca/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
PROV = Namespace("http://www.w3.org/ns/prov#")
CARBON_UNIT = URIRef("http://qudt.org/vocab/unit/KiloGM-CO2_Equivalent")


def read_table(name: str) -> list[dict[str, str]]:
    with (HERE / "reference" / name).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def box_dimensions(path: Path, product) -> tuple[float, float, float, float]:
    """Read this fixture's metre-based, axis-aligned rectangular extrusion."""
    records = {}
    for line in iter_step_records(path):
        match = ENTITY_RE.match(line)
        if match:
            records[int(match[1])] = (match[2], split_step_args(match[3]))

    def arguments(step, kind):
        entity, fields = records[step]
        if entity != kind:
            raise ValueError(f"Example expects {kind}, not {entity}")
        return fields

    units = [args for kind, args in records.values() if kind == "IFCSIUNIT"]
    if not any(args[1:] == [".LENGTHUNIT.", "$", ".METRE."] for args in units):
        raise ValueError("This example requires metre-based IFC geometry")
    shape = arguments(product.representation_ref, "IFCPRODUCTDEFINITIONSHAPE")
    representation = arguments(int(shape[2].strip("()#")), "IFCSHAPEREPRESENTATION")
    solid = arguments(int(representation[3].strip("()#")), "IFCEXTRUDEDAREASOLID")
    rectangle = arguments(parse_ref(solid[0]), "IFCRECTANGLEPROFILEDEF")
    placement = arguments(product.placement_ref, "IFCLOCALPLACEMENT")
    axis = arguments(parse_ref(placement[1]), "IFCAXIS2PLACEMENT3D")
    point = arguments(parse_ref(axis[0]), "IFCCARTESIANPOINT")
    z = float(point[0].strip("()").split(",")[2])
    return float(rectangle[3]), float(rectangle[4]), float(solid[3]), z


def run(output: Path) -> dict:
    context = json.loads((HERE / "reference/context.json").read_text("utf8"))
    ifc = HERE / "ifc/station_fragment.ifc"
    inventory = build_inventory([IfcSource(ifc, "main_structure")])
    if inventory.combined_product_count != 1:
        raise ValueError("This example expects one synthetic slab")
    slab = inventory.products[0]
    length, width, depth, z = box_dimensions(ifc, slab)
    volume = length * width * depth
    report = extract_properties(ifc, "main_structure", [slab.step_id])
    evidence = {row.property_name: row.value for row in report.assertions}
    if not math.isclose(volume, float(evidence["NetVolume"])):
        raise ValueError("The IFC quantity and extrusion volume disagree")

    graph = Graph().parse(HERE / "ontology/schema.ttl")
    rules = Graph().parse(HERE / "ontology/rules.ttl")
    graph.add((EX.Frame, RDF.type, MSCA.MainFrameSystem))
    graph.add((EX.Slab, RDF.type, MSCA.Slab))
    graph.add((EX.Slab, MSCA.belongsToSystem, EX.Frame))
    graph.add((EX.Slab, IFCB.hasIfcEntityKind, IFCB.IFCKind_IfcSlab))
    graph.add((EX.Slab, IFCB.hasIfcPredefinedTypeConcept, IFCB.IfcSlabPredefinedType_FLOOR))
    if slab.predefined_type != "FLOOR" or "bottom" not in slab.name.lower():
        raise ValueError("Missing slab type or name evidence")
    graph.add((EX.Slab, BRIDGE.hasNormalizedNameConcept, BRIDGE.Name_StructuralBottomSlab))
    if evidence.get("LoadBearing") is True:
        graph.add((EX.Slab, BRIDGE.hasLoadBearingStatus, BRIDGE.LoadBearing_True))
    if evidence.get("StructuralUsage") == "LoadBearingFloorPlate":
        graph.add((EX.Slab, BRIDGE.hasStructuralUsageConcept, BRIDGE.StructuralUsage_LoadBearingFloorPlate))
    if evidence.get("MaterialFamily") == "Concrete":
        graph.add((EX.Slab, BRIDGE.hasMaterialFamily, BRIDGE.MaterialFamily_Concrete))
    if math.isclose(z, context["design_bottom_elevation_m"], abs_tol=context["connection_tolerance_m"]):
        graph.add((EX.Slab, BRIDGE.hasVerticalRole, BRIDGE.VerticalRole_LowestStructuralSlab))
    graph.add((MSCA.WPT_SLAB_BOTTOM, MSCA.hasProcessFamily, MSCA.PF_Concrete))
    first = IndexedSWRLForwardChainer(graph, rules)
    first.run()
    if (EX.Slab, MSCA.hasElementType, MSCA.Type_BottomSlab) not in graph:
        raise ValueError("Insufficient evidence for structural bottom-slab classification")

    domain = context["excavation_domain"]
    area = (domain["xmax"] - domain["xmin"]) * (domain["ymax"] - domain["ymin"])
    excavation = area * (domain["site_z"] - domain["bottom_z"])
    layers = {}
    for layer in context["soil_layers"]:
        height = max(0, min(domain["site_z"], layer["top_z"]) - max(domain["bottom_z"], layer["bottom_z"]))
        layers[layer["name"]] = area * height
    if not math.isclose(sum(layers.values()), excavation):
        raise ValueError("The synthetic soil layers do not partition the excavation")

    factors = {row["factor_id"]: row for row in read_table("efd.csv")}
    ledger = []
    for row in read_table("cpd.csv"):
        work_quantity = volume if row["entry_id"] == "EX-CONCRETE" else excavation
        reference = float(row["reference_quantity"])
        if reference <= 0 or row["work_unit"] != "m3":
            raise ValueError("Invalid CPD reference quantity or work unit")
        demand = work_quantity / reference * float(row["quota"])
        factor = factors[row["factor_id"]]
        expected_stage = "A1-A3_ProductStage" if row["resource_kind"] == "material" else "A5_ConstructionStage"
        if (factor["resource_kind"], factor["activity_unit"], factor["stage"]) != (
            row["resource_kind"], row["resource_unit"], expected_stage
        ):
            raise ValueError("Incompatible resource, activity unit or lifecycle stage")
        ledger.append({
            "resource": row["resource_code"], "work": row["entry_id"],
            "quantity": demand, "unit": row["resource_unit"],
            "stage": factor["stage"], "factor": factor["factor_id"],
            "factor_value": float(factor["factor_value"]),
            "carbon": demand * float(factor["factor_value"]),
        })
    material = next(row for row in ledger if row["resource"] == "EX-CONCRETE-MAT")
    freight = factors["EX-TRANSPORT"]
    if (freight["activity_unit"], freight["stage"]) != ("t_km", "A4_TransportationStage"):
        raise ValueError("Incompatible freight factor")
    mass = material["quantity"] * context["concrete_density_t_per_m3"]
    a4 = transport_emission(mass, "t", context["delivery_distance_km"], float(freight["factor_value"]))
    ledger.append({
        "resource": "EX-DELIVERY", "work": "EX-CONCRETE",
        "quantity": mass * context["delivery_distance_km"], "unit": "t_km",
        "stage": "A4_TransportationStage", "factor": "EX-TRANSPORT",
        "factor_value": float(freight["factor_value"]), "carbon": a4["result"],
    })
    for i, row in enumerate(ledger):
        work, resource, emission, quantity = (EX[row["work"]], EX[row["resource"]], EX[f"emission-{i}"], EX[f"quantity-{i}"])
        activity_quantity = EX[f"activity-{i}"]
        descriptor = EX[row["resource"] + "-descriptor"]
        activity_unit = {"m3": URIRef("http://qudt.org/vocab/unit/M3"),
                         "shift": EX.MachineShift, "t_km": EX.TonneKilometre}[row["unit"]]
        graph.add((work, RDF.type, MSCA.ProjectWorkPackage))
        graph.add((work, MSCA.requiresResource, resource))
        graph.add((resource, RDF.type, MSCA.ResourceInstance))
        graph.add((resource, MSCA.describedBy, descriptor))
        graph.add((descriptor, RDF.type, MSCA.ResourceDescriptor))
        graph.add((resource, MSCA.hasActualQuantity, activity_quantity))
        graph.add((activity_quantity, QUDT.numericValue, Literal(row["quantity"], datatype=XSD.double)))
        graph.add((activity_quantity, QUDT.unit, activity_unit))
        graph.add((EX[row["factor"]], RDF.type, MSCA.EFD_Entry))
        graph.add((EX[row["factor"]], EX.factorValue, Literal(row["factor_value"], datatype=XSD.double)))
        graph.add((EX[row["factor"]], EX.activityUnit, activity_unit))
        if row["work"] == "EX-CONCRETE":
            graph.add((EX.Slab, MSCA.requiresWorkPackage, work))
            graph.add((work, MSCA.instantiatedFrom, MSCA.WPT_SLAB_BOTTOM))
        else:
            graph.add((EX.ExcavationDomain, MSCA.requiresWorkPackage, work))
            graph.add((EX.ExcavationDomain, RDF.type, MSCA.Element))
            graph.add((EX.ExcavationDomain, MSCA.belongsToSystem, EX.PitSystem))
        graph.add((EX.PitSystem, RDF.type, MSCA.EngineeringSystem))
        for triple in (
            (emission, RDF.type, MSCA.CarbonEmission),
            (emission, MSCA.calculatedFromResource, resource),
            (emission, MSCA.calculatedUsingFactor, EX[row["factor"]]),
            (emission, MSCA.belongsToStage, MSCA[row["stage"]]),
            (emission, MSCA.hasEmissionQuantity, quantity),
            (emission, PROV.wasGeneratedBy, EX.DeterministicCalculation),
            (quantity, QUDT.numericValue, Literal(row["carbon"], datatype=XSD.double)),
            (quantity, QUDT.unit, CARBON_UNIT),
        ):
            graph.add(triple)
    second = IndexedSWRLForwardChainer(graph, rules)
    second.run()
    conforms, _, message = validate(graph, shacl_graph=Graph().parse(HERE / "ontology/shapes.ttl"))
    if not conforms:
        raise ValueError(message)
    totals = {str(row.stage).split("#")[-1]: float(row.carbon)
              for row in graph.query((HERE / "ontology/total_by_stage.rq").read_text("utf8"))}
    summary = {
        "data_kind": "synthetic demonstration; not a station result or model evaluation",
        "slab_volume_m3": volume,
        "excavation_volume_m3": excavation,
        "soil_layer_volumes_m3": layers,
        "concrete_demand_m3": material["quantity"],
        "stage_totals_kgco2e": totals,
        "total_kgco2e": sum(totals.values()),
        "shacl_conforms": bool(conforms),
        "rule_ids": sorted({inf.rule for inf in first.inferences + second.inferences}),
        "ifc_sha256": hashlib.sha256(ifc.read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    graph.serialize(output / "project_graph.ttl", format="turtle")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    (output / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    summary = run(args.output)
    expected = json.loads((HERE / "expected_outputs.json").read_text("utf8"))
    for key in ("slab_volume_m3", "excavation_volume_m3", "concrete_demand_m3", "total_kgco2e"):
        if not math.isclose(summary[key], expected[key], abs_tol=1e-8):
            raise AssertionError(key)
    if summary["soil_layer_volumes_m3"] != expected["soil_layer_volumes_m3"]:
        raise AssertionError("soil layers")
    for stage, value in expected["stage_totals_kgco2e"].items():
        if not math.isclose(summary["stage_totals_kgco2e"][stage], value, abs_tol=1e-8):
            raise AssertionError(stage)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
