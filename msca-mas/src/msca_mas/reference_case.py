from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph, Namespace

from .config import ProjectConfig
from .knowledge import KnowledgeIndex, local_name
from .util import atomic_write_json


MSCA = Namespace("http://bimlca.uni.edu/ontology#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
RECON = "http://bimlca.uni.edu/reconstructed/station01/reference-case#"


@dataclass(frozen=True)
class EfdFactor:
    resource_code: str
    entry_code: str
    name: str
    value: float
    unit_uri: str
    factor_uri: str
    resource_uri: str


@dataclass(frozen=True)
class RecipeRow:
    cpd_code: str
    subproject: str
    entry_name: str
    work_unit: str
    measure_unit: float
    resource_kind: str
    resource_code: str
    resource_quantity: float
    resource_unit: str
    resource_role: str
    source_basis: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _safe_id(*parts: object) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    return digest


def _literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _material_code_from_efd(entry_code: str) -> str | None:
    match = re.fullmatch(r"EFD_1_M_(\d+)-(\d+)", entry_code)
    return f"M-{match.group(1)}-{match.group(2)}" if match else None


def _equipment_code_from_efd(entry_code: str) -> str | None:
    match = re.fullmatch(r"EFD_E_[A-Z]+_(\d+)-(\d+)", entry_code)
    return f"E-{match.group(1)}-{match.group(2)}" if match else None


def _preferred_name(graph: Graph, subject: Any) -> str:
    values = list(graph.objects(subject, MSCA.hasEFDEntryName))
    for value in values:
        if getattr(value, "language", None) == "en":
            return str(value)
    return str(values[0]) if values else local_name(str(subject))


def load_efd_factors(path: Path) -> tuple[dict[str, EfdFactor], dict[str, EfdFactor]]:
    graph = Graph()
    graph.parse(path, format="turtle")
    materials: dict[str, EfdFactor] = {}
    equipment: dict[str, EfdFactor] = {}
    for entry in graph.subjects(MSCA.hasEFDEntryCode, None):
        entry_code_value = graph.value(entry, MSCA.hasEFDEntryCode)
        factor = graph.value(entry, MSCA.hasFactor)
        quantity_value = graph.value(factor, MSCA.factorValue) if factor else None
        numeric = graph.value(quantity_value, QUDT.numericValue) if quantity_value else None
        unit = graph.value(quantity_value, QUDT.unit) if quantity_value else None
        resource = graph.value(factor, MSCA.factorResource) if factor else None
        if entry_code_value is None or factor is None or numeric is None or unit is None or resource is None:
            continue
        entry_code = str(entry_code_value)
        code = _material_code_from_efd(entry_code)
        target = materials
        if code is None:
            code = _equipment_code_from_efd(entry_code)
            target = equipment
        if code is None:
            continue
        target[code] = EfdFactor(
            resource_code=code,
            entry_code=entry_code,
            name=_preferred_name(graph, entry),
            value=float(numeric),
            unit_uri=str(unit),
            factor_uri=str(factor),
            resource_uri=str(resource),
        )
    return materials, equipment


def _load_recipes(path: Path) -> dict[str, list[RecipeRow]]:
    recipes: dict[str, list[RecipeRow]] = defaultdict(list)
    for row in _read_csv(path):
        item = RecipeRow(
            cpd_code=row["cpd_code"],
            subproject=row["subproject"],
            entry_name=row["entry_name"],
            work_unit=row["work_unit"],
            measure_unit=float(row["measure_unit"]),
            resource_kind=row["resource_kind"],
            resource_code=row["resource_code"],
            resource_quantity=float(row["resource_quantity"]),
            resource_unit=row["resource_unit"],
            resource_role=row["resource_role"],
            source_basis=row["source_basis"],
        )
        recipes[item.cpd_code].append(item)
    return dict(recipes)


MATERIAL_FACTOR_UNITS = {
    "kg": "Unit_KgCO2ePerKg",
    "t": "Unit_KgCO2ePerMetricTonne",
    "m3": "Unit_KgCO2ePerM3",
    "m2": "Unit_KgCO2ePerM2",
    "m": "Unit_KgCO2ePerM",
    "kWh": "Unit_KgCO2ePerKWh",
}


def _factor_activity(
    recipe: RecipeRow,
    factor: EfdFactor,
    resource_quantity: float,
    machine_shift_hours: float,
) -> tuple[float, str, float, str]:
    """Return the quantity expressed in the denominator unit of the EFD factor."""
    factor_unit = local_name(factor.unit_uri)
    if recipe.resource_kind == "material":
        expected = MATERIAL_FACTOR_UNITS.get(recipe.resource_unit)
        if expected != factor_unit:
            raise ValueError(
                f"material unit mismatch for {recipe.resource_code}: "
                f"CPD uses {recipe.resource_unit}, EFD uses {factor_unit}"
            )
        return resource_quantity, recipe.resource_unit, 1.0, "DIRECT_COMPATIBLE_UNIT"

    if recipe.resource_unit != "machine_shift":
        raise ValueError(
            f"unsupported equipment activity unit for {recipe.resource_code}: {recipe.resource_unit}"
        )
    if factor_unit == "Unit_KgCO2ePerEquipmentShift":
        return resource_quantity, "machine_shift", 1.0, "DIRECT_EQUIPMENT_SHIFT"
    if factor_unit == "Unit_KgCO2ePerHour":
        if machine_shift_hours <= 0:
            raise ValueError("machine_shift_hours must be positive")
        return (
            resource_quantity * machine_shift_hours,
            "hour",
            machine_shift_hours,
            "MACHINE_SHIFT_TO_HOUR",
        )
    raise ValueError(
        f"unsupported equipment factor unit for {recipe.resource_code}: {factor_unit}"
    )


def _load_transport(reference_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    vehicles = {
        row["vehicle_code"]: {
            **row,
            "factor_kgco2e_per_tkm": float(row["factor_kgco2e_per_tkm"]),
        }
        for row in _read_csv(reference_dir / "transport_vehicles.csv")
    }
    materials = {
        row["material_code"]: {
            **row,
            "default_distance_km": float(row["default_distance_km"]),
            "mass_t_per_source_unit": float(row["mass_t_per_source_unit"]),
            "transport_applicable": _bool(row["transport_applicable"]),
        }
        for row in _read_csv(reference_dir / "material_transport_defaults.csv")
    }
    missing_vehicles = sorted({row["vehicle_code"] for row in materials.values()} - set(vehicles))
    if missing_vehicles:
        raise ValueError(f"transport defaults reference unknown vehicles: {missing_vehicles}")
    return vehicles, materials


def _quantity_index(path: Path) -> dict[int, dict[str, float]]:
    candidates: dict[int, dict[str, tuple[int, float]]] = defaultdict(dict)
    for row in _read_csv(path):
        try:
            step_id = int(row["product_step_id"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if value < 0:
            continue
        kind = row.get("property_kind", "").upper()
        name = row.get("property_name", "").strip().lower()
        quantity_kind = None
        for token in ("volume", "length", "area", "weight"):
            if kind.endswith(token.upper()) or name == token:
                quantity_kind = token
                break
        if quantity_kind is None:
            continue
        score = 2 if name == quantity_kind else 1
        current = candidates[step_id].get(quantity_kind)
        if current is None or score > current[0]:
            candidates[step_id][quantity_kind] = (score, value)
    return {
        step_id: {kind: scored[1] for kind, scored in values.items()}
        for step_id, values in candidates.items()
    }


def _element_variant(element_type: str, name: str, object_type: str) -> str:
    text = f"{name} {object_type}".lower()
    if element_type == "Type_Waling":
        return "steel_waling" if "steel enclosure beam" in text else "concrete"
    if element_type == "Type_SectionSteelStrut":
        return "steel_beam" if "steel connecting beam" in text else "steel_support"
    return {
        "Type_CappingBeam": "concrete",
        "Type_ConcreteStrut": "concrete",
        "Type_SteelPipeStrut": "steel_support",
        "Type_LatticeColumn": "steel_column",
        "SupportColumnPile": "concrete_pile",
        "Type_MixedInPlacePile": "cement_soil",
        "Type_PlainConcreteRetainingWall": "plain_concrete",
        "Type_DiaphragmWall": "diaphragm_wall",
        "Type_SiteSurfaceReference": "display_reference",
    }.get(element_type, "default")


def _load_element_mappings(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_csv(path):
        row["work_package_codes"] = [code for code in row["work_package_codes"].split(";") if code]
        row["primary_quantity_per_work_unit"] = (
            float(row["primary_quantity_per_work_unit"])
            if row["primary_quantity_per_work_unit"]
            else None
        )
        result[(row["element_type"], row["variant"])] = row
    return result


def _build_enclosure_work_packages(
    reference_dir: Path,
    classification_path: Path,
    quantity_path: Path,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mappings = _load_element_mappings(reference_dir / "element_work_package_map.csv")
    quantities = _quantity_index(quantity_path)
    assignments: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str, str], float] = defaultdict(float)
    aggregate_counts: Counter[tuple[str, str, str]] = Counter()
    diaphragm_volume = 0.0
    diaphragm_length_m = 0.0
    applicable = mapped = not_applicable = missing_quantity = 0
    steel_density = float(profile["enclosure"]["steel_density_t_per_m3"])

    for row in _read_csv(classification_path):
        step_id = int(row["step_id"])
        element_type = local_name(row["element_type"] or row["base_class"])
        variant = _element_variant(element_type, row.get("name", ""), row.get("object_type", ""))
        mapping = mappings.get((element_type, variant))
        qto = quantities.get(step_id, {})
        assignment: dict[str, Any] = {
            "global_id": row["global_id"],
            "step_id": step_id,
            "ifc_entity": row["ifc_entity"],
            "name": row["name"],
            "element_type": element_type,
            "variant": variant,
            "volume_m3": qto.get("volume"),
            "length_m": qto.get("length", 0.0) / 1000.0 if "length" in qto else None,
            "area_m2": qto.get("area"),
            "mapping_status": "unmapped",
            "quantity_basis": None,
            "work_quantity": None,
            "work_quantity_unit": None,
            "work_package_codes": None,
            "source_basis": "IFC_DERIVED",
            "parameter_class": "inferred",
        }
        if mapping is None:
            assignments.append(assignment)
            continue
        assignment["work_package_codes"] = ";".join(mapping["work_package_codes"])
        assignment["quantity_basis"] = mapping["quantity_basis"]
        if mapping["applicability"] == "not_applicable":
            not_applicable += 1
            assignment["mapping_status"] = "not_applicable"
            assignments.append(assignment)
            continue
        applicable += 1
        volume = qto.get("volume")
        if volume is None:
            missing_quantity += 1
            assignment["mapping_status"] = "missing_quantity"
            assignments.append(assignment)
            continue
        if mapping["quantity_basis"] == "steel_mass_t":
            work_quantity = volume * steel_density
            work_unit = "t"
        else:
            work_quantity = volume
            work_unit = "m3"
        assignment["work_quantity"] = work_quantity
        assignment["work_quantity_unit"] = work_unit
        assignment["mapping_status"] = "mapped"
        mapped += 1
        if element_type == "Type_DiaphragmWall":
            diaphragm_volume += volume
            diaphragm_length_m += qto.get("length", 0.0) / 1000.0
        else:
            code = mapping["work_package_codes"][0]
            key = (element_type, variant, code)
            aggregates[key] += work_quantity
            aggregate_counts[key] += 1
        assignments.append(assignment)

    work_packages: list[dict[str, Any]] = []
    for (element_type, variant, code), quantity in sorted(aggregates.items()):
        mapping = mappings[(element_type, variant)]
        work_packages.append(
            {
                "scope": "enclosure_structure",
                "owner_type": element_type,
                "variant": variant,
                "element_count": aggregate_counts[(element_type, variant, code)],
                "cpd_code": code,
                "work_quantity": quantity,
                "work_unit": "t" if mapping["quantity_basis"] == "steel_mass_t" else "m3",
                "repetitions": 1.0,
                "quantity_basis": mapping["quantity_basis"],
                "source_basis": "IFC_DERIVED",
                "parameter_class": "inferred",
                "primary_material_code": mapping["primary_material_code"] or None,
                "primary_quantity_per_work_unit": mapping["primary_quantity_per_work_unit"],
                "primary_resource_unit": mapping["primary_resource_unit"] or None,
            }
        )

    enclosure = profile["enclosure"]
    diaphragm_specs = (
        ("OC-WP-2-1-2-1", diaphragm_length_m * float(enclosure["guide_wall_excavation_m3_per_m"]), "m3", "guide_wall_excavation"),
        ("OC-WP-2-1-2-2", diaphragm_length_m * float(enclosure["guide_wall_concrete_m3_per_m"]), "m3", "guide_wall_concrete"),
        ("OC-WP-2-1-2-3", diaphragm_length_m * float(enclosure["guide_wall_formwork_m2_per_m"]), "m2", "guide_wall_formwork"),
        ("OC-WP-2-1-2-4", diaphragm_length_m * float(enclosure["guide_wall_rebar_t_per_m"]), "t", "guide_wall_rebar"),
        ("OC-WP-2-4-1-1", diaphragm_volume, "m3", "ifc_wall_volume"),
        ("OC-WP-2-1-2-6", diaphragm_volume * float(enclosure["diaphragm_wall_rebar_t_per_m3"]), "t", "wall_rebar"),
        ("OC-WP-2-1-2-7", diaphragm_volume * float(enclosure["diaphragm_wall_rebar_t_per_m3"]), "t", "wall_rebar"),
        ("OC-WP-2-1-4-1", diaphragm_volume, "m3", "ifc_wall_volume"),
    )
    for code, quantity, unit, basis in diaphragm_specs:
        work_packages.append(
            {
                "scope": "enclosure_structure",
                "owner_type": "Type_DiaphragmWall",
                "variant": "diaphragm_wall",
                "element_count": sum(1 for row in assignments if row["element_type"] == "Type_DiaphragmWall"),
                "cpd_code": code,
                "work_quantity": quantity,
                "work_unit": unit,
                "repetitions": 1.0,
                "quantity_basis": basis,
                "source_basis": "IFC_DERIVED+MODEL_PARAMETER" if basis != "ifc_wall_volume" else "IFC_DERIVED",
                "parameter_class": "scenario_parameter" if basis != "ifc_wall_volume" else "inferred",
                "primary_material_code": None,
                "primary_quantity_per_work_unit": None,
                "primary_resource_unit": None,
            }
        )

    type_counts = Counter(row["element_type"] for row in assignments)
    variant_counts = Counter(f"{row['element_type']}::{row['variant']}" for row in assignments)
    coverage = {
        "total_elements": len(assignments),
        "applicable_elements": applicable,
        "mapped_applicable_elements": mapped,
        "not_applicable_elements": not_applicable,
        "missing_quantity_elements": missing_quantity,
        "mapping_rate": mapped / applicable if applicable else 1.0,
        "element_type_counts": dict(sorted(type_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "diaphragm_wall_volume_m3": diaphragm_volume,
        "diaphragm_wall_length_m": diaphragm_length_m,
    }
    return work_packages, assignments, coverage


def _foundation_geometry(profile: dict[str, Any], utilisation: float | None = None) -> dict[str, float]:
    geometry = profile["public_geometry"]
    parameters = profile["foundation_pit"]
    gross = (
        float(geometry["station_length_m"])
        * float(geometry["station_width_m"])
        * float(geometry["station_depth_m"])
    )
    factor = float(parameters["plan_utilisation_factor"] if utilisation is None else utilisation)
    excavation = gross * factor
    backfill = excavation * float(parameters["backfill_ratio"])
    disposal_in_situ = excavation - backfill
    disposal_loose = disposal_in_situ * float(parameters["loose_volume_factor"])
    return {
        "gross_envelope_m3": gross,
        "plan_utilisation_factor": factor,
        "excavation_m3": excavation,
        "backfill_m3": backfill,
        "offsite_disposal_in_situ_m3": disposal_in_situ,
        "offsite_transport_loose_m3": disposal_loose,
        "offsite_haul_distance_km": float(parameters["offsite_haul_distance_km"]),
    }


def _foundation_work_packages(geometry: dict[str, float]) -> list[dict[str, Any]]:
    added_distance = max(geometry["offsite_haul_distance_km"] - 1.0, 0.0)
    specs = (
        ("OC-WP-3-1-1-10", geometry["excavation_m3"], 1.0, "braced_excavation"),
        ("OC-WP-3-1-1-8", geometry["offsite_transport_loose_m3"], 1.0, "first_transport_km"),
        ("OC-WP-3-1-2-4", geometry["offsite_transport_loose_m3"], added_distance, "additional_transport_km"),
        ("OC-WP-3-1-1-4", geometry["backfill_m3"], 1.0, "graded_backfill"),
        ("OC-WP-3-1-1-7", geometry["backfill_m3"], 1.0, "backfill_compaction"),
    )
    return [
        {
            "scope": "foundation_pit",
            "owner_type": "FoundationPitEarthwork",
            "variant": "reference_profile",
            "element_count": 1,
            "cpd_code": code,
            "work_quantity": quantity,
            "work_unit": "m3",
            "repetitions": repetitions,
            "quantity_basis": basis,
            "source_basis": "PUBLIC_PROJECT_FACT+MODEL_PARAMETER",
            "parameter_class": "scenario_parameter",
            "primary_material_code": None,
            "primary_quantity_per_work_unit": None,
            "primary_resource_unit": None,
        }
        for code, quantity, repetitions, basis in specs
    ]


def _evaluate_work_packages(
    work_packages: list[dict[str, Any]],
    recipes: dict[str, list[RecipeRow]],
    material_factors: dict[str, EfdFactor],
    equipment_factors: dict[str, EfdFactor],
    machine_shift_hours: float,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = {"A1-A3_ProductStage": 0.0, "A5_ConstructionStage": 0.0}
    resource_rows: list[dict[str, Any]] = []
    material_flows: list[dict[str, Any]] = []
    for wp_index, wp in enumerate(work_packages, start=1):
        code = wp["cpd_code"]
        if code not in recipes:
            raise ValueError(f"no recipe is available for {code}")
        wp_id = "WP_" + _safe_id(wp["scope"], wp["owner_type"], wp["variant"], code, wp_index)
        inputs: list[tuple[RecipeRow, float]] = [(recipe, recipe.measure_unit) for recipe in recipes[code]]
        if wp.get("primary_material_code"):
            inputs.append(
                (
                    RecipeRow(
                        cpd_code=code,
                        subproject="Enclosure",
                        entry_name=recipes[code][0].entry_name,
                        work_unit=wp["work_unit"],
                        measure_unit=1.0,
                        resource_kind="material",
                        resource_code=wp["primary_material_code"],
                        resource_quantity=float(wp["primary_quantity_per_work_unit"]),
                        resource_unit=wp["primary_resource_unit"],
                        resource_role="primary",
                        source_basis="REFERENCE_QUOTA_PRIMARY_FLOW",
                    ),
                    1.0,
                )
            )
        for input_index, (recipe, measure) in enumerate(inputs, start=1):
            actual_quantity = (
                float(wp["work_quantity"])
                / measure
                * recipe.resource_quantity
                * float(wp.get("repetitions", 1.0))
            )
            factors = material_factors if recipe.resource_kind == "material" else equipment_factors
            factor = factors.get(recipe.resource_code)
            if factor is None:
                raise ValueError(f"no EFD factor is available for {recipe.resource_code} ({code})")
            stage = "A1-A3_ProductStage" if recipe.resource_kind == "material" else "A5_ConstructionStage"
            factor_activity_quantity, factor_activity_unit, conversion_multiplier, conversion_basis = (
                _factor_activity(recipe, factor, actual_quantity, machine_shift_hours)
            )
            emission = factor_activity_quantity * factor.value
            totals[stage] += emission
            resource_id = "RI_" + _safe_id(wp_id, recipe.resource_code, input_index)
            row = {
                "work_package_id": wp_id,
                "resource_instance_id": resource_id,
                "scope": wp["scope"],
                "owner_type": wp["owner_type"],
                "variant": wp["variant"],
                "element_count": wp["element_count"],
                "cpd_code": code,
                "entry_name": recipe.entry_name,
                "work_quantity": wp["work_quantity"],
                "work_unit": wp["work_unit"],
                "measure_unit": measure,
                "repetitions": wp.get("repetitions", 1.0),
                "quantity_basis": wp["quantity_basis"],
                "resource_kind": recipe.resource_kind,
                "resource_code": recipe.resource_code,
                "resource_role": recipe.resource_role,
                "resource_quantity": actual_quantity,
                "resource_unit": recipe.resource_unit,
                "factor_activity_quantity": factor_activity_quantity,
                "factor_activity_unit": factor_activity_unit,
                "activity_conversion_multiplier": conversion_multiplier,
                "factor_application_basis": conversion_basis,
                "efd_entry_code": factor.entry_code,
                "efd_factor_value": factor.value,
                "efd_factor_unit": local_name(factor.unit_uri),
                "efd_factor_uri": factor.factor_uri,
                "efd_resource_uri": factor.resource_uri,
                "stage": stage,
                "emission_kgco2e": emission,
                "formula": (
                    "resource_quantity_machine_shift * machine_shift_hours * EFD_factor"
                    if conversion_basis == "MACHINE_SHIFT_TO_HOUR"
                    else "resource_quantity * EFD_factor"
                ),
                "source_basis": recipe.source_basis,
                "parameter_class": wp["parameter_class"],
            }
            resource_rows.append(row)
            if recipe.resource_kind == "material":
                material_flows.append(
                    {
                        "scope": wp["scope"],
                        "material_code": recipe.resource_code,
                        "quantity": actual_quantity,
                        "unit": recipe.resource_unit,
                        "source_basis": recipe.source_basis,
                    }
                )
    return totals, resource_rows, material_flows


def _normalise_qudt_unit(uri: str) -> str | None:
    return {
        "KiloGM": "kg",
        "TON_Metric": "t",
        "M3": "m3",
        "M2": "m2",
        "M": "m",
    }.get(local_name(uri))


def _main_material_flows(knowledge: KnowledgeIndex) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aliases = {
        "Resource_ProjectConcrete_C35_P8": "M-3-1",
        "Resource_20049": "M-8-12",
    }
    pattern = re.compile(r"Material_EFD_1_M_(\d+)_(\d+)$")
    connection = sqlite3.connect(knowledge.database_path)
    try:
        rows = connection.execute(
            """
            SELECT descriptor_uri, quantity_unit, SUM(quantity_value) AS quantity, COUNT(*) AS row_count
            FROM resources
            WHERE transport_status='PENDING_PROJECT_DISTANCE'
            GROUP BY descriptor_uri, quantity_unit
            ORDER BY descriptor_uri, quantity_unit
            """
        ).fetchall()
    finally:
        connection.close()
    flows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for descriptor_uri, quantity_unit, quantity, row_count in rows:
        descriptor = local_name(descriptor_uri)
        match = pattern.fullmatch(descriptor)
        material_code = f"M-{match.group(1)}-{match.group(2)}" if match else aliases.get(descriptor)
        unit = _normalise_qudt_unit(quantity_unit)
        if material_code is None or unit is None:
            exclusions.append(
                {
                    "descriptor_uri": descriptor_uri,
                    "quantity_unit": quantity_unit,
                    "quantity": quantity,
                    "row_count": row_count,
                    "reason": "no governed physical-material transport alias",
                }
            )
            continue
        flows.append(
            {
                "scope": "main_structure",
                "material_code": material_code,
                "quantity": float(quantity),
                "unit": unit,
                "source_basis": "CKG_RC4",
            }
        )
    coverage = {
        "pending_transport_resource_groups": len(rows),
        "mapped_resource_groups": len(flows),
        "excluded_resource_groups": len(exclusions),
        "mapping_rate": len(flows) / len(rows) if rows else 1.0,
        "exclusions": exclusions,
    }
    return flows, coverage


def _build_transport_legs(
    flows: list[dict[str, Any]],
    vehicles: dict[str, dict[str, Any]],
    materials: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    for flow in flows:
        default = materials.get(flow["material_code"])
        if default is None:
            exclusions.append({**flow, "reason": "material transport default is absent"})
            continue
        if not default["transport_applicable"]:
            exclusions.append({**flow, "reason": "transport is not applicable to this resource"})
            continue
        if flow["unit"] != default["source_unit"]:
            exclusions.append(
                {
                    **flow,
                    "reason": f"source unit {flow['unit']} differs from transport mass basis {default['source_unit']}",
                }
            )
            continue
        key = (flow["scope"], flow["material_code"], flow["unit"])
        if key not in grouped:
            grouped[key] = {**flow, "quantity": 0.0, "source_bases": set()}
        grouped[key]["quantity"] += float(flow["quantity"])
        grouped[key]["source_bases"].add(flow["source_basis"])

    rows: list[dict[str, Any]] = []
    totals: dict[str, float] = defaultdict(float)
    for (scope, material_code, unit), flow in sorted(grouped.items()):
        default = materials[material_code]
        vehicle = vehicles[default["vehicle_code"]]
        mass_t = flow["quantity"] * default["mass_t_per_source_unit"]
        distance = default["default_distance_km"]
        factor = vehicle["factor_kgco2e_per_tkm"]
        emission = mass_t * distance * factor
        totals[scope] += emission
        rows.append(
            {
                "scope": scope,
                "material_code": material_code,
                "material_name": default["material_name"],
                "source_quantity": flow["quantity"],
                "source_unit": unit,
                "mass_t": mass_t,
                "mass_t_per_source_unit": default["mass_t_per_source_unit"],
                "vehicle_code": default["vehicle_code"],
                "vehicle_uri": vehicle["vehicle_uri"],
                "vehicle_name": vehicle["vehicle_name"],
                "distance_km": distance,
                "factor_kgco2e_per_tkm": factor,
                "emission_kgco2e": emission,
                "stage": "A4_TransportationStage",
                "formula": "mass_t * distance_km * vehicle_factor_kgco2e_per_tkm",
                "vehicle_source_basis": default["vehicle_source_basis"],
                "distance_source_basis": default["distance_source_basis"],
                "mass_source_basis": default["mass_source_basis"],
                "parameter_class": default["parameter_class"],
                "quantity_source_basis": ";".join(sorted(flow["source_bases"])),
            }
        )
    return rows, dict(totals), exclusions


def _baseline_stage_totals(knowledge: KnowledgeIndex) -> dict[str, float]:
    connection = sqlite3.connect(knowledge.database_path)
    try:
        rows = connection.execute(
            """
            SELECT stage_uri, SUM(value)
            FROM summary_emissions
            WHERE owner_kind='assessment'
            GROUP BY stage_uri
            """
        ).fetchall()
    finally:
        connection.close()
    return {local_name(stage): float(value) for stage, value in rows}


def _write_reference_abox(
    path: Path,
    resource_rows: list[dict[str, Any]],
    transport_rows: list[dict[str, Any]],
    totals: dict[str, float],
    profile_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_uri = {
        "A1-A3_ProductStage": "msca:A1-A3_ProductStage",
        "A4_TransportationStage": "msca:A4_TransportationStage",
        "A5_ConstructionStage": "msca:A5_ConstructionStage",
    }
    unit_uri = {
        "m3": "unit:M3",
        "m2": "unit:M2",
        "t": "unit:TON_Metric",
        "kg": "unit:KiloGM",
        "m": "unit:M",
        "hour": "unit:HR",
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "@prefix msca: <http://bimlca.uni.edu/ontology#> .\n"
            "@prefix cpds: <http://bimlca.uni.edu/cpd/schema#> .\n"
            "@prefix int: <http://bimlca.uni.edu/integration/schema#> .\n"
            "@prefix recon: <http://bimlca.uni.edu/reconstructed/station01/reference-case#> .\n"
            "@prefix qudt: <http://qudt.org/schema/qudt/> .\n"
            "@prefix unit: <http://qudt.org/vocab/unit/> .\n"
            "@prefix prov: <http://www.w3.org/ns/prov#> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
            "recon:ReferenceCaseABox a owl:Ontology .\n"
            f"recon:Assessment a msca:CarbonAssessment ; rdfs:label {_literal(profile_id)} .\n"
            "recon:System_MainStructure a msca:MainStructureSystem .\n"
            "recon:System_Enclosure a msca:EnclosureStructureSystem .\n"
            "recon:System_FoundationPit a msca:FoundationPitSystem .\n"
            "recon:sourceBasis a rdf:Property .\n"
            "recon:parameterClass a rdf:Property .\n"
            "recon:transportDistanceKm a rdf:Property .\n"
            "recon:usesVehicle a rdf:Property .\n"
            "recon:hasFactorActivityQuantity a rdf:Property .\n"
            "recon:activityConversionMultiplier a rdf:Property .\n"
            "recon:Unit_MachineShift a qudt:Unit .\n\n"
        )
        work_packages: dict[str, dict[str, Any]] = {}
        for row in resource_rows:
            work_packages.setdefault(row["work_package_id"], row)
        for wp_id, row in sorted(work_packages.items()):
            owner_id = "ElementGroup_" + re.sub(r"[^A-Za-z0-9_]", "_", f"{row['scope']}_{row['owner_type']}_{row['variant']}")
            system = "System_Enclosure" if row["scope"] == "enclosure_structure" else "System_FoundationPit"
            quantity_id = "WorkQuantity_" + wp_id.removeprefix("WP_")
            decision_id = "CPDSelection_" + wp_id.removeprefix("WP_")
            work_unit = unit_uri.get(row["work_unit"], "recon:Unit_MachineShift")
            handle.write(
                f"recon:{owner_id} a msca:PhysicalElement ; msca:belongsToSystem recon:{system} .\n"
                f"recon:{wp_id} a msca:ProjectWorkPackage ; msca:constructsElement recon:{owner_id} ;\n"
                f"    msca:hasCPDSelectionDecision recon:{decision_id} ; msca:hasQuantity recon:{quantity_id} ;\n"
                f"    recon:sourceBasis {_literal(row['source_basis'])} ; recon:parameterClass {_literal(row['parameter_class'])} .\n"
                f"recon:{decision_id} a cpds:CPDSelectionDecision ; cpds:selectedOperationalEntry msca:Entry_{row['cpd_code']} ; int:cpdSelectionStatus \"SELECTED\" .\n"
                f"recon:{quantity_id} a qudt:QuantityValue ; qudt:numericValue \"{row['work_quantity']}\"^^xsd:decimal ; qudt:unit {work_unit} .\n\n"
            )
        for row in resource_rows:
            resource_id = row["resource_instance_id"]
            actual_id = "ActualQuantity_" + resource_id.removeprefix("RI_")
            emission_id = "CE_" + resource_id.removeprefix("RI_")
            emission_quantity_id = "EmissionQuantity_" + resource_id.removeprefix("RI_")
            factor_activity_id = "FactorActivityQuantity_" + resource_id.removeprefix("RI_")
            resource_unit = unit_uri.get(row["resource_unit"], "recon:Unit_MachineShift")
            factor_activity_unit = unit_uri.get(
                row["factor_activity_unit"], "recon:Unit_MachineShift"
            )
            handle.write(
                f"recon:{resource_id} a msca:ResourceInstance ; msca:describedBy <{row['efd_resource_uri']}> ;\n"
                f"    msca:consumedByWorkPackage recon:{row['work_package_id']} ; msca:hasActualQuantity recon:{actual_id} ;\n"
                f"    recon:hasFactorActivityQuantity recon:{factor_activity_id} ; msca:hasEmissions recon:{emission_id} .\n"
                f"recon:{actual_id} a qudt:QuantityValue ; qudt:numericValue \"{row['resource_quantity']}\"^^xsd:decimal ; qudt:unit {resource_unit} .\n"
                f"recon:{factor_activity_id} a qudt:QuantityValue ; qudt:numericValue \"{row['factor_activity_quantity']}\"^^xsd:decimal ;\n"
                f"    qudt:unit {factor_activity_unit} ; recon:activityConversionMultiplier \"{row['activity_conversion_multiplier']}\"^^xsd:decimal .\n"
                f"recon:{emission_id} a msca:CarbonEmission ; msca:calculatedFromResource recon:{resource_id} ;\n"
                f"    msca:calculatedUsingFactor <{row['efd_factor_uri']}> ; msca:belongsToStage {stage_uri[row['stage']]} ;\n"
                f"    msca:hasEmissionQuantity recon:{emission_quantity_id} .\n"
                f"recon:{emission_quantity_id} a qudt:QuantityValue ; qudt:numericValue \"{row['emission_kgco2e']}\"^^xsd:decimal ; qudt:unit msca:Unit_KgCO2e .\n\n"
            )
        for index, row in enumerate(transport_rows, start=1):
            suffix = _safe_id(row["scope"], row["material_code"], index)
            system = {
                "main_structure": "System_MainStructure",
                "enclosure_structure": "System_Enclosure",
                "foundation_pit": "System_FoundationPit",
            }[row["scope"]]
            handle.write(
                f"recon:A4WP_{suffix} a msca:ProjectWorkPackage ; msca:constructsElement recon:{system} ;\n"
                f"    recon:usesVehicle recon:Vehicle_{row['vehicle_code'].replace('-', '_')} ; recon:transportDistanceKm \"{row['distance_km']}\"^^xsd:decimal .\n"
                f"recon:A4RI_{suffix} a msca:ResourceInstance ; msca:consumedByWorkPackage recon:A4WP_{suffix} ; msca:hasEmissions recon:A4CE_{suffix} .\n"
                f"recon:A4CE_{suffix} a msca:CarbonEmission ; msca:calculatedFromResource recon:A4RI_{suffix} ;\n"
                f"    msca:belongsToStage msca:A4_TransportationStage ; msca:hasEmissionQuantity recon:A4EmissionQuantity_{suffix} .\n"
                f"recon:A4EmissionQuantity_{suffix} a qudt:QuantityValue ; qudt:numericValue \"{row['emission_kgco2e']}\"^^xsd:decimal ; qudt:unit msca:Unit_KgCO2e .\n\n"
            )
        for stage, value in totals.items():
            suffix = re.sub(r"[^A-Za-z0-9_]", "_", stage)
            handle.write(
                f"recon:Combined_{suffix} a msca:CarbonEmission ; msca:belongsToStage {stage_uri[stage]} ; msca:hasEmissionQuantity recon:CombinedQuantity_{suffix} .\n"
                f"recon:CombinedQuantity_{suffix} a qudt:QuantityValue ; qudt:numericValue \"{value}\"^^xsd:decimal ; qudt:unit msca:Unit_KgCO2e .\n"
                f"recon:Assessment msca:hasEmissions recon:Combined_{suffix} .\n\n"
            )


RESOURCE_FIELDS = [
    "work_package_id",
    "resource_instance_id",
    "scope",
    "owner_type",
    "variant",
    "element_count",
    "cpd_code",
    "entry_name",
    "work_quantity",
    "work_unit",
    "measure_unit",
    "repetitions",
    "quantity_basis",
    "resource_kind",
    "resource_code",
    "resource_role",
    "resource_quantity",
    "resource_unit",
    "factor_activity_quantity",
    "factor_activity_unit",
    "activity_conversion_multiplier",
    "factor_application_basis",
    "efd_entry_code",
    "efd_factor_value",
    "efd_factor_unit",
    "efd_factor_uri",
    "stage",
    "emission_kgco2e",
    "formula",
    "source_basis",
    "parameter_class",
]


ASSIGNMENT_FIELDS = [
    "global_id",
    "step_id",
    "ifc_entity",
    "name",
    "element_type",
    "variant",
    "volume_m3",
    "length_m",
    "area_m2",
    "mapping_status",
    "quantity_basis",
    "work_quantity",
    "work_quantity_unit",
    "work_package_codes",
    "source_basis",
    "parameter_class",
]


TRANSPORT_FIELDS = [
    "scope",
    "material_code",
    "material_name",
    "source_quantity",
    "source_unit",
    "mass_t",
    "mass_t_per_source_unit",
    "vehicle_code",
    "vehicle_uri",
    "vehicle_name",
    "distance_km",
    "factor_kgco2e_per_tkm",
    "emission_kgco2e",
    "stage",
    "formula",
    "vehicle_source_basis",
    "distance_source_basis",
    "mass_source_basis",
    "parameter_class",
    "quantity_source_basis",
]


def build_reference_case(config: ProjectConfig, knowledge: KnowledgeIndex | None = None) -> dict[str, Any]:
    reference_dir = config.reference_case_dir
    with (reference_dir / "case_profile.toml").open("rb") as handle:
        profile = tomllib.load(handle)
    recipes = _load_recipes(reference_dir / "cpd_work_package_recipes.csv")
    vehicles, transport_materials = _load_transport(reference_dir)
    material_factors, equipment_factors = load_efd_factors(config.efd_reference)
    machine_shift_hours = float(profile["activity_units"]["machine_shift_hours"])
    knowledge = knowledge or KnowledgeIndex(config)
    if not knowledge.is_current():
        knowledge.build()

    classification_path = config.artifact_dir / "abox" / "station01_enclosure_classification_register.csv"
    quantity_path = config.artifact_dir / "ifc" / "enclosure_structure_quantity_candidates.csv"
    if not classification_path.exists() or not quantity_path.exists():
        raise FileNotFoundError("run prepare QTO/ABox extraction before building the reference case")

    enclosure_wps, assignments, enclosure_coverage = _build_enclosure_work_packages(
        reference_dir, classification_path, quantity_path, profile
    )
    if enclosure_coverage["mapping_rate"] != 1.0 or enclosure_coverage["missing_quantity_elements"]:
        raise ValueError(f"enclosure mapping coverage is incomplete: {enclosure_coverage}")
    enclosure_totals, enclosure_resources, enclosure_material_flows = _evaluate_work_packages(
        enclosure_wps, recipes, material_factors, equipment_factors, machine_shift_hours
    )

    foundation_geometry = _foundation_geometry(profile)
    foundation_wps = _foundation_work_packages(foundation_geometry)
    foundation_totals, foundation_resources, foundation_material_flows = _evaluate_work_packages(
        foundation_wps, recipes, material_factors, equipment_factors, machine_shift_hours
    )

    main_material_flows, main_a4_coverage = _main_material_flows(knowledge)
    all_material_flows = main_material_flows + enclosure_material_flows + foundation_material_flows
    transport_rows, a4_by_scope, transport_exclusions = _build_transport_legs(
        all_material_flows, vehicles, transport_materials
    )
    if transport_exclusions:
        raise ValueError(f"physical material transport coverage is incomplete: {transport_exclusions}")
    baseline_totals = _baseline_stage_totals(knowledge)

    combined_totals = {
        "A1-A3_ProductStage": baseline_totals.get("A1-A3_ProductStage", 0.0)
        + enclosure_totals["A1-A3_ProductStage"]
        + foundation_totals["A1-A3_ProductStage"],
        "A4_TransportationStage": sum(a4_by_scope.values()),
        "A5_ConstructionStage": baseline_totals.get("A5_ConstructionStage", 0.0)
        + enclosure_totals["A5_ConstructionStage"]
        + foundation_totals["A5_ConstructionStage"],
    }

    low_geometry = _foundation_geometry(profile, float(profile["foundation_pit"]["low_plan_utilisation_factor"]))
    high_geometry = _foundation_geometry(profile, float(profile["foundation_pit"]["high_plan_utilisation_factor"]))
    base_volume = foundation_geometry["excavation_m3"]
    foundation_combined_base = (
        foundation_totals["A1-A3_ProductStage"]
        + foundation_totals["A5_ConstructionStage"]
        + a4_by_scope.get("foundation_pit", 0.0)
    )
    sensitivity = {
        "low": {
            "plan_utilisation_factor": low_geometry["plan_utilisation_factor"],
            "excavation_m3": low_geometry["excavation_m3"],
            "combined_kgco2e": foundation_combined_base * low_geometry["excavation_m3"] / base_volume,
        },
        "base": {
            "plan_utilisation_factor": foundation_geometry["plan_utilisation_factor"],
            "excavation_m3": foundation_geometry["excavation_m3"],
            "combined_kgco2e": foundation_combined_base,
        },
        "high": {
            "plan_utilisation_factor": high_geometry["plan_utilisation_factor"],
            "excavation_m3": high_geometry["excavation_m3"],
            "combined_kgco2e": foundation_combined_base * high_geometry["excavation_m3"] / base_volume,
        },
    }

    output_dir = config.artifact_dir / "reference_case"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "element_work_package_assignments.csv"
    enclosure_path = output_dir / "enclosure_cpd_efd_mapping.csv"
    foundation_path = output_dir / "foundation_pit_work_packages.csv"
    transport_path = output_dir / "a4_transport_legs.csv"
    abox_path = output_dir / "station01_reference_case_abox.ttl"
    _write_csv(assignment_path, assignments, ASSIGNMENT_FIELDS)
    _write_csv(enclosure_path, enclosure_resources, RESOURCE_FIELDS)
    _write_csv(foundation_path, foundation_resources, RESOURCE_FIELDS)
    _write_csv(transport_path, transport_rows, TRANSPORT_FIELDS)
    _write_reference_abox(
        abox_path,
        enclosure_resources + foundation_resources,
        transport_rows,
        combined_totals,
        profile["profile"]["id"],
    )

    def portable_artifact_path(path: Path) -> str:
        """Store release-relative POSIX paths so artifacts move across OSes."""

        return path.resolve().relative_to(config.project_root.resolve()).as_posix()

    sources = _read_csv(reference_dir / "source_register.csv")
    report = {
        "schema_version": "1.0",
        "profile": profile["profile"],
        "calculation_status": "COMPLETE_REFERENCE_CASE",
        "formulae": {
            "A1-A3": "resource_quantity * material_EFD_factor",
            "A4": "mass_t * default_distance_km * vehicle_factor_kgco2e_per_tkm",
            "A5": "CPD_machine_shifts * activity_unit_conversion * equipment_EFD_factor",
            "foundation_pit_volume": "station_length_m * station_width_m * station_depth_m * plan_utilisation_factor",
        },
        "public_geometry": profile["public_geometry"],
        "public_construction_context": profile["public_construction_context"],
        "parameters": {
            "foundation_pit": profile["foundation_pit"],
            "enclosure": profile["enclosure"],
            "activity_units": profile["activity_units"],
        },
        "main_structure": {
            "baseline_stage_totals_kgco2e": baseline_totals,
            "A4_kgco2e": a4_by_scope.get("main_structure", 0.0),
            "A4_mapping_coverage": main_a4_coverage,
        },
        "enclosure_structure": {
            "mapping_coverage": enclosure_coverage,
            "work_package_count": len(enclosure_wps),
            "resource_flow_count": len(enclosure_resources),
            "stage_totals_kgco2e": {
                **enclosure_totals,
                "A4_TransportationStage": a4_by_scope.get("enclosure_structure", 0.0),
            },
        },
        "foundation_pit": {
            "geometry": foundation_geometry,
            "work_package_count": len(foundation_wps),
            "resource_flow_count": len(foundation_resources),
            "stage_totals_kgco2e": {
                **foundation_totals,
                "A4_TransportationStage": a4_by_scope.get("foundation_pit", 0.0),
            },
            "sensitivity": sensitivity,
        },
        "combined_stage_totals_kgco2e": combined_totals,
        "combined_total_kgco2e": sum(combined_totals.values()),
        "transport": {
            "leg_count": len(transport_rows),
            "A4_by_scope_kgco2e": a4_by_scope,
            "exclusions": transport_exclusions,
            "material_default_count": len(transport_materials),
            "vehicle_factor_count": len(vehicles),
        },
        "provenance_vocabulary": {
            "source_basis": [
                "PUBLIC_PROJECT_FACT",
                "IFC_DERIVED",
                "CKG_RC4",
                "SOURCE_WORKBOOK",
                "REFERENCE_DISTANCE_PROFILE",
                "REFERENCE_QUOTA",
                "REFERENCE_QUOTA_PRIMARY_FLOW",
                "MODEL_PARAMETER",
            ],
            "parameter_class": ["observed", "inferred", "scenario_parameter", "fixed_reference_parameter"],
            "applicability_scope": profile["profile"]["applicability_scope"],
        },
        "sources": sources,
        "artifacts": {
            "element_work_package_assignments": portable_artifact_path(assignment_path),
            "enclosure_cpd_efd_mapping": portable_artifact_path(enclosure_path),
            "foundation_pit_work_packages": portable_artifact_path(foundation_path),
            "a4_transport_legs": portable_artifact_path(transport_path),
            "reference_case_abox": portable_artifact_path(abox_path),
        },
    }
    summary_path = output_dir / "reference_case_summary.json"
    report["artifacts"]["summary"] = portable_artifact_path(summary_path)
    atomic_write_json(summary_path, report)
    return report
