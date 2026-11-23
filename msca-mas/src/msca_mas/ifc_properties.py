from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .ifc import decode_ifc_string, iter_step_records, parse_enum, parse_ref, split_step_args
from .util import atomic_write_json


PROPERTY_TYPES = {
    "IFCPROPERTYSINGLEVALUE",
    "IFCPROPERTYENUMERATEDVALUE",
    "IFCPROPERTYLISTVALUE",
    "IFCPROPERTYBOUNDEDVALUE",
    "IFCPROPERTYREFERENCEVALUE",
    "IFCPROPERTYTABLEVALUE",
}
QUANTITY_TYPES = {
    "IFCQUANTITYLENGTH",
    "IFCQUANTITYAREA",
    "IFCQUANTITYVOLUME",
    "IFCQUANTITYCOUNT",
    "IFCQUANTITYWEIGHT",
    "IFCQUANTITYTIME",
}
QUANTITY_NAME_RE = re.compile(
    r"volume|area|length|height|width|depth|mass|weight|quantity|体积|面积|长度|高度|宽度|深度|质量|重量|工程量",
    re.IGNORECASE,
)
GENERIC_ENTITY_RE = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*;$", re.DOTALL)


class IfcPropertyAssertion(BaseModel):
    source_role: str
    source_file: str
    product_step_id: int
    pset_step_id: int
    pset_kind: str
    pset_name: str | None
    property_step_id: int
    property_name: str | None
    property_kind: str
    value_type: str | None
    value: str | float | bool | None
    unit_ref: int | None
    is_quantity_candidate: bool


class PropertyExtractionReport(BaseModel):
    source_role: str
    source_file: str
    product_count: int
    property_set_count: int
    quantity_set_count: int
    assertion_count: int
    quantity_candidate_count: int
    top_property_names: dict[str, int]
    assertions: list[IfcPropertyAssertion]

    def compact(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"assertions"})


def _refs(token: str) -> list[int]:
    return [int(value) for value in re.findall(r"#(\d+)", token)]


def _number(token: str) -> float | None:
    token = token.strip()
    if token in {"", "$", "*"}:
        return None
    typed = re.fullmatch(r"[A-Z0-9_]+\((.*)\)", token, re.DOTALL)
    if typed:
        token = typed.group(1).strip()
    try:
        return float(token)
    except ValueError:
        return None


def _typed_value(token: str) -> tuple[str | None, str | float | bool | None]:
    token = token.strip()
    if token in {"", "$", "*"}:
        return None, None
    typed = re.fullmatch(r"([A-Z0-9_]+)\((.*)\)", token, re.DOTALL)
    value_type = typed.group(1) if typed else None
    inner = typed.group(2).strip() if typed else token
    if inner.startswith("'"):
        return value_type or "STRING", decode_ifc_string(inner)
    enumeration = parse_enum(inner)
    if enumeration is not None:
        if enumeration in {"T", "TRUE"}:
            return value_type or "BOOLEAN", True
        if enumeration in {"F", "FALSE"}:
            return value_type or "BOOLEAN", False
        return value_type or "ENUM", enumeration
    number = _number(inner)
    if number is not None:
        return value_type or "NUMBER", number
    return value_type or "RAW", inner


def extract_properties(
    path: Path,
    role: str,
    product_step_ids: Iterable[int],
) -> PropertyExtractionReport:
    product_ids = set(product_step_ids)
    relations: list[tuple[list[int], int]] = []
    psets: dict[int, tuple[str, str | None, list[int]]] = {}
    properties: dict[int, tuple[str, str | None, str | None, str | float | bool | None, int | None]] = {}

    for statement in iter_step_records(path):
        match = GENERIC_ENTITY_RE.match(statement)
        if not match:
            continue
        step_id = int(match.group(1))
        entity = match.group(2)
        if entity not in PROPERTY_TYPES | QUANTITY_TYPES | {
            "IFCRELDEFINESBYPROPERTIES",
            "IFCPROPERTYSET",
            "IFCELEMENTQUANTITY",
        }:
            continue
        args = split_step_args(match.group(3))
        if entity == "IFCRELDEFINESBYPROPERTIES" and len(args) >= 6:
            pset_ref = parse_ref(args[5])
            if pset_ref:
                related = [value for value in _refs(args[4]) if value in product_ids]
                if related:
                    relations.append((related, pset_ref))
        elif entity == "IFCPROPERTYSET" and len(args) >= 5:
            psets[step_id] = ("PSET", decode_ifc_string(args[2]), _refs(args[4]))
        elif entity == "IFCELEMENTQUANTITY" and len(args) >= 6:
            psets[step_id] = ("QTO", decode_ifc_string(args[2]), _refs(args[5]))
        elif entity == "IFCPROPERTYSINGLEVALUE" and len(args) >= 3:
            value_type, value = _typed_value(args[2])
            properties[step_id] = (
                entity,
                decode_ifc_string(args[0]),
                value_type,
                value,
                parse_ref(args[3]) if len(args) > 3 else None,
            )
        elif entity in QUANTITY_TYPES and len(args) >= 4:
            properties[step_id] = (
                entity,
                decode_ifc_string(args[0]),
                entity.removeprefix("IFCQUANTITY"),
                _number(args[3]),
                parse_ref(args[2]),
            )
        elif entity in PROPERTY_TYPES:
            properties[step_id] = (entity, decode_ifc_string(args[0]) if args else None, "RAW", match.group(3), None)

    assertions: list[IfcPropertyAssertion] = []
    for related, pset_ref in relations:
        pset = psets.get(pset_ref)
        if not pset:
            continue
        pset_kind, pset_name, property_refs = pset
        for property_ref in property_refs:
            value = properties.get(property_ref)
            if not value:
                continue
            property_kind, property_name, value_type, actual_value, unit_ref = value
            quantity_candidate = bool(property_name and QUANTITY_NAME_RE.search(property_name)) or property_kind in QUANTITY_TYPES
            for product_step_id in related:
                assertions.append(
                    IfcPropertyAssertion(
                        source_role=role,
                        source_file=path.name,
                        product_step_id=product_step_id,
                        pset_step_id=pset_ref,
                        pset_kind=pset_kind,
                        pset_name=pset_name,
                        property_step_id=property_ref,
                        property_name=property_name,
                        property_kind=property_kind,
                        value_type=value_type,
                        value=actual_value,
                        unit_ref=unit_ref,
                        is_quantity_candidate=quantity_candidate,
                    )
                )
    names = Counter(item.property_name or "" for item in assertions)
    return PropertyExtractionReport(
        source_role=role,
        source_file=path.name,
        product_count=len(product_ids),
        property_set_count=sum(1 for kind, _, _ in psets.values() if kind == "PSET"),
        quantity_set_count=sum(1 for kind, _, _ in psets.values() if kind == "QTO"),
        assertion_count=len(assertions),
        quantity_candidate_count=sum(item.is_quantity_candidate for item in assertions),
        top_property_names=dict(names.most_common(50)),
        assertions=assertions,
    )


def write_property_report(report: PropertyExtractionReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / f"{report.source_role}_property_summary.json", report.compact())
    path = output_dir / f"{report.source_role}_quantity_candidates.csv"
    fields = list(IfcPropertyAssertion.model_fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for assertion in report.assertions:
            if assertion.is_quantity_candidate:
                writer.writerow(assertion.model_dump(mode="json"))
