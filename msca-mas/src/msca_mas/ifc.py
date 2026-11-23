from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, Field

from .config import IfcSource
from .util import atomic_write_json, sha256_file


# Keep opening geometry for downstream quantity deductions.
PRODUCT_TYPES = {
    "IFCBEAM",
    "IFCBUILDINGELEMENTPROXY",
    "IFCCOLUMN",
    "IFCCURTAINWALL",
    "IFCDOOR",
    "IFCFOOTING",
    "IFCMEMBER",
    "IFCOPENINGELEMENT",
    "IFCPILE",
    "IFCPLATE",
    "IFCRAILING",
    "IFCREINFORCINGBAR",
    "IFCREINFORCINGMESH",
    "IFCSLAB",
    "IFCSTAIR",
    "IFCSTAIRFLIGHT",
    "IFCWALL",
    "IFCWALLSTANDARDCASE",
    "IFCWINDOW",
}

ENTITY_RE = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*;$", re.DOTALL)


class IfcHeader(BaseModel):
    ifc_schema: str | None = None
    file_name: str | None = None
    timestamp: str | None = None
    originating_system: str | None = None


class IfcProduct(BaseModel):
    source_role: str
    source_file: str
    step_id: int
    ifc_entity: str
    global_id: str | None = None
    name: str | None = None
    description: str | None = None
    object_type: str | None = None
    placement_ref: int | None = None
    representation_ref: int | None = None
    tag: str | None = None
    predefined_type: str | None = None


class IfcSourceSummary(BaseModel):
    role: str
    path: str
    bytes: int
    sha256: str
    header: IfcHeader
    all_step_entities: int
    scoped_product_count: int
    unique_global_ids: int
    entity_counts: dict[str, int]
    product_type_counts: dict[str, int]


class IfcInventoryReport(BaseModel):
    sources: list[IfcSourceSummary]
    combined_product_count: int
    combined_unique_global_ids: int
    cross_file_duplicate_global_ids: list[str] = Field(default_factory=list)
    products: list[IfcProduct] = Field(default_factory=list)

    def compact(self) -> dict:
        return {
            "sources": [source.model_dump(mode="json") for source in self.sources],
            "combined_product_count": self.combined_product_count,
            "combined_unique_global_ids": self.combined_unique_global_ids,
            "cross_file_duplicate_global_id_count": len(self.cross_file_duplicate_global_ids),
            "cross_file_duplicate_global_ids": self.cross_file_duplicate_global_ids[:100],
        }


def split_step_args(text: str) -> list[str]:
    """Split STEP arguments while preserving nested lists and escaped strings."""
    values: list[str] = []
    start = 0
    depth = 0
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                in_string = False
            index += 1
            continue
        if char == "'":
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            values.append(text[start:index].strip())
            start = index + 1
        index += 1
    values.append(text[start:].strip())
    return values


def decode_ifc_string(token: str) -> str | None:
    token = token.strip()
    if token in {"", "$", "*"}:
        return None
    if not (token.startswith("'") and token.endswith("'")):
        return token
    value = token[1:-1].replace("''", "'")

    def decode_x2(match: re.Match[str]) -> str:
        try:
            return bytes.fromhex(match.group(1)).decode("utf-16-be")
        except (UnicodeError, ValueError):
            return match.group(0)

    def decode_x4(match: re.Match[str]) -> str:
        payload = match.group(1)
        try:
            return "".join(chr(int(payload[pos : pos + 8], 16)) for pos in range(0, len(payload), 8))
        except (ValueError, OverflowError):
            return match.group(0)

    value = re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\", decode_x2, value)
    value = re.sub(r"\\X4\\([0-9A-Fa-f]+)\\X0\\", decode_x4, value)

    # Read IFC2X3 as Latin-1 to preserve nonstandard UTF-8 bytes. Repair CJK only when decoding improves it.
    def cjk_count(text: str) -> int:
        return sum("\u3400" <= character <= "\u9fff" for character in text)

    try:
        utf8_candidate = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        utf8_candidate = value
    if cjk_count(utf8_candidate) > cjk_count(value):
        value = utf8_candidate
    return value


def parse_ref(token: str) -> int | None:
    match = re.fullmatch(r"#(\d+)", token.strip())
    return int(match.group(1)) if match else None


def parse_enum(token: str) -> str | None:
    match = re.fullmatch(r"\.([A-Z0-9_]+)\.", token.strip())
    return match.group(1) if match else None


def _statement_complete(text: str) -> bool:
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "'":
            if in_string and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            in_string = not in_string
        elif char == ";" and not in_string:
            return True
        index += 1
    return False


def iter_step_records(path: Path) -> Iterator[str]:
    """Yield entity statements without assuming that every STEP record is one line."""
    buffer: list[str] = []
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not buffer:
                if not stripped.startswith("#"):
                    continue
                buffer.append(stripped)
            else:
                buffer.append(stripped)
            candidate = " ".join(buffer)
            if _statement_complete(candidate):
                yield candidate[: candidate.rfind(";") + 1]
                buffer.clear()


def read_header(path: Path, max_bytes: int = 256 * 1024) -> IfcHeader:
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        text = handle.read(max_bytes)
    schema_match = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text, re.IGNORECASE | re.DOTALL)
    name_match = re.search(r"FILE_NAME\s*\((.*?)\)\s*;", text, re.IGNORECASE | re.DOTALL)
    file_name = timestamp = originating_system = None
    if name_match:
        args = split_step_args(name_match.group(1))
        file_name = decode_ifc_string(args[0]) if args else None
        timestamp = decode_ifc_string(args[1]) if len(args) > 1 else None
        originating_system = decode_ifc_string(args[5]) if len(args) > 5 else None
    return IfcHeader(
        ifc_schema=schema_match.group(1) if schema_match else None,
        file_name=file_name,
        timestamp=timestamp,
        originating_system=originating_system,
    )


def parse_product(statement: str, role: str, source_file: str) -> tuple[str, IfcProduct | None]:
    match = ENTITY_RE.match(statement)
    if not match:
        return "UNPARSED", None
    step_id = int(match.group(1))
    entity = match.group(2)
    if entity not in PRODUCT_TYPES:
        return entity, None
    args = split_step_args(match.group(3))
    extras = args[8:] if len(args) > 8 else []
    predefined = next((value for value in (parse_enum(item) for item in reversed(extras)) if value), None)
    return entity, IfcProduct(
        source_role=role,
        source_file=source_file,
        step_id=step_id,
        ifc_entity=entity,
        global_id=decode_ifc_string(args[0]) if args else None,
        name=decode_ifc_string(args[2]) if len(args) > 2 else None,
        description=decode_ifc_string(args[3]) if len(args) > 3 else None,
        object_type=decode_ifc_string(args[4]) if len(args) > 4 else None,
        placement_ref=parse_ref(args[5]) if len(args) > 5 else None,
        representation_ref=parse_ref(args[6]) if len(args) > 6 else None,
        tag=decode_ifc_string(args[7]) if len(args) > 7 else None,
        predefined_type=predefined,
    )


def scan_ifc(source: IfcSource) -> tuple[IfcSourceSummary, list[IfcProduct]]:
    entity_counts: Counter[str] = Counter()
    products: list[IfcProduct] = []
    for statement in iter_step_records(source.path):
        entity, product = parse_product(statement, source.role, source.path.name)
        entity_counts[entity] += 1
        if product is not None:
            products.append(product)
    product_counts = Counter(product.ifc_entity for product in products)
    global_ids = {product.global_id for product in products if product.global_id}
    summary = IfcSourceSummary(
        role=source.role,
        path=str(source.path),
        bytes=source.path.stat().st_size,
        sha256=sha256_file(source.path),
        header=read_header(source.path),
        all_step_entities=sum(entity_counts.values()),
        scoped_product_count=len(products),
        unique_global_ids=len(global_ids),
        entity_counts=dict(sorted(entity_counts.items())),
        product_type_counts=dict(sorted(product_counts.items())),
    )
    return summary, products


def build_inventory(sources: Iterable[IfcSource]) -> IfcInventoryReport:
    summaries: list[IfcSourceSummary] = []
    products: list[IfcProduct] = []
    ids_by_role: dict[str, set[str]] = {}
    for source in sources:
        summary, source_products = scan_ifc(source)
        summaries.append(summary)
        products.extend(source_products)
        ids_by_role[source.role] = {item.global_id for item in source_products if item.global_id}
    duplicate_ids: set[str] = set()
    roles = list(ids_by_role)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            duplicate_ids.update(ids_by_role[left] & ids_by_role[right])
    all_ids = {item.global_id for item in products if item.global_id}
    return IfcInventoryReport(
        sources=summaries,
        combined_product_count=len(products),
        combined_unique_global_ids=len(all_ids),
        cross_file_duplicate_global_ids=sorted(duplicate_ids),
        products=products,
    )


def write_inventory(report: IfcInventoryReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "ifc_inventory_summary.json", report.compact())
    fields = list(IfcProduct.model_fields)
    with (output_dir / "ifc_products.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for product in report.products:
            writer.writerow(product.model_dump(mode="json"))
