from __future__ import annotations

import re
from typing import Any, Mapping


TRANSPORT_TARGET_PREFIX = "material_transport_default:"


def canonical_entity_label(value: Any) -> str:
    """Normalise presentation differences without fuzzy or keyword matching."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def entity_label_tokens(value: Any) -> set[str]:
    """Return conservative unordered tokens for a secondary unique-match check."""

    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        token = raw
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def canonical_query_entity_label(query: str, value: Any) -> str:
    """Collapse a full-name/parenthetical acronym pair to one grounded key."""

    key = canonical_entity_label(value)
    aliases: dict[str, str] = {}
    pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9]*(?:[ -]+[A-Z][A-Za-z0-9]*){1,5})\s*"
        r"\(([A-Z][A-Z0-9-]{1,11})\)"
    )
    for match in pattern.finditer(query):
        full_name = match.group(1)
        acronym = match.group(2)
        initials = "".join(
            token[0]
            for token in re.findall(r"[A-Za-z0-9]+", full_name)
            if token
        ).casefold()
        acronym_key = canonical_entity_label(acronym)
        if initials != acronym_key:
            continue
        full_key = canonical_entity_label(full_name)
        aliases[full_key] = full_key
        aliases[acronym_key] = full_key
    return aliases.get(key, key)


def transport_material_labels(
    target: str,
    query_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """Collect material mentions already grounded by the semantic contract."""

    contract = query_contract or {}
    labels: list[str] = []
    if target.casefold().startswith(TRANSPORT_TARGET_PREFIX):
        labels.append(target[len(TRANSPORT_TARGET_PREFIX) :])
    labels.extend(str(value) for value in contract.get("resource_targets", []))
    labels.extend(str(value) for value in contract.get("explicit_identifiers", []))
    labels.extend(
        str(item.get("target"))
        for item in contract.get("parameter_mentions", [])
        if item.get("target")
    )
    return list(dict.fromkeys(value.strip() for value in labels if value.strip()))


def transport_default_row_keys(row: Mapping[str, Any]) -> set[str]:
    material_name = str(row.get("material_name", ""))
    keys = {
        canonical_entity_label(row.get("material_code")),
        canonical_entity_label(material_name),
    }
    parenthetical = re.search(r"\(([^)]+)\)", material_name)
    if parenthetical:
        keys.add(canonical_entity_label(parenthetical.group(1)))
        keys.add(
            canonical_entity_label(
                re.sub(r"\s*\([^)]+\)\s*", " ", material_name)
            )
        )
    keys.discard("")
    return keys


def matching_transport_defaults(
    rows: list[dict[str, Any]],
    labels: list[str],
    *,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most-specific unique controlled-label match.

    A semantic contract may legitimately contain both a specific material name
    and a broader category (for example, a concrete grade and ``concrete``).
    Treating all labels as an unordered union lets the broad category turn one
    exact intent into many candidates.  Resolve labels independently from the
    most informative token set to the least informative one and accept only a
    unique match.  Zero and genuinely ambiguous matches remain unresolved.
    """

    requested_keys = {canonical_entity_label(value) for value in labels}
    requested_keys.discard("")
    if not requested_keys:
        return []
    exact = [row for row in rows if requested_keys & transport_default_row_keys(row)]
    if exact:
        return exact

    requested_token_sets = sorted(
        {
            frozenset(entity_label_tokens(value))
            for value in labels
            if entity_label_tokens(value)
        },
        key=lambda tokens: (-len(tokens), sorted(tokens)),
    )
    ambiguous: list[dict[str, Any]] = []
    for requested_tokens in requested_token_sets:
        contained: list[dict[str, Any]] = []
        for row in rows:
            row_tokens = entity_label_tokens(row.get("material_name"))
            if not row_tokens:
                continue
            if row_tokens <= requested_tokens or requested_tokens <= row_tokens:
                contained.append(row)
        if len(contained) == 1:
            return contained
        if contained and not ambiguous:
            ambiguous = contained
    if query:
        query_tokens = entity_label_tokens(query)
        catalogue_grounded = [
            row
            for row in rows
            if entity_label_tokens(row.get("material_name"))
            and entity_label_tokens(row.get("material_name")) <= query_tokens
        ]
        if catalogue_grounded:
            maximum_specificity = max(
                len(entity_label_tokens(row.get("material_name")))
                for row in catalogue_grounded
            )
            most_specific = [
                row
                for row in catalogue_grounded
                if len(entity_label_tokens(row.get("material_name")))
                == maximum_specificity
            ]
            return most_specific
    return ambiguous
