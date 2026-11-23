"""Tool restrictions for schema-only operation."""

from __future__ import annotations

from typing import Any, Callable

from .tooling import ToolRegistry

def restrict_ontology_capabilities(
    registry: ToolRegistry,
    *,
    ontology_mode: str,
) -> ToolRegistry:
    """Apply the schema-only policy without changing the tool interface."""

    if ontology_mode != "schema_only":
        return registry
    return (
        registry.with_wrapped("map_ontology_entities", _flat_entity_mapping)
        .with_wrapped("hybrid_reference_lookup", _lexical_reference_lookup)
        .with_wrapped("read_element_trace", _flat_element_trace)
        .with_wrapped("parse_query_contract", _profile_free_contract)
    )


def _flat_entity_mapping(handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return mention text without ontology identifiers."""

    def wrapped(arguments: Any) -> Any:
        pointers = handler(arguments)
        flat: list[dict[str, Any]] = []
        for pointer in pointers or []:
            if hasattr(pointer, "model_dump"):
                pointer = pointer.model_dump(mode="json")
            flat.append(
                {
                    "mention": pointer.get("mention"),
                    "uri": "",
                    "label": pointer.get("mention"),
                    "confidence": pointer.get("confidence"),
                    "evidence": "flat_lexical_surface_match",
                }
            )
        return flat

    return wrapped


def _lexical_reference_lookup(handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Limit reference ranking to BM25 and exact-code matches."""

    def wrapped(arguments: Any) -> Any:
        patched = arguments.model_copy(update={"lexical_only": True})
        return handler(patched)

    return wrapped


def _flat_element_trace(handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return trace rows without semantic path annotations."""

    def wrapped(arguments: Any) -> Any:
        result = handler(arguments)
        if isinstance(result, dict):
            result = dict(result)
            result.pop("semantic_route", None)
            result["row_projection"] = "flat"
        return result

    return wrapped


def _profile_free_contract(handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Remove ontology profile and path requirements from the contract."""

    def wrapped(arguments: Any) -> Any:
        contract = handler(arguments)
        if isinstance(contract, dict):
            contract = dict(contract)
            contract["planner_profile_ids"] = []
            contract["ontology_obligations"] = []
            contract["ontology_mode"] = "schema_only"
        return contract

    return wrapped
