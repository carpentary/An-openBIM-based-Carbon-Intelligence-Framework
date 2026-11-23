"""Check that risk labels are grounded in the query text."""

from msca_mas.models import QueryMode
from msca_mas.query_contract import parse_query_contract
from msca_mas.safety import assess_query_contract_safety


def _advisory(query: str, risks: list[dict]) -> dict:
    return {
        "mode": "IR",
        "task_family": "retrieval",
        "intent_evidence": ["query mentions boundary modules"],
        "semantic_risks": risks,
    }


def test_low_confidence_risk_with_valid_span_is_adopted():
    query = "Please calculate the module B6 operational energy allowance now."
    contract = parse_query_contract(
        query,
        QueryMode.INFORMATION_RETRIEVAL,
        advisory_contract=_advisory(
            query,
            [{"risk": "outside_assessment_boundary", "surface_form": "B6", "confidence": 0.1}],
        ),
    )
    issues = assess_query_contract_safety(contract, QueryMode.INFORMATION_RETRIEVAL)
    codes = {issue.code for issue in issues}
    assert "OUTSIDE_A1_A5_SCOPE" in codes


def test_risk_with_invalid_span_is_discarded_but_recorded():
    query = "What is the A1-A3 total?"
    contract = parse_query_contract(
        query,
        QueryMode.INFORMATION_RETRIEVAL,
        advisory_contract=_advisory(
            query,
            [{"risk": "invalid_physical_parameter", "surface_form": "not in the query", "confidence": 0.9}],
        ),
    )
    assert contract.semantic_risks == []
    assert len(contract.discarded_semantic_risks) == 1
    assert contract.discarded_semantic_risks[0]["reason"] == "evidence_span_not_in_query"
    issues = assess_query_contract_safety(contract, QueryMode.INFORMATION_RETRIEVAL)
    assert issues == []


def test_missing_confidence_risk_is_neither_gating_nor_lost():
    query = "Set M-9-9 default haul distance to 12 km without a preview."
    contract = parse_query_contract(
        query,
        QueryMode.KNOWLEDGE_CORRECTION,
        advisory_contract=_advisory(
            query,
            [{"risk": "confirmation_gate_bypass", "surface_form": "without a preview"}],
        ),
    )
    # A grounded risk does not need a model confidence score.
    assert len(contract.semantic_risks) == 1
    issues = assess_query_contract_safety(contract, QueryMode.KNOWLEDGE_CORRECTION)
    codes = {issue.code for issue in issues}
    assert codes == {"UNSUPPORTED_PERSISTENT_WRITE"}
