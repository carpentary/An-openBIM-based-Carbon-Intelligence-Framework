from pathlib import Path

import pytest
from rdflib import RDF, Graph, URIRef

from msca_mas.scenarios import extract_overlay
from msca_mas.transactions import TransactionService


def test_transaction_requires_exact_digest_and_writes_new_delta(tmp_path: Path):
    service = TransactionService(tmp_path)
    overlay = extract_overlay("Persist the project transport distance as 50 km.", persistent=True)
    preview = service.preview(overlay)
    assert preview is not None
    preview_path = service.save_preview(preview)
    with pytest.raises(PermissionError):
        service.commit(preview_path, "wrong-digest")

    output = service.commit(preview_path, preview.confirmation_digest)
    assert output.parent.name == "committed"
    assert preview.source_graph_mutated is False
    graph = Graph()
    graph.parse(output, format="turtle")
    assert len(graph) == 5
    assert (
        None,
        RDF.type,
        URIRef("http://bimlca.uni.edu/ontology#UserCorrection"),
    ) in graph
    with pytest.raises(FileExistsError):
        service.commit(preview_path, preview.confirmation_digest)
