from retrieval import SimulatedRetriever


def test_retriever_loads_sample_docs():
    retriever = SimulatedRetriever()
    assert "INV-001" in retriever.documents
    assert "CON-001" in retriever.documents
    assert "CLM-001" in retriever.documents


def test_keyword_search_finds_invoice():
    retriever = SimulatedRetriever()
    hits = retriever.retrieve_by_keyword("Acme invoice")
    assert hits
    assert any(hit.doc_id == "INV-001" for hit in hits)


def test_amount_over_filter():
    retriever = SimulatedRetriever()
    hits = retriever.retrieve_by_amount_range(min_amount=50000)
    ids = {hit.doc_id for hit in hits}
    assert "INV-002" in ids
    assert "INV-003" in ids
    assert "CLM-001" not in ids


def test_get_document_by_id():
    retriever = SimulatedRetriever()
    doc = retriever.get_document_by_id("CON-001")
    assert doc is not None
    assert "Healthcare Partners" in doc.content
