"""
tests/test_references.py — Unit tests for ReferenceManager (no IBM API required).
"""
import pytest
from research_agent.literature_search import Paper
from research_agent.references import ReferenceManager


def _make_paper(title="Test Paper", year=2024, authors=None, doi=None):
    return Paper(
        title=title,
        authors=authors or ["Alice Smith", "Bob Jones"],
        year=year,
        abstract="This is a test abstract.",
        url="https://arxiv.org/abs/0000.00000",
        source="arxiv",
        doi=doi,
    )


def test_add_paper():
    rm = ReferenceManager()
    key = rm.add(_make_paper())
    assert key  # non-empty cite key
    assert rm.count() == 1


def test_add_deduplication():
    rm = ReferenceManager()
    p1 = _make_paper(title="Alpha Paper")
    p2 = _make_paper(title="Alpha Paper")  # same title → different cite key suffix
    k1 = rm.add(p1)
    k2 = rm.add(p2)
    assert rm.count() == 2
    assert k1 != k2


def test_bibtex_export():
    rm = ReferenceManager()
    rm.add(_make_paper(doi="10.1234/test"))
    bib = rm.export_bibtex()
    assert "@article" in bib
    assert "10.1234/test" in bib


def test_apa_export():
    rm = ReferenceManager()
    rm.add(_make_paper(year=2023))
    apa = rm.export_apa()
    assert "2023" in apa
    assert "Test Paper" in apa


def test_json_export():
    import json
    rm = ReferenceManager()
    rm.add(_make_paper())
    data = json.loads(rm.export_json())
    assert len(data) == 1
    assert data[0]["title"] == "Test Paper"


def test_cite_key_format():
    rm = ReferenceManager()
    paper = _make_paper(authors=["Zhang Wei"], year=2022, title="Quantum Dynamics Study")
    key = rm.add(paper)
    assert "2022" in key
    assert "Zhang" in key or "Wei" in key or "quantum" in key.lower()
