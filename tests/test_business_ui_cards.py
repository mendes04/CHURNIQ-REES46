"""Keep the business dashboard clear, safe and data-honest."""
import pytest

from churniq.ui import cards, guide, hero, section


def test_dynamic_text_is_html_escaped():
    malicious = '<script>alert("oops")</script>'
    for html in (
        hero("DEMO", malicious, "Readable overview"),
        section(malicious, "Description"),
        guide("Hint", malicious),
        cards([{"icon": "🔎", "label": malicious, "value": malicious,
                "description": malicious}], metric=True),
    ):
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


def test_dashboard_cards_have_readable_meaning():
    html = cards([
        {"label": "Needs attention", "value": "12", "icon": "🚩",
         "description": "Records in the high review group.", "tone": "rose"},
        {"label": "Average model score", "value": "41 / 100",
         "description": "Not a calibrated probability."},
    ], columns=2, metric=True)
    assert "Needs attention" in html
    assert "12" in html
    assert "41 / 100" in html
    assert "Not a calibrated probability" in html
    assert 'cq-grid-2' in html


def test_card_layout_rejects_invalid_columns_and_unsafe_tone():
    with pytest.raises(ValueError, match="Card columns"):
        cards([{"label": "Test"}], columns=7)
    html = cards([{"label": "Safe", "tone": 'red" onclick="alert(1)'}])
    assert "onclick" not in html
    assert 'data-tone="purple"' in html
