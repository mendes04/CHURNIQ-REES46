"""Presentation-only elements for CHURNIQ's business-first Streamlit interface.

Keep these helpers independent of the research pipeline. All dynamic values
are HTML-escaped, and no metric is fabricated or silently computed here.
"""
from __future__ import annotations

from html import escape


CSS = """
<style>
:root {
    --cq-bg: #090d1b;
    --cq-panel: #151a30;
    --cq-border: #313551;
    --cq-purple: #a388fa;
    --cq-text: #f4f2ff;
    --cq-muted: #aeb4cc;
}
[data-testid="stApp"] {
    background:
      radial-gradient(ellipse at 87% 0%, rgba(117,64,230,.16), transparent 34rem),
      radial-gradient(ellipse at 0% 80%, rgba(23,98,157,.10), transparent 39rem),
      var(--cq-bg);
    color: var(--cq-text);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#12172b 0%,#101426 100%);
    border-right: 1px solid rgba(176,166,245,.12);
}
.block-container {padding-top: 1.8rem;max-width:1490px;padding-bottom:3rem}
h1,h2,h3 {letter-spacing:-.025em}
h2 {font-weight:720}
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#1b213c,#181d34);
    border:1px solid #313755;
    border-radius:18px;padding:16px 18px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius:18px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {font-weight:570}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {gap: .28rem}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    border-radius: 12px;
    padding: .24rem .35rem;
}
.stButton>button {
    border-radius: 12px;
    min-height: 2.55rem;
    font-weight: 650;
    transition: border-color .15s ease, background .15s ease;
}
.stButton>button[kind="primary"] {
    background:linear-gradient(105deg,#7055dc,#9275fa);
    color:#fff;border:1px solid #a58dff;
}
.stButton>button:hover {border-color:#ad96ff}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:.35rem}
[data-testid="stTabs"] button {border-radius:9px 9px 0 0}

.cq-brand {padding:.35rem 0 1rem}
.cq-brand-title {font-size:1.36rem;font-weight:850;letter-spacing:.015em;color:#f6f2ff}
.cq-brand-title span {color:#b79dff}
.cq-brand-caption {font-size:.73rem;color:#a8abc9;letter-spacing:.08em;margin-top:.3rem}
.cq-side-note {margin:1rem 0 0;padding:14px;border:1px solid #383659;
    border-radius:14px;background:linear-gradient(145deg,rgba(91,71,161,.23),rgba(33,38,67,.4));}
.cq-side-note strong {font-size:.91rem;color:#fff}
.cq-side-note p {font-size:.79rem;line-height:1.48;color:#c4c6df;margin:.32rem 0 0}
.cq-hero {
    padding: 28px 30px;border-radius:23px;
    background:
      radial-gradient(circle at 89% 22%, rgba(174,128,255,.19), transparent 38%),
      linear-gradient(117deg,#25204d 0%,#1c2446 64%,#173747 100%);
    border:1px solid #51437b;margin:0 0 20px;
}
.cq-hero h1 {font-size:clamp(1.9rem,3vw,2.65rem);line-height:1.18;letter-spacing:-.04em;color:#fff;margin:.45rem 0}
.cq-hero p {font-size:1rem;line-height:1.6;color:#d9daf1;margin:.55rem 0 0;max-width:760px}
.cq-eyebrow {font-size:.73rem;letter-spacing:.15em;font-weight:850;color:#acffdc}
.cq-section {margin:1.3rem 0 .8rem}
.cq-section-title {font-size:1.25rem;letter-spacing:-.025em;color:#f6f4ff;font-weight:750}
.cq-section-description {font-size:.9rem;color:#aeb5ce;line-height:1.56;margin-top:.3rem}
.cq-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:12px 0 18px}
.cq-grid-3 {grid-template-columns:repeat(3,minmax(0,1fr))}
.cq-grid-2 {grid-template-columns:repeat(2,minmax(0,1fr))}
.cq-card {
    min-width:0;border-radius:18px;padding:19px 20px;
    background:linear-gradient(155deg,#1d223c 0%,#171b30 100%);
    border:1px solid #353858;
    box-shadow:0 7px 23px rgba(0,0,0,.09);
}
.cq-card[data-tone="purple"] {border-top:2px solid #b28dff}
.cq-card[data-tone="rose"] {border-top:2px solid #ee849f}
.cq-card[data-tone="teal"] {border-top:2px solid #5bd5c5}
.cq-card[data-tone="amber"] {border-top:2px solid #f0bf6f}
.cq-card-icon {font-size:1.25rem;line-height:1.15;margin-bottom:.68rem}
.cq-card-label {font-size:.83rem;font-weight:670;color:#cdd1e9;line-height:1.45}
.cq-card-value {font-size:clamp(1.45rem,2vw,2rem);font-weight:790;color:#fff;letter-spacing:-.04em;margin:.35rem 0 .18rem;line-height:1.2;overflow-wrap:anywhere}
.cq-card-title {font-size:.99rem;font-weight:760;color:#fff;margin:.08rem 0 .48rem;line-height:1.35}
.cq-card-description {font-size:.82rem;line-height:1.55;color:#b5bcd6}
.cq-inline-guide {padding:14px 17px;margin:13px 0;border-radius:15px;
    background:rgba(105,86,192,.12);border:1px solid rgba(163,145,244,.29);
    color:#dcdaf4;font-size:.88rem;line-height:1.55}
.cq-inline-guide strong {color:#fff}
.cq-mini-label {font-size:.75rem;font-weight:750;color:#a9efdd;letter-spacing:.09em}
@media(max-width:1100px) {.cq-grid {grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:680px) {
    .cq-grid,.cq-grid-3,.cq-grid-2 {grid-template-columns:1fr}
    .cq-hero{padding:22px 20px}
    .cq-hero h1 {font-size:1.8rem}
    .cq-card{padding:17px}
}
</style>
"""


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def hero(eyebrow: str, title: str, description: str) -> str:
    return (
        '<section class="cq-hero">'
        f'<div class="cq-eyebrow">{_safe(eyebrow)}</div>'
        f'<h1>{_safe(title)}</h1><p>{_safe(description)}</p>'
        '</section>'
    )


def section(title: str, description: str = "") -> str:
    return (
        '<div class="cq-section">'
        f'<div class="cq-section-title">{_safe(title)}</div>'
        + (f'<div class="cq-section-description">{_safe(description)}</div>'
           if description else "")
        + '</div>'
    )


def cards(items: list[dict], columns: int = 3, *, metric: bool = False) -> str:
    if columns not in (2, 3, 4):
        raise ValueError("Card columns must be 2, 3, or 4.")
    tiles = []
    for item in items:
        tone = item.get("tone", "purple")
        if tone not in {"purple", "rose", "teal", "amber"}:
            tone = "purple"
        icon = _safe(item.get("icon", "◆"))
        label = _safe(item["label"])
        desc = _safe(item.get("description", ""))
        if metric:
            contents = (
                f'<div class="cq-card-label">{label}</div>'
                f'<div class="cq-card-value">{_safe(item["value"])}</div>'
            )
        else:
            contents = f'<div class="cq-card-title">{label}</div>'
        tiles.append(
            f'<article class="cq-card" data-tone="{tone}">'
            f'<div class="cq-card-icon" aria-hidden="true">{icon}</div>'
            + contents
            + f'<div class="cq-card-description">{desc}</div>'
            + '</article>'
        )
    return f'<div class="cq-grid cq-grid-{columns}">' + "".join(tiles) + '</div>'


def guide(title: str, description: str) -> str:
    return (
        '<div class="cq-inline-guide">'
        f'<strong>{_safe(title)}</strong> {_safe(description)}'
        '</div>'
    )
