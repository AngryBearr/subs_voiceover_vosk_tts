from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from utils import shortening_html_view as view
from utils.shortening_html_view import render_report


MALICIOUS = '</script><img src=x onerror="alert(1)">'
PRIVATE_RAW_MARKER = "PRIVATE_RAW_REASONING_SHOULD_NOT_BE_EXPORTED"
EXPECTED_CSP = (
    "default-src 'none'; connect-src 'none'; img-src 'none'; "
    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'"
)
REQUIRED_SHELL_IDS = {
    "rows",
    "none-found",
    "pager",
    "pg-prev",
    "pg-next",
    "filter-text",
    "filter-reset",
    "count-shown",
    "count-unique",
    "count-range",
    "page-state",
}
FILTER_FIELDS = ("generator", "arbiter", "repetition", "status", "source")


class _HTMLProbe(HTMLParser):
    """Small stdlib-only structural probe; it does not execute the report JS."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str]]] = []
        self.scripts: list[str] = []
        self.text: list[str] = []
        self.list_items: list[tuple[dict[str, str], str]] = []
        self._script: list[str] | None = None
        self._list_item: list[str] | None = None
        self._list_item_attrs: dict[str, str] | None = None

    def _record_element(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, {name: value or "" for name, value in attrs}))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record_element(tag, attrs)
        if tag == "script":
            self._script = []
        elif tag == "li":
            self._list_item = []
            self._list_item_attrs = {name: value or "" for name, value in attrs}

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record_element(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None
        elif tag == "li" and self._list_item is not None:
            self.list_items.append((self._list_item_attrs or {}, "".join(self._list_item)))
            self._list_item = None
            self._list_item_attrs = None

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self._script is not None:
            self._script.append(data)
        if self._list_item is not None:
            self._list_item.append(data)


def _probe(html: str) -> _HTMLProbe:
    parsed = _HTMLProbe()
    parsed.feed(html)
    parsed.close()
    return parsed


def _attrs(probe: _HTMLProbe, tag: str) -> list[dict[str, str]]:
    return [attributes for element, attributes in probe.elements if element == tag]


def _candidate_row(accepted_text: str | None = "accepted") -> dict[str, Any]:
    alternatives = [
        {
            "id": f"alternative-{index}",
            "phase": "alternative",
            "text": "duplicate candidate text",
            "status": "candidate",
            "length": 24,
            "max_chars": 40,
            "fits": True,
            "selected": index == 3,
        }
        for index in range(5)
    ]
    return {
        "case_id": "security-case",
        "source": MALICIOUS,
        "generator": MALICIOUS,
        "generator_mode": "medium",
        "arbiter": MALICIOUS,
        "arbiter_mode": "off",
        "decision_kind": "arbiter",
        "status": "accepted",
        "original": MALICIOUS,
        "initial_verdict": "revise",
        "verdict": "accept",
        "accepted_text": accepted_text,
        "candidates": [
            {
                "id": "initial",
                "phase": "initial",
                "text": MALICIOUS,
                "status": "candidate",
                "fits": False,
            },
            {
                "id": "repair",
                "phase": "repair",
                "text": "repair proposal",
                "status": "candidate",
                "fits": True,
            },
            *alternatives,
        ],
        "issues": ["safe issue"],
        "advice": "safe advice",
        "raw": PRIVATE_RAW_MARKER,
        "reasoning": PRIVATE_RAW_MARKER,
        "prompt": PRIVATE_RAW_MARKER,
    }


def test_render_report_escapes_dynamic_values_and_keeps_safe_contract_only() -> None:
    document: dict[str, Any] = {
        "title": MALICIOUS,
        "run_id": MALICIOUS,
        "status": "accepted",
        "metadata": [{"label": MALICIOUS, "value": MALICIOUS}],
        "rows": [
            _candidate_row(MALICIOUS),
            {**_candidate_row(None), "case_id": "null-case", "candidates": []},
        ],
        "raw_reasoning": PRIVATE_RAW_MARKER,
        "chain_of_thought": PRIVATE_RAW_MARKER,
        "request_headers": PRIVATE_RAW_MARKER,
        "ledger": PRIVATE_RAW_MARKER,
    }

    html = render_report(document)
    probe = _probe(html)

    escaped_payload = escape(MALICIOUS, quote=True)
    assert MALICIOUS not in html
    assert escaped_payload in html
    assert MALICIOUS in "".join(probe.text)
    assert "repair proposal" in "".join(probe.text)
    assert any("Первый вариант" in text and MALICIOUS in text for _, text in probe.list_items)
    assert any("Исправление" in text and "repair proposal" in text for _, text in probe.list_items)
    assert "режим: medium" in "".join(probe.text)
    assert "режим: off" in "".join(probe.text)
    assert any(attributes.get("data-source") == MALICIOUS for _, attributes in probe.elements)
    assert any(attributes.get("data-generator") == MALICIOUS for _, attributes in probe.elements)
    assert any(attributes.get("data-arbiter") == MALICIOUS for _, attributes in probe.elements)
    assert "Принято" in html
    assert "Нет принятого варианта" in html
    assert PRIVATE_RAW_MARKER not in html

    allowed_tags = {
        "html", "head", "meta", "title", "style", "body", "header", "h1", "div",
        "span", "section", "h2", "dl", "dt", "dd", "p", "ul", "li",
        "details", "summary", "table", "caption", "colgroup", "col", "thead", "tbody",
        "tr", "th", "td", "input", "select", "option", "button", "nav", "script",
    }
    assert {tag for tag, _ in probe.elements} <= allowed_tags
    assert not {tag for tag, _ in probe.elements} & {"img", "svg", "iframe", "object", "embed"}
    assert len(probe.scripts) == 1
    assert probe.scripts == [view._SCRIPT]
    assert len(_attrs(probe, "style")) == 1
    assert not _attrs(probe, "link")
    assert len(_attrs(probe, "meta")) >= 2
    csp = [
        attributes for attributes in _attrs(probe, "meta")
        if attributes.get("http-equiv", "").lower() == "content-security-policy"
    ]
    assert len(csp) == 1
    assert csp[0]["content"] == EXPECTED_CSP
    for _, attributes in probe.elements:
        assert not any(name.lower().startswith("on") for name in attributes)
        assert "src" not in attributes
        assert "href" not in attributes
    assert "http://" not in html
    assert "https://" not in html
    assert "@import" not in html.lower()

    selected_items = [
        text for attributes, text in probe.list_items
        if "alternative-3" in text and "выбран" in text
    ]
    assert len(selected_items) == 2  # proposal column and the expanded all-candidates block
    for index in range(5):
        candidate_items = [
            text for _, text in probe.list_items if f"alternative-{index}" in text
        ]
        assert len(candidate_items) == 2
        assert all("duplicate candidate text" in text for text in candidate_items)
        if index != 3:
            assert all("выбран" not in text for text in candidate_items)


def _bulk_row(index: int) -> dict[str, Any]:
    return {
        "case_id": f"case-{index // 2}",
        "source": f"source-{index % 2}",
        "generator": f"generator-{index % 2}",
        "generator_mode": "medium",
        "arbiter": f"arbiter-{index % 2}",
        "arbiter_mode": "off",
        "decision_kind": "arbiter",
        "status": "accepted" if index % 2 == 0 else "rejected",
        "original": f"original-{index}",
        "repetition": 1 + (index % 2),
        "accepted_text": f"final-{index}" if index % 2 == 0 else None,
        "candidates": [{
            "id": "initial",
            "phase": "initial",
            "text": f"proposal-{index}",
            "status": "candidate",
            "fits": True,
        }],
    }


def test_large_report_emits_all_rows_and_static_paging_filter_contract() -> None:
    html = render_report({"title": "bulk", "run_id": "bulk-run", "rows": [_bulk_row(i) for i in range(120)]})
    probe = _probe(html)
    script = probe.scripts[0]
    main_rows = [
        attributes for tag, attributes in probe.elements
        if tag == "tr"
        and "row" in attributes.get("class", "").split()
        and "details-row" not in attributes.get("class", "").split()
    ]
    detail_rows = [
        attributes for tag, attributes in probe.elements
        if tag == "tr" and "details-row" in attributes.get("class", "").split()
    ]
    assert len(main_rows) == 120
    assert len(detail_rows) == 120
    assert {attributes["data-case"] for attributes in main_rows} == {
        f"case-{index}" for index in range(60)
    }
    assert all("js-only" not in attributes.get("class", "") for attributes in main_rows + detail_rows)
    assert '<span id="count-shown">120</span>' in html
    assert '<span id="count-unique">60</span>' in html
    assert '<span id="count-range">1–120</span>' in html
    assert 'id="pg-prev"' in html and 'id="pg-next"' in html
    assert "Назад" in html and "Вперёд" in html and "Страница 1 из 1" in html
    assert "PER_PAGE=50" in script

    body = _attrs(probe, "body")[0]
    filter_params = dict(parse_qsl(body["data-filter-params"], keep_blank_values=True))
    assert filter_params == {
        "search": "filter-text",
        "generator": "filter-generator",
        "arbiter": "filter-arbiter",
        "repetition": "filter-repetition",
        "status": "filter-status",
        "source": "filter-source",
    }
    for control_id in (*REQUIRED_SHELL_IDS, *(f"filter-{field}" for field in FILTER_FIELDS)):
        assert any(attributes.get("id") == control_id for _, attributes in probe.elements)
    for control_id in REQUIRED_SHELL_IDS:
        assert f"q('{control_id}')" in script
    for field in FILTER_FIELDS:
        assert f'data-{field}="' in html
        assert f'id="{filter_params[field]}"' in html
    assert "url.forEach(function(id,field)" in script
    assert "var el=q(id)" in script
    assert "if(el){inputs.push([field,el])}" in script
    assert "tr.dataset[kind]" in script
    # Structural contract only: this test intentionally does not execute JavaScript.


def test_empty_report_has_every_unconditional_script_control_and_no_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    html = render_report({"title": "empty", "rows": []})
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    probe = _probe(html)

    assert before == after
    emitted_ids = {
        attributes["id"] for _, attributes in probe.elements if "id" in attributes
    }
    assert REQUIRED_SHELL_IDS <= emitted_ids
    assert not [
        attributes for tag, attributes in probe.elements
        if tag == "tr" and "row" in attributes.get("class", "").split()
    ]
    assert '<span id="count-shown">0</span>' in html
    assert '<span id="count-unique">0</span>' in html
    assert '<span id="count-range">—</span>' in html
    assert len(probe.scripts) == 1
    assert probe.scripts == [view._SCRIPT]
    assert "q('filter-reset').addEventListener" in probe.scripts[0]
    assert "q('filter-text').addEventListener" in probe.scripts[0]


def test_evaluated_row_uses_nonselection_status_and_accepted_block() -> None:
    row = _candidate_row("not-selected-text")
    row.update({"status": "evaluated", "decision_kind": "evaluation"})

    html = render_report({"title": "evaluation", "rows": [row]})

    assert "Оценён, без выбора" in html
    assert "Не выбирался: только оценка" in html
    assert '<div class="accepted ok">' not in html
