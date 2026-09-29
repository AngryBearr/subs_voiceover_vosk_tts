"""Pure offline HTML comparison view for subtitle shortening runs.

``render_report`` turns the shared report document (built by the IO adapter)
into one self-contained UTF-8 HTML string: no filesystem, network, subprocess,
or model calls. All dynamic values are escaped with :func:`html.escape`, and
the only script is static: it filters already-rendered rows through data
attributes and never receives user data as code.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

HTML_MARKER = "<!-- subs-shortening-html:v1 -->"
PAGE_TITLE = "Сравнение сокращений"

_LABEL_FALLBACK = "—"

_STATUS_LABELS: dict[str, str] = {
    "accepted": "Принято",
    "rejected": "Отклонено",
    "unresolved": "Не решено",
    "pending": "Ожидает",
    "error": "Ошибка",
    "blocked": "Заблокировано",
    "cannot_shorten": "Модель не сократила",
    "zero": "Нулевой лимит",
    "invalid": "Некорректный случай",
    "not_reviewed": "Без проверки",
    "evaluated": "Оценён, без выбора",
    # Saved runner artifacts may carry "manual"; the IO adapter maps it to
    # pending/unresolved, so this label only keeps raw tokens out of the UI.
    "manual": "Ручная проверка",
}

_DECISION_LABELS: dict[str, str] = {
    "arbiter": "Арбитр",
    "selector": "Выбор из вариантов",
    "unreviewed": "Без арбитра",
    "evaluation": "Оценка варианта",
}

_VERDICT_LABELS: dict[str, str] = {
    # Review-arbiter vocabulary ("accept"|"revise"|"impossible") and the
    # selector vocabulary ("accept"|"reject") from the saved artifacts.
    "accept": "принято",
    "reject": "отклонено",
    "revise": "нужно исправление",
    "impossible": "сокращение невозможно",
    "accepted": "принято",
    "rejected": "отклонено",
    "unresolved": "не решено",
    "pending": "ожидает",
    "error": "ошибка",
    "cannot_shorten": "сокращение невозможно",
    "zero": "нулевой лимит",
    "invalid": "некорректно",
    "not_reviewed": "не проверялось",
}

# Candidate phase wording: "repair" is an internal retry with feedback, so it
# must not read as the run-level repetition filter named "повтор".
_PHASE_LABELS: dict[str, str] = {
    "initial": "Первый вариант",
    "repair": "Исправление",
    "alternative": "Вариант",
}

_PHASE_GROUP_LABELS: dict[str, str] = {
    "initial": "Первые варианты",
    "repair": "Исправления",
    "alternative": "Варианты",
}

# Internal candidate lifecycle tokens that would leak raw English into the UI.
# "complete"/"candidate" carry no user-visible meaning next to the phase and
# length/fits badges; a length failure is spelled out instead.
_CANDIDATE_STATUS_LABELS: dict[str, str] = {
    "complete": "",
    "candidate": "",
    "invalid_length": "не проходит по длине",
    "accepted": "принято",
    "rejected": "отклонено",
    "unreviewed": "без проверки",
    "pending": "ожидает",
    "error": "ошибка",
}

_STATUS_TONES: dict[str, str] = {
    "accepted": "ok",
    "rejected": "err",
    "unresolved": "warn",
    "pending": "warn",
    "error": "err",
    "blocked": "err",
    "cannot_shorten": "warn",
    "zero": "warn",
    "invalid": "err",
    "not_reviewed": "info",
    "evaluated": "info",
    "manual": "warn",
}

_MODE_TONES: dict[str, str] = {
    "medium": "info",
    "off": "mute",
    "low": "warn",
    "unknown": "mute",
}

# Plain-language reason shown in the accepted column when no validated text exists.
_STATUS_NOTES: dict[str, str] = {
    "accepted": "Принятый вариант не сохранён или не подтверждён арбитром.",
    "rejected": "Предложенный вариант отклонён.",
    "unresolved": "Вариант не подтверждён.",
    "pending": "Решение ещё не принято.",
    "error": "Ошибка обработки.",
    "blocked": "Запуск остановлен; причина указана в отчёте.",
    "cannot_shorten": "Модель сообщила, что не смогла безопасно сократить текст.",
    "zero": "Нет места под текст (нулевой лимит).",
    "invalid": "Некорректные исходные данные.",
    "not_reviewed": "Строку не проверяли.",
    "evaluated": "Вариант оценён, но не выбирался.",
    "manual": "Текст проверил человек, автоматического вердикта нет.",
}

_CSS = """
:root{--bg:#f4f6fa;--panel:#ffffff;--ink:#1c2333;--muted:#5a6478;--line:#dfe4ee;
--accent:#2563eb;--accent-soft:#e8effd;--accent-ink:#1d4ed8;
--ok:#15803d;--ok-soft:#e7f6ec;--warn:#b45309;--warn-soft:#fdf3e3;
--err:#b91c1c;--err-soft:#fdeaea;--mute-soft:#eef1f6;}
*{box-sizing:border-box}
body{margin:0;padding:1.5rem clamp(.5rem,2vw,2rem) 3rem;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans","Helvetica Neue",Arial,sans-serif}
h1{font-size:1.3rem;margin:0}
p,ul,ol,dl,dd{margin:.4rem 0}
a{color:var(--accent-ink)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;font-size:.85em}
.page-head{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;align-items:center;justify-content:space-between;margin-bottom:1rem}
.run-id{color:var(--muted);font-size:.85rem}
.cards{display:flex;flex-wrap:wrap;gap:1rem;margin-bottom:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.8rem 1rem;flex:1 1 18rem;min-width:0}
.card h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 .5rem}
.kv{display:grid;grid-template-columns:minmax(6rem,auto) 1fr;gap:.2rem .8rem;margin:0;font-size:.88rem}
.kv dt{color:var(--muted)}
.kv dd{margin:0;overflow-wrap:anywhere}
.notices{margin:0;padding:0;list-style:none}
.notices li{padding:.4rem .65rem;border-radius:8px;background:var(--warn-soft);color:#7c4a06;margin:.3rem 0;font-size:.88rem;overflow-wrap:anywhere}
.caution{background:var(--accent-soft);border:1px solid #c8d8f8;color:#1c3a72;border-radius:10px;
padding:.6rem .85rem;font-size:.88rem;margin-bottom:1rem}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:999px;background:var(--mute-soft);color:var(--muted);
font-size:.75rem;font-weight:600;line-height:1.4;vertical-align:baseline;white-space:nowrap}
.badge.ok{background:var(--ok-soft);color:var(--ok)}
.badge.warn{background:var(--warn-soft);color:var(--warn)}
.badge.err{background:var(--err-soft);color:var(--err)}
.badge.info{background:var(--accent-soft);color:var(--accent-ink)}
.badge.line{background:transparent;border:1px solid var(--line);color:var(--muted)}
.badges{display:flex;flex-wrap:wrap;gap:.3rem;align-items:center;margin:.15rem 0}
.filters{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:.6rem .75rem;margin-bottom:.6rem}
.filters input,.filters select{font:inherit;padding:.4rem .55rem;border:1px solid var(--line);border-radius:7px;background:#fff;color:inherit;max-width:100%}
.filters input:focus-visible,.filters select:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.filters input[type=search]{flex:1 1 15rem;min-width:9rem}
.filters button{font:inherit;font-weight:600;padding:.4rem .9rem;border:1px solid var(--line);border-radius:7px;
background:#fff;color:var(--ink);cursor:pointer}
.filters button:hover{background:var(--accent-soft)}
.counter{color:var(--muted);font-size:.85rem;margin:.1rem 0 .7rem}
.pager{display:none;align-items:center;flex-wrap:wrap;gap:.6rem;margin:0 0 .7rem}
.pager.js-on{display:flex}
.pager button{font:inherit;font-weight:600;padding:.4rem .9rem;border:1px solid var(--line);border-radius:7px;
background:#fff;color:var(--ink);cursor:pointer}
.pager button:hover:not(:disabled){background:var(--accent-soft)}
.pager button:disabled{opacity:.55;cursor:default}
.pager button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.pg-state{color:var(--muted);font-size:.85rem;font-variant-numeric:tabular-nums}
.empty{background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:1.2rem;text-align:center;color:var(--muted)}
.table-wrap{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow-x:auto}
table{width:100%;border-collapse:collapse;table-layout:fixed;min-width:60rem}
th,td{padding:.6rem .7rem;vertical-align:top;text-align:left;border-bottom:1px solid var(--line);overflow-wrap:anywhere;white-space:normal}
thead th{position:sticky;top:0;z-index:2;background:#eef2f8;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em;color:#46506a;border-bottom:1px solid var(--line)}
tbody tr{background:var(--panel)}
tbody tr:nth-child(even){background:#fbfcfe}
tbody tr.details-row td{background:#f8fafd;border-bottom:1px solid var(--line)}
th:nth-child(1),td:nth-child(1){width:21%}
th:nth-child(2),td:nth-child(2){width:28%}
th:nth-child(3),td:nth-child(3){width:28%}
th:nth-child(4),td:nth-child(4){width:22%}
th:nth-child(5),td:nth-child(5){width:21%}
.cid{font-weight:600}
.src,.when{color:var(--muted);font-size:.82rem}
.when{font-variant-numeric:tabular-nums}
.lbl{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;margin:.55rem 0 .1rem}
.lbl:first-child{margin-top:0}
.original{white-space:pre-wrap;overflow-wrap:anywhere}
.pick{background:var(--ok-soft);border:1px solid #bfe3ca;border-radius:8px;padding:.4rem .55rem;margin:.25rem 0;overflow-wrap:anywhere}
.accepted{border:1px solid var(--line);border-radius:8px;padding:.4rem .55rem;background:#fbfcfe;overflow-wrap:anywhere}
.accepted.ok{border-color:#bfe3ca;background:var(--ok-soft)}
.note{color:var(--muted);font-size:.85rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin:.15rem 0}
.chip{display:inline-block;padding:.15rem .5rem;border-radius:999px;background:var(--mute-soft);color:#46506a;
font-size:.78rem;line-height:1.5;overflow-wrap:anywhere}
.chip .meta{color:var(--muted);font-weight:600}
.fits-ok{color:var(--ok)}
.fits-no{color:var(--err)}
details{margin:.35rem 0;border:1px solid var(--line);border-radius:8px;padding:.35rem .55rem;background:#f8fafd}
details[open]{padding-bottom:.55rem}
summary{cursor:pointer;font-size:.85rem;color:var(--muted)}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
details>*:not(summary){margin-top:.5rem}
.advice,.issues{margin:0 0 0 .2rem;padding-left:1rem;overflow-wrap:anywhere}
.ctx{list-style:none;margin:.15rem 0;padding:0}
.ctx li{margin:.25rem 0;overflow-wrap:anywhere}
.receipts{font-size:.8rem;color:var(--muted);overflow-wrap:anywhere}
.visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap;border:0}
.js-only{display:none!important}
@media (max-width:860px){
  body{padding:1rem .6rem 2rem}
  table{min-width:0}
  thead{display:none}
  thead th{position:static}
  .table-wrap{border:0;background:transparent;overflow:visible}
  .table-wrap table,tbody,tr,td{display:block;width:auto}
  th:nth-child(n),td:nth-child(n){width:auto}
  tbody tr{margin:0 0 1rem;padding:.7rem .8rem .5rem}
  tbody tr.row{border:1px solid var(--line);border-radius:10px;box-shadow:0 1px 2px rgba(28,35,51,.05)}
  tbody td{border:0;padding:.3rem 0}
  tbody td[data-label]::before{content:attr(data-label);display:block;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);margin-bottom:.15rem}
  tbody tr.details-row td{border:0}
}
@media print{
  body{background:#fff;padding:0}
  .filters,.counter,.pager,.empty,.page-head .run-id{display:none}
  .card,.table-wrap,details,tr{break-inside:avoid;page-break-inside:avoid}
  details{display:block}
  details:not([open])>*{display:block}
  table{min-width:0;font-size:11px}
  thead th{position:static;background:#fff}
}
"""

_SCRIPT = """
(function(){
  var q=function(id){return document.getElementById(id)};
  var wrap=q('rows'), note=q('none-found');
  if(!wrap||!note){return}
  var rows=wrap.querySelectorAll('tr.row:not(.details-row)');
  var PER_PAGE=50;
  var page=1, pages=1;
  var url=new URLSearchParams(document.body.dataset.filterParams||'');
  var inputs=[];
  // Keys are filter element ids; values are the row data-* field names.
  url.forEach(function(id,field){var el=q(id); if(el){inputs.push([field,el])}});
  var pager=q('pager'), prev=q('pg-prev'), next=q('pg-next');
  function detailRow(tr){var nextRow=tr.nextElementSibling;
    return nextRow&&nextRow.classList&&nextRow.classList.contains('details-row')?nextRow:null}
  function show(tr,on){
    if(on){tr.classList.remove('js-only')}else{tr.classList.add('js-only')}}
  function renderPage(matched){
    for(var i=0;i<rows.length;i++){
      show(rows[i],false); var detail=detailRow(rows[i]); if(detail){show(detail,false)}}
    var start=(page-1)*PER_PAGE;
    var end=Math.min(start+PER_PAGE,matched.length);
    var seen={}, unique=0;
    for(var j=0;j<matched.length;j++){
      var tr=matched[j];
      if(j>=start&&j<end){show(tr,true);
        var open=detailRow(tr); if(open){show(open,true)}}
      var cid=tr.getAttribute('data-case')||'';
      if(!seen[cid]){seen[cid]=1; unique++}
    }
    q('count-shown').textContent=matched.length;
    q('count-unique').textContent=unique;
    q('count-range').textContent=matched.length?((start+1)+'\\u2013'+end):'\\u2014';
    q('page-state').textContent=pages?('Страница '+page+' из '+pages):'Страница 0 из 0';
    prev.disabled=page<=1;
    next.disabled=pages===0||page>=pages;
  }
  function apply(reset){
    if(reset){page=1}
    var needle=q('filter-text').value.trim().toLowerCase();
    var tests=inputs.map(function(pair){var el=pair[1];
      return [pair[0],el.tagName==='SELECT'?el.value:el.value.trim().toLowerCase()]});
    var matched=[];
    for(var i=0;i<rows.length;i++){
      var tr=rows[i], ok=!needle||(tr.dataset.search||'').indexOf(needle)!==-1;
      for(var j=0;ok&&j<tests.length;j++){var kind=tests[j][0], value=tests[j][1];
        if(!value){continue}
        ok=kind==='search'?(tr.dataset.search||'').indexOf(value)!==-1
                          :(tr.dataset[kind]||'')===value;
      }
      if(ok){matched.push(tr)}
    }
    pages=Math.ceil(matched.length/PER_PAGE);
    if(pages&&page>pages){page=pages}
    renderPage(matched);
    note.hidden=matched.length!==0;
  }
  function step(delta){
    var target=page+delta;
    if(target<1||target>pages){return}
    page=target;
    apply(false);
  }
  if(pager){pager.classList.add('js-on')}
  if(prev){prev.addEventListener('click',function(){step(-1)})}
  if(next){next.addEventListener('click',function(){step(1)})}
  q('filter-reset').addEventListener('click',function(){
    q('filter-text').value='';
    for(var i=0;i<inputs.length;i++){inputs[i][1].value=''}
    apply(true);
  });
  q('filter-text').addEventListener('input',function(){apply(true)});
  for(var i=0;i<inputs.length;i++){inputs[i][1].addEventListener('change',function(){apply(true)})}
  apply(true);
})();
"""

_FILTER_FIELDS = (
    ("generator", "Выбрать генератор"),
    ("arbiter", "Выбрать арбитра"),
    ("repetition", "Выбрать повтор"),
    ("status", "Выбрать статус"),
    ("source", "Выбрать источник"),
)
_FILTER_FIELD_LABELS: dict[str, str] = dict(_FILTER_FIELDS)

# Neutral first option: an empty value means "no filter", so the initial
# apply() and the reset button both restore every row instead of selecting the
# first real value the browser would otherwise preselect.
_FILTER_ALL_LABELS: dict[str, str] = {
    "generator": "Все генераторы",
    "arbiter": "Все арбитры",
    "repetition": "Все повторы",
    "status": "Все статусы",
    "source": "Все источники",
}


def _text(value: Any) -> str:
    """Return a display string for any JSON scalar; null/empty becomes an em dash."""
    if value is None:
        return _LABEL_FALLBACK
    if isinstance(value, str):
        return value if value else _LABEL_FALLBACK
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _value(value: Any) -> str:
    """Full display text without the empty dash, for row content."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _esc(value: Any) -> str:
    """Escape every dynamic scalar for text and attribute contexts."""
    return escape(_text(value), quote=True)


def _status_label(value: Any) -> str:
    raw = _value(value)
    return _STATUS_LABELS.get(raw, raw if raw else _LABEL_FALLBACK)


def _status_tone(value: Any) -> str:
    return _STATUS_TONES.get(_value(value), "info")


def _decision_label(value: Any) -> str:
    raw = _value(value)
    return _DECISION_LABELS.get(raw, raw if raw else _LABEL_FALLBACK)


def _verdict_label(value: Any) -> str:
    raw = _value(value)
    if raw in _VERDICT_LABELS:
        return _VERDICT_LABELS[raw]
    # Unknown or missing verdicts read as pending instead of a bare token.
    return raw if raw else _VERDICT_LABELS["pending"]


def _phase_chip_text(phase: Any) -> str:
    raw = _value(phase)
    return _PHASE_LABELS.get(raw, raw if raw else "без фазы")


def _mode_badge(mode: Any) -> str:
    """Raw saved reasoning-mode code (medium/off/low); never a chain of thought."""
    raw = _value(mode)
    tone = _MODE_TONES.get(raw, "mute")
    return f'<span class="badge {tone}">режим: {escape(raw or "неизвестно", quote=True)}</span>'


def _badge(text: str, tone: str) -> str:
    return f'<span class="badge {tone}">{text}</span>'


def _time_ms(value: Any) -> str:
    """Format saved milliseconds as HH:MM:SS.mmm; unknown/negative is an em dash."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return _LABEL_FALLBACK
    hours, rest = divmod(value, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _window(start: Any, end: Any) -> str:
    start_text = _time_ms(start)
    end_text = _time_ms(end)
    if start_text == _LABEL_FALLBACK and end_text == _LABEL_FALLBACK:
        return _LABEL_FALLBACK
    return f"{start_text} – {end_text}"


def _fits_badge(fits: Any) -> str:
    if fits is True:
        return _badge("умещается", "ok")
    if fits is False:
        return _badge("не умещается", "warn")
    return ""


def _length_badge(length: Any, max_chars: Any) -> str:
    if length is None and max_chars is None:
        return ""
    return (
        f'<span class="badge">{escape(_text(length), quote=True)}</span>'
        f'<span class="badge line">лимит {escape(_text(max_chars), quote=True)}</span>'
    )


def _label_value_list(items: Any) -> str:
    if not isinstance(items, (list, tuple)):
        return ""
    rendered: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        label = _esc(item.get("label"))
        value = _esc(item.get("value"))
        rendered.append(f"<dt>{label}</dt><dd>{value}</dd>")
    if not rendered:
        return ""
    return f'<dl class="kv">{"".join(rendered)}</dl>'


def _details(summary: str, inner: str) -> str:
    if not inner:
        return ""
    return (
        '<details><summary>'
        + escape(summary, quote=True)
        + "</summary>"
        + inner
        + "</details>"
    )


def _context_block(context: Any) -> str:
    if not isinstance(context, (list, tuple)):
        return ""
    lines: list[str] = []
    for item in context:
        if not isinstance(item, Mapping):
            continue
        text = escape(_value(item.get("text")), quote=True)
        when = escape(_window(item.get("start"), item.get("end")), quote=True)
        lines.append(f'<li><span class="when">{when}</span> — {text}</li>')
    if not lines:
        return ""
    return f'<ul class="ctx">{"".join(lines)}</ul>'


def _receipts_block(receipt_ids: Any) -> str:
    if not isinstance(receipt_ids, (list, tuple)) or not receipt_ids:
        return '<p class="receipts">Нет идентификаторов запросов.</p>'
    ids = ", ".join(escape(_value(item), quote=True) for item in receipt_ids)
    return f'<p class="receipts">Запросы: {ids}</p>'


def _candidate_line(candidate: Mapping[str, Any]) -> str:
    phase_raw = _value(candidate.get("phase"))
    phase = _phase_chip_text(phase_raw)
    variant = _text(candidate.get("id"))
    text = _value(candidate.get("text"))
    raw_status = _value(candidate.get("status"))
    status = _CANDIDATE_STATUS_LABELS.get(raw_status, raw_status)
    parts = [f"фаза: {phase}"]
    # The saved candidate id repeats the phase token for first/repair answers;
    # show it only when it adds real information (e.g. selector candidate ids).
    if variant and variant != phase_raw:
        parts.append(f"вариант {variant}")
    if status:
        parts.append(status)
    badges = [
        f'<span class="badge">{escape(part, quote=True)}</span>'
        for part in parts
    ]
    badges.append(_length_badge(candidate.get("length"), candidate.get("max_chars")))
    badges.append(_fits_badge(candidate.get("fits")))
    if candidate.get("selected") is True:
        badges.append(_badge("выбран", "ok"))
    if not text:
        badges.append(_badge("пустой текст", "mute"))
    body = f'<div class="chips">{"".join(b for b in badges if b)}</div>'
    if text:
        body += f'<div>{escape(text, quote=True)}</div>'
    tone = "ok" if candidate.get("selected") is True else ""
    class_attr = f' class="{tone}"' if tone else ""
    return f"<li{class_attr}>{body}</li>"


def _candidates_html(candidates: Any) -> str:
    if not isinstance(candidates, (list, tuple)):
        return ""
    initial: list[str] = []
    repair: list[str] = []
    other: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        phase = _value(candidate.get("phase"))
        line = _candidate_line(candidate)
        if phase == "initial":
            initial.append(line)
        elif phase == "repair":
            repair.append(line)
        else:
            other.append(line)
    if not (initial or repair or other):
        return ""
    blocks: list[str] = []
    if initial:
        blocks.append(f'<p class="lbl">{_PHASE_GROUP_LABELS["initial"]}</p><ul class="ctx">{ "".join(initial) }</ul>')
    if repair:
        blocks.append(f'<p class="lbl">{_PHASE_GROUP_LABELS["repair"]}</p><ul class="ctx">{ "".join(repair) }</ul>')
    if other:
        blocks.append(f'<p class="lbl">{_PHASE_GROUP_LABELS["alternative"]}</p><ul class="ctx">{ "".join(other) }</ul>')
    return "".join(blocks)


def _candidates_visible_html(candidates: Any) -> str:
    """Candidates shown inline in the proposal column (all when few)."""
    if not isinstance(candidates, (list, tuple)):
        return ""
    visible = [c for c in candidates if isinstance(c, Mapping)]
    if not visible:
        return ""
    head = visible[:5]
    tail = visible[5:]
    body = "".join(_candidate_line(c) for c in head)
    html = f'<ul class="ctx">{body}</ul>'
    if tail:
        rest = "".join(_candidate_line(c) for c in tail)
        html += _details(f"Ещё варианты: {len(tail)}", f'<ul class="ctx">{rest}</ul>')
    return html


def _accepted_block(row: Mapping[str, Any]) -> str:
    accepted_text = row.get("accepted_text")
    evaluation = _value(row.get("decision_kind")) == "evaluation"
    if not evaluation and isinstance(accepted_text, str) and accepted_text:
        return (
            f'<div class="accepted ok"><p><span class="badge ok">Принято</span></p>'
            f'<p class="original">{escape(accepted_text, quote=True)}</p>'
            f'<p class="note">Одобрен арбитром; проверка длины пройдена.</p></div>'
        )
    if evaluation:
        return '<div class="accepted"><p><span class="badge">Не выбирался: только оценка</span></p></div>'
    note = _STATUS_NOTES.get(_value(row.get("status")), "")
    return (
        '<div class="accepted"><p><span class="badge">Нет принятого варианта</span></p>'
        f'<p class="note">{escape(note, quote=True) if note else ""}</p></div>'
    )


def _status_select_options(rows: list[Mapping[str, Any]]) -> str:
    seen: dict[str, int] = {}
    for row in rows:
        raw = _value(row.get("status"))
        if raw:
            seen[raw] = seen.get(raw, 0) + 1
    if not seen:
        return ""
    options = [f'<option value="">{_FILTER_ALL_LABELS["status"]}</option>']
    for raw, count in seen.items():
        label = f"{_status_label(raw)} ({count})"
        options.append(f'<option value="{escape(raw, quote=True)}">{escape(label, quote=True)}</option>')
    return f'<select id="filter-status" aria-label="Статус">{ "".join(options) }</select>'


def _select_for(rows: list[Mapping[str, Any]], field: str) -> str:
    seen: list[str] = []
    for row in rows:
        value = _value(row.get(field))
        if value and value not in seen:
            seen.append(value)
    if len(seen) < 2:
        return ""
    options = [f'<option value="">{_FILTER_ALL_LABELS[field]}</option>'] + [
        f'<option value="{escape(value, quote=True)}">{escape(value, quote=True)}</option>'
        for value in seen
    ]
    return (
        f'<select id="filter-{escape(field, quote=True)}" '
        f'aria-label="{escape(_FILTER_FIELD_LABELS[field], quote=True)}">'
        f'{"".join(options)}</select>'
    )


def _repetition_select(rows: list[Mapping[str, Any]]) -> str:
    seen: list[int] = []
    missing = False
    for row in rows:
        value = row.get("repetition")
        if isinstance(value, int) and not isinstance(value, bool):
            if value not in seen:
                seen.append(value)
        else:
            missing = True
    if not seen:
        return ""
    options = [f'<option value="">{_FILTER_ALL_LABELS["repetition"]}</option>'] + [
        f'<option value="{value}">повтор {value}</option>' for value in sorted(seen)
    ]
    if missing:
        options.append('<option value="none">без повтора</option>')
    return (
        '<select id="filter-repetition" aria-label="Повтор">'
        + "".join(options)
        + "</select>"
    )


def _source_select(rows: list[Mapping[str, Any]]) -> str:
    seen: list[str] = []
    for row in rows:
        value = _value(row.get("source"))
        if value and value not in seen:
            seen.append(value)
    if len(seen) < 2:
        return ""
    options = [f'<option value="">{_FILTER_ALL_LABELS["source"]}</option>'] + [
        f'<option value="{escape(value, quote=True)}">{escape(value, quote=True)}</option>'
        for value in seen
    ]
    return f'<select id="filter-source" aria-label="Источник">{ "".join(options) }</select>'


def _repetition_text(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"повтор {value}"
    return _text(value)


def _row_html(row: Mapping[str, Any], index: int) -> str:
    case_id = _value(row.get("case_id"))
    source = _value(row.get("source"))
    generator = _value(row.get("generator"))
    generator_mode = _value(row.get("generator_mode"))
    arbiter = _value(row.get("arbiter"))
    arbiter_mode = _value(row.get("arbiter_mode"))
    decision_kind = _value(row.get("decision_kind"))
    status = _value(row.get("status"))
    original = _value(row.get("original"))
    accepted_text = row.get("accepted_text")
    candidates = row.get("candidates")

    search_parts = [case_id, source, _value(row.get("original"))]
    if isinstance(candidates, (list, tuple)):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                search_parts.append(_value(candidate.get("text")))
                search_parts.append(_value(candidate.get("id")))
    if isinstance(accepted_text, str):
        search_parts.append(accepted_text)
    for entry in ("issues", "advice", "verdict", "initial_verdict"):
        value = row.get(entry)
        if isinstance(value, str):
            search_parts.append(value)
    search = " ".join(part for part in search_parts if part).lower()

    details_parts: list[str] = []
    ctx = _context_block(row.get("context"))
    if ctx:
        details_parts.append(f'<p class="lbl">Контекст</p>{ctx}')
    details_html = _label_value_list(row.get("details"))
    if details_html:
        details_parts.append(f'<p class="lbl">Дополнительно</p>{details_html}')
    details_parts.append(_receipts_block(row.get("receipt_ids")))
    candidates_html = _candidates_html(candidates)
    if candidates_html:
        details_parts.append(f'<p class="lbl">Все варианты</p>{candidates_html}')

    # Per-row disclosure block, spanning the table width.
    detail_row = (
        '<tr class="details-row row" data-case="' + escape(case_id, quote=True) + '">'
        '<td colspan="5">'
        '<details><summary>Контекст, детали и идентификаторы запросов</summary>'
        + "".join(details_parts)
        + "</details></td></tr>"
    )

    decision_html = _decision_cell(row)
    proposal_html = _proposal_cell(row, candidates)

    return (
        f'<tr class="row" id="row-{index}" data-case="{escape(case_id, quote=True)}" '
        f'data-source="{escape(source, quote=True)}" '
        f'data-generator="{escape(generator, quote=True)}" '
        f'data-arbiter="{escape(arbiter, quote=True)}" '
        f'data-repetition="{escape(_repetition_token(row.get("repetition")), quote=True)}" '
        f'data-status="{escape(status, quote=True)}" '
        f'data-search="{escape(search, quote=True)}">'
        f'<td data-label="Реплика / источник">'
        f'<div class="cid mono">{escape(case_id, quote=True)}</div>'
        f'<div class="src">{escape(source, quote=True)}</div>'
        f'<div class="when">{escape(_window(row.get("start"), row.get("end")), quote=True)}</div>'
        f'<div class="src">{escape(_repetition_text(row.get("repetition")), quote=True)}</div>'
        f"</td>"
        f'<td data-label="Оригинал"><div class="original">{escape(original, quote=True)}</div></td>'
        f'{proposal_html}'
        f'{decision_html}'
        f'<td data-label="Принятый вариант">{_accepted_block(row)}</td>'
        f"</tr>"
        f"{detail_row}"
    )


def _proposal_cell(row: Mapping[str, Any], candidates: Any) -> str:
    generator = _value(row.get("generator"))
    mode = _value(row.get("generator_mode"))
    badges = [
        _badge(escape(generator, quote=True), "info") if generator
        else _badge("генератор не указан", "line"),
        _mode_badge(mode),
    ]
    return (
        f'<td data-label="Предложение модели">'
        f'<div class="badges">{ "".join(badges) }</div>'
        f'{_candidates_visible_html(candidates)}'
        f"</td>"
    )


def _decision_cell(row: Mapping[str, Any]) -> str:
    arbiter = _value(row.get("arbiter"))
    mode = _value(row.get("arbiter_mode"))
    kind = _value(row.get("decision_kind"))
    status = _value(row.get("status"))
    initial = _value(row.get("initial_verdict"))
    final = _value(row.get("verdict"))
    badges: list[str] = []
    if arbiter:
        badges.append(_badge(escape(arbiter, quote=True), "info"))
    badges.append(_mode_badge(mode))
    badges.append(_badge(escape(_decision_label(kind), quote=True), "line"))
    badges.append(_badge(escape(_status_label(status), quote=True), _status_tone(status)))
    verdicts: list[str] = []
    if initial and final and initial != final:
        verdicts.append(f'<div class="note">Исходно: {escape(_verdict_label(initial), quote=True)}</div>')
        verdicts.append(f'<div class="note">После исправления: {escape(_verdict_label(final), quote=True)}</div>')
    elif final and not initial:
        # No saved initial verdict: name it pending instead of claiming one.
        verdicts.append(f'<div class="note">Исходно: {escape(_verdict_label(None), quote=True)}</div>')
        verdicts.append(f'<div class="note">После исправления: {escape(_verdict_label(final), quote=True)}</div>')
    elif initial or final:
        verdicts.append(f'<div class="note">Решение: {escape(_verdict_label(final or initial), quote=True)}</div>')
    issues = row.get("issues")
    issues_html = ""
    if isinstance(issues, (list, tuple)) and issues:
        items = "".join(f"<li>{escape(_value(item), quote=True)}</li>" for item in issues)
        issues_html = _details(
            f"Замечания ({len(issues)})",
            f'<ul class="issues">{items}</ul>',
        )
    advice = _value(row.get("advice"))
    advice_html = ""
    if advice:
        advice_html = _details("Совет", f'<p class="advice">{escape(advice, quote=True)}</p>')
    return (
        f'<td data-label="Арбитр и решение">'
        f'<div class="badges">{ "".join(badges) }</div>'
        f'{"".join(verdicts)}'
        f"{issues_html}{advice_html}"
        f"</td>"
    )


def _repetition_token(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "none"


def _summary_cards(document: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    cards: list[str] = []
    title = _esc(document.get("title"))
    run_id = _esc(document.get("run_id"))
    status = _value(document.get("status"))
    cards.append(
        '<section class="card"><h2>Запуск</h2><dl class="kv">'
        f"<dt>Название</dt><dd>{title}</dd>"
        f"<dt>Идентификатор</dt><dd class=\"mono\">{run_id}</dd>"
        f'<dt>Статус</dt><dd>{_badge(escape(_status_label(status), quote=True), _status_tone(status))}</dd>'
        f"</dl></section>"
    )
    metadata = document.get("metadata")
    metadata_html = _label_value_list(metadata)
    if metadata_html:
        cards.append(f'<section class="card"><h2>Метаданные</h2>{metadata_html}</section>')
    notices = document.get("notices")
    if isinstance(notices, (list, tuple)) and notices:
        items = "".join(f"<li>{escape(_value(item), quote=True)}</li>" for item in notices)
        cards.append(f'<section class="card"><h2>Предупреждения</h2><ul class="notices">{items}</ul></section>')
    counts = (
        '<section class="card"><h2>Объём</h2><dl class="kv">'
        f"<dt>Строк</dt><dd>{len(rows)}</dd>"
        f"<dt>Уникальных кейсов</dt><dd>{len({_value(r.get('case_id')) for r in rows})}</dd>"
        "</dl></section>"
    )
    cards.append(counts)
    return f'<div class="cards">{ "".join(cards) }</div>'


def render_report(document: Mapping[str, Any]) -> str:
    """Render the shared comparison document as one self-contained HTML string."""
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    rows_value = document.get("rows")
    rows: list[Mapping[str, Any]] = [
        row for row in rows_value if isinstance(row, Mapping)
    ] if isinstance(rows_value, (list, tuple)) else []

    filters: list[str] = [
        '<input type="search" id="filter-text" placeholder="Поиск: оригинал, вариант, кейс, источник" '
        'aria-label="Поиск по строкам">',
        _select_for(rows, "generator"),
        _select_for(rows, "arbiter"),
        _repetition_select(rows),
        _status_select_options(rows),
        _source_select(rows),
        '<button type="button" id="filter-reset">Сбросить</button>',
    ]
    filter_controls = "".join(part for part in filters if part)
    table_rows = "".join(_row_html(row, index) for index, row in enumerate(rows))
    title = _value(document.get("title")) or PAGE_TITLE
    total = len(rows)
    unique_total = len({_value(row.get("case_id")) for row in rows})
    range_text = f"1–{total}" if total else "—"
    caution = (
        "Отметка «умещается» подтверждает только длину, а не смысл, качество озвучки "
        "или пригодность аудио. «Нет принятого варианта» означает, что принятый текст "
        "не сохранён или не подтверждён арбитром, а не отказ сокращать."
    )
    body = (
        '<header class="page-head">'
        f"<h1>{escape(title, quote=True)}</h1>"
        f'<div class="run-id">запуск: <span class="mono">{_esc(document.get("run_id"))}</span></div>'
        "</header>"
        f"{_summary_cards(document, rows)}"
        f'<p class="caution">{caution}</p>'
        f'<div class="filters">{filter_controls}</div>'
        '<p class="counter" role="status">Найдено: '
        f'<span id="count-shown">{total}</span> · уникальных кейсов: '
        f'<span id="count-unique">{unique_total}</span> · Показаны: '
        f'<span id="count-range">{range_text}</span></p>'
        '<nav class="pager" id="pager" aria-label="Постраничная навигация">'
        '<button type="button" id="pg-prev" disabled>Назад</button>'
        '<span class="pg-state" id="page-state">Страница 1 из 1</span>'
        '<button type="button" id="pg-next" disabled>Вперёд</button>'
        "</nav>"
        '<div class="table-wrap"><table>'
        "<caption class=\"visually-hidden\">"
        "Сравнение сокращений: оригинал, предложение модели, решение арбитра, принятый вариант"
        "</caption>"
        "<colgroup>"
        '<col style="width:21%"><col style="width:28%"><col style="width:28%">'
        '<col style="width:22%"><col style="width:21%">'
        "</colgroup>"
        "<thead><tr>"
        '<th scope="col">Реплика / источник</th>'
        '<th scope="col">Оригинал</th>'
        '<th scope="col">Предложение модели</th>'
        '<th scope="col">Арбитр и решение</th>'
        '<th scope="col">Принятый вариант</th>'
        "</tr></thead>"
        f'<tbody id="rows">{table_rows}</tbody>'
        "</table></div>"
        f'<p class="empty" id="none-found" hidden>Ничего не найдено.</p>'
    )
    return (
        "<!DOCTYPE html>\n"
        f"{HTML_MARKER}\n"
        '<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "connect-src 'none'; img-src 'none'; style-src 'unsafe-inline'; "
        "script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">\n"
        f"<title>{escape(PAGE_TITLE, quote=True)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n"
        '<body data-filter-params="search=filter-text&amp;generator=filter-generator'
        '&amp;arbiter=filter-arbiter&amp;repetition=filter-repetition'
        '&amp;status=filter-status&amp;source=filter-source">\n'
        f"{body}\n"
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )
