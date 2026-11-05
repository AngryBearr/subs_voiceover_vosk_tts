"""Tab «Кто это?»: speaker groups without a name, a few snippets to listen to, the person's answer saved to the
episode's ``speakers.override.json`` (backend: ``stt.who``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore

from stt import who
from ui import seasons

log = logging.getLogger(__name__)

INTRO = (
    "Иногда программа не может понять, кто говорит: такие голоса остаются без имени и озвучиваются общим голосом. "
    "Если вы назовёте человека, он получит свой голос (клонированный по образцу), а его слова помогут программе "
    "узнавать его в следующих сериях сезона.\n\n"
    "Лучше всего заполнять эту вкладку, когда весь сезон уже подготовлен (распознан), но ещё не озвучен: тогда ответы "
    "применятся при первой озвучке. Послушайте отрывки, выберите имя из списка героев сезона и нажмите «Сохранить»."
)
MIXED_CHOICE = "Несколько человек (смешанная группа)"
SKIP_CHOICE = "Не знаю / пропустить"
GENDER_CHOICES = [("Не менять", ""), ("Мужской", "male"), ("Женский", "female")]
GENDER_WORD = {"male": "мужской", "female": "женский", "unknown": "не определён"}
SLOTS = who.SNIPPETS
KEY_SEP = "|"


def results_cache(results: Path) -> Path:
    return results / ".ui" / "who"


def group_key(group: who.Group) -> str:
    return f"{group.episode}{KEY_SEP}{group.speaker}"


def split_key(key: str | None) -> tuple[str, str] | None:
    if not key or KEY_SEP not in key:
        return None
    episode, speaker = key.rsplit(KEY_SEP, 1)
    return episode, speaker


def group_label(group: who.Group) -> str:
    return (f"{seasons.episode_label(group.episode)} · {group.speaker} · {group.cues} реплик, "
            f"{group.seconds:.0f} с")


def episode_names(results: Path) -> list[str]:
    return sorted(p.parent.name for p in results.glob("*/speakers.json")) if results.is_dir() else []


def groups_for(results: Path, episodes: list[str] | None) -> list[who.Group]:
    return who.episode_groups(results, episodes or None)


def name_choices(results: Path, season_name: str) -> list[str]:
    return [*who.cast_names(results / f"{season_name}.cast.json"), MIXED_CHOICE, SKIP_CHOICE]


def group_card(group: who.Group) -> str:
    lines = [f"**{seasons.episode_label(group.episode)} · группа {group.speaker}**: {group.cues} реплик, "
             f"{group.seconds:.0f} с речи. Пол по голосу и тексту: {GENDER_WORD.get(group.gender, group.gender)}."]
    if group.role:
        lines.append(f"Что думает программа: *{group.role}*.")
    if group.gender_evidence:
        lines.append(f"Почему так: {group.gender_evidence}")
    return "\n\n".join(lines)


def video_of(source: Path, episode: str) -> Path:
    for name in seasons.video_names(source):
        if Path(name).stem == episode:
            return source / name
    return source / f"{episode}.mkv"


def after_text(source: Path, results: Path, episode: str) -> str:
    """What happens next with the saved answer."""
    stage = who.episode_stage(results, results / episode)
    if stage == "voiced":
        return ("Эта серия уже озвучена. Ответ сам не применится: чтобы переозвучить серию с новым именем, выполните "
                f"в терминале из папки проекта (серия будет озвучена заново):\n\n```\n"
                f"{who.redo_command(video_of(source, episode), results)}\n```")
    if stage == "translated":
        return ("Серия уже переведена, но ещё не озвучена. Ответ применится при следующем запуске с пересчётом шага "
                "«говорящие» (`--force speakers`); если запускать серию обычным способом, она останется как есть.\n\n"
                f"```\n{who.redo_command(video_of(source, episode), results)}\n```")
    return "Серия ещё не переведена: ответ применится при следующем запуске этой серии."


def save(src: str, res: str, key: str | None, choice: str | None, gender: str | None) -> str:
    """Writes the answer; returns the message shown under the button."""
    if not src or not res:
        return "⚠ Сначала выберите сезон вверху страницы."
    parts = split_key(key)
    if parts is None:
        return "⚠ Нечего сохранять: выберите группу в списке."
    episode, speaker = parts
    source, results = Path(src), Path(res)
    if not choice or choice == SKIP_CHOICE:
        return "Ничего не сохранено: выберите имя или «Несколько человек»."
    folder = results / episode
    if not (folder / "speakers.json").is_file():
        return f"⚠ Не найдена серия {episode}."
    try:
        if choice == MIXED_CHOICE:
            who.save_answer(folder, speaker, mixed=True, gender=gender or None)
            done = f"✅ Группа {speaker} отмечена как смешанная: имя ей не даётся, больше не спрашиваем."
        else:
            gender = gender or who.cast_gender(results / f"{source.name}.cast.json", choice)
            who.save_answer(folder, speaker, name=choice, gender=gender)
            done = f"✅ Сохранено: группа {speaker} в серии {seasons.episode_label(episode)} это {choice}."
    except (OSError, ValueError) as exc:
        log.exception("saving the answer")
        return f"⚠ Не удалось сохранить: {exc}"
    return f"{done}\n\n{after_text(source, results, episode)}"


class Tab:
    """What ``build`` hands back to the page: the line under the tab title and the reload wired to the season picker."""

    def __init__(self) -> None:
        self.note: Any = None
        self.reload: Any = None
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []


def build(folder: Any, out_root: Any) -> Tab:
    tab = Tab()
    tab.note = gr.Markdown(seasons.NO_SEASON_NOTE)
    gr.Markdown(INTRO)
    with gr.Row(equal_height=False):
        episodes = gr.Dropdown(label="Серии (фильтр)", choices=[], value=[], multiselect=True, scale=3,
                               info="Пусто: все серии сезона")
        group = gr.Dropdown(label="Группа без имени", choices=[], value=None, scale=4)
        refresh = gr.Button("Обновить список", scale=1, min_width=160)
    summary = gr.Markdown()
    card = gr.Markdown()
    audios: list[Any] = []
    texts: list[Any] = []
    rows: list[Any] = []
    for n in range(SLOTS):
        with gr.Row(equal_height=False, visible=False) as row:
            audios.append(gr.Audio(label=f"Отрывок {n + 1}", type="filepath", interactive=False, scale=3))
            texts.append(gr.Markdown())
        rows.append(row)
    with gr.Row(equal_height=False):
        name = gr.Dropdown(label="Кто это", choices=[SKIP_CHOICE], value=SKIP_CHOICE, scale=4)
        gender = gr.Radio(label="Пол голоса", choices=GENDER_CHOICES, value="", scale=2)
    save_button = gr.Button("Сохранить", variant="primary")
    message = gr.Markdown()

    def listing(src: str, res: str, eps: list[str] | None, keep: str | None = None) -> tuple[Any, ...]:
        if not src or not res or not Path(res).is_dir():
            return (gr.update(choices=[], value=[]), gr.update(choices=[], value=None), "Сезон не выбран.",
                    gr.update(choices=[SKIP_CHOICE], value=SKIP_CHOICE))
        results = Path(res)
        names = episode_names(results)
        eps = [e for e in (eps or []) if e in names]
        groups = groups_for(results, eps)
        keys = [group_key(g) for g in groups]
        text = (f"Групп без имени: **{len(groups)}**." if groups else
                "Групп без имени нет: все заметные голоса названы, спрашивать не о чем. "
                "(Серии появляются здесь после шага «говорящие».)")
        return (gr.update(choices=seasons.episode_choices(names), value=eps),
                gr.update(choices=[(group_label(g), group_key(g)) for g in groups],
                          value=keep if keep in keys else (keys[0] if keys else None)),
                text, gr.update(choices=name_choices(results, Path(src).name), value=SKIP_CHOICE))

    def show(src: str, res: str, key: str | None) -> tuple[Any, ...]:
        empty = ["", *[v for _ in range(SLOTS) for v in (gr.update(value=None), gr.update(value=""))],
                 *[gr.update(visible=False) for _ in range(SLOTS)], gr.update(value="")]
        parts = split_key(key)
        if parts is None or not res:
            return tuple(empty)
        episode, speaker = parts
        results = Path(res)
        folder_path = results / episode
        found = [g for g in who.unnamed_groups(folder_path) if g.speaker == speaker]
        if not found:
            return tuple(empty)
        g = found[0]
        who.cut_snippets(g, folder_path, results_cache(results))
        values: list[Any] = []
        for n in range(SLOTS):
            snip = g.snippets[n] if n < len(g.snippets) and g.snippets[n].wav else None
            values += [gr.update(value=str(snip.wav) if snip else None),
                       gr.update(value=f"«{snip.text}»\n\n*{snip.start:.1f}-{snip.end:.1f} с*" if snip else "")]
        shown = [gr.update(visible=n < len(g.snippets) and bool(g.snippets[n].wav)) for n in range(SLOTS)]
        note = "" if g.snippets else "Подходящих отрывков для прослушивания не нашлось (нет audio.wav или чистых реплик)."
        return (group_card(g) + ("\n\n" + note if note else ""), *values, *shown, gr.update(value=""))

    slot_outputs = [x for pair in zip(audios, texts) for x in pair]
    show_outputs = [card, *slot_outputs, *rows, message]
    inputs = [folder, out_root]

    def do_save(src: str, res: str, eps: list[str] | None, key: str | None, choice: str | None,
                sex: str | None) -> tuple[Any, ...]:
        text = save(src, res, key, choice, sex)
        if not text.startswith("✅"):
            return (gr.update(), gr.update(), gr.update(), gr.update(), text)
        listed = listing(src, res, eps)
        return (*listed, text)

    refresh.click(listing, [*inputs, episodes], [episodes, group, summary, name]).then(
        show, [*inputs, group], show_outputs)
    episodes.input(listing, [*inputs, episodes, group], [episodes, group, summary, name]).then(
        show, [*inputs, group], show_outputs)
    group.input(show, [*inputs, group], show_outputs)
    saved = save_button.click(do_save, [*inputs, episodes, group, name, gender],
                              [episodes, group, summary, name, message])
    saved.then(lambda src, res, key, text: show(src, res, key)[:-1] + (text,),
               [*inputs, group, message], show_outputs)

    def on_season(src: str, res: str) -> tuple[Any, ...]:
        listed = listing(src, res, [])
        note = (f"📂 Ответы сохраняются в `{res}/<серия>/speakers.override.json`; отрывки для прослушивания — в "
                f"`{res}/.ui/who`." if res else seasons.NO_SEASON_NOTE)
        return (note, *listed, *show(src, res, listed[1]["value"] if isinstance(listed[1], dict) else None))

    tab.reload = on_season
    tab.inputs = inputs
    tab.outputs = [tab.note, episodes, group, summary, name, *show_outputs]
    return tab

