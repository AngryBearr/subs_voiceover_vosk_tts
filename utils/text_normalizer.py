import json
import ast
import re
from typing import List, Dict, Any

from ruaccent import RUAccent
from runorm import RUNorm


DEFAULT_BATCH_SIZE = 10
DEFAULT_SEPARATOR = " мемы-квадробер "


cust_dict = {
    "аганоа": "аган+оа",
    "санаапу": "сан+аапу",
    "вавау": "вав+ау",
    "дэз": "д+эз",
    "кюча": "к+юча",
    "огакор": "ог+акор",
}


class ProtectedNameRUNorm(RUNorm):
    """Protect Кэт and ellipsis boundaries only during abbreviation expansion."""

    def proccess_abbr(self, text: str) -> str:
        parts = re.split(r"(\.{2,}|…)", text)
        if len(parts) > 1:
            # Ellipses are punctuation boundaries, not learned abbreviations.
            return "".join(part if i % 2 or not part.strip() else self.proccess_abbr(part)
                           for i, part in enumerate(parts))
        punctuation = '.,!?;:«»"\'…()[]{}'
        spans: List[tuple[int, int]] = []
        for match in re.finditer(r"(?<!\w)кэт(?!\w)", text, flags=re.IGNORECASE):
            start, end = match.span()
            while start > 0 and text[start - 1] in punctuation:
                start -= 1
            while end < len(text) and text[end] in punctuation:
                end += 1
            spans.append((start, end))
        if not spans:
            return super().proccess_abbr(text)

        tags = [dict(tag) for tag in self.tagger(text)]
        previous_end = 0
        for tag in tags:
            start, end, word = tag.get("start"), tag.get("end"), tag.get("word")
            if (type(start) is not int or type(end) is not int
                    or not previous_end <= start < end <= len(text) or not isinstance(word, str)
                    or re.sub(r"\s+", "", word.removeprefix("##")).casefold()
                    != re.sub(r"\s+", "", text[start:end]).casefold()):
                raise ValueError("RUNorm name protection requires valid source-bound tag offsets/words")
            previous_end = end
            overlaps = [(left, right) for left, right in spans if start < right and end > left]
            if overlaps:
                if not any(left <= start < end <= right for left, right in overlaps):
                    raise ValueError("RUNorm tag crosses protected name and unrelated source text")
                tag["entity_group"] = "PLAIN"
                tag["word"] = text[start:end]
            elif tag["entity_group"] not in {"PLAIN", "TIME"}:
                tag["entity_group"] = "TAG"

        prompt, current, counter = "", 0, 0
        for tag in self.process_tags(tags):
            prompt += text[current:tag["start"]]
            if tag["entity_group"] == "TAG":
                prompt += f"[{tag['word']}]<extra_id_{counter}>"
                counter += 1
            else:
                prompt += text[tag["start"]:tag["end"]]
            current = tag["end"]
        prompt += text[current:]
        return self.construct_answer(prompt, self.predict_abbr(prompt)) if counter else text


def _load_runorm(device: str = "cpu") -> RUNorm:
    normalizer = ProtectedNameRUNorm()
    normalizer.load(workdir="./runorm_cache", model_size="big", device=device)
    return normalizer


def _load_ruaccent(device: str = "cpu") -> RUAccent:
    accentizer = RUAccent()
    accentizer.load(
        omograph_model_size="turbo3.1",
        use_dictionary=True,
        custom_dict=cust_dict,
        device=device,
    )
    # transformers 5 tokenizers return only the inputs they declare; the stress model's ONNX
    # graph also needs token_type_ids, which its character tokenizer does not declare.
    tokenizer = accentizer.accent_model.tokenizer
    if "token_type_ids" not in tokenizer.model_input_names:
        tokenizer.model_input_names = [*tokenizer.model_input_names, "token_type_ids"]
    return accentizer


def _word_tokens(text: str) -> List[str]:
    plain = text.replace("+", "").replace("\u0301", "").casefold().replace("ё", "е")
    return re.findall(r"[^\W\d_]+|\d+(?:[.,]\d+)*", plain)


def _check_lexical_invariant(before: str, after: str, *, stage: str, cue_index: Any) -> None:
    """Fail closed on letter changes, not a general semantic/number validator.

    RUNorm may expand digits/Latin input; already-written Russian words must stay
    unchanged. RUAccent may only change stress, case, punctuation and e/yo.
    """
    if not isinstance(after, str):
        raise ValueError(f"{stage} returned non-text at cue index {cue_index}")
    if stage == "RUNorm" and re.search(r"[A-Za-z]|\d", before):
        return
    old, new = _word_tokens(before), _word_tokens(after)
    if old != new:
        raise ValueError(f"{stage} lexical change at cue index {cue_index}: {old!r} -> {new!r}")


def replace_ellipsis(text: str) -> str:
    return text.replace("…", "...")


def restore_ellipsis(text: str) -> str:
    return text.replace("…", "...")


def normalize_subtitles_runorm_only(
    subtitles: List[Dict[str, Any]],
    normalizer: RUNorm,
    batch_size: int = DEFAULT_BATCH_SIZE,
    separator: str = DEFAULT_SEPARATOR,
) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []

    for i in range(0, len(subtitles), batch_size):
        chunk = subtitles[i : i + batch_size]

        original_texts: List[str] = []
        for sub in chunk:
            text_list = sub.get("text", []) or []
            if isinstance(text_list, list):
                joined = " ".join(text_list)
            else:
                joined = str(text_list)
            original_texts.append(joined)

        merged_text = separator.join(original_texts)
        normalized_merged = normalizer.norm(merged_text)

        parts = normalized_merged.split(separator)
        if len(parts) != len(chunk):
            parts = [normalizer.norm(t) for t in original_texts]

        for sub, original_text, norm_text in zip(chunk, original_texts, parts):
            _check_lexical_invariant(original_text, norm_text, stage="RUNorm", cue_index=sub.get("index"))
            new_sub = dict(sub)
            new_sub["text"] = [norm_text]
            processed.append(new_sub)

    return processed


def runorm_pass(
    subtitles: List[Dict[str, Any]],
    normalizer: RUNorm,
) -> List[Dict[str, Any]]:
    """Только RUNorm: нормализует text и добавляет временное поле _normalized_text.

    Используется как отдельный шаг или как часть общего пайплайна.
    """
    normalized_items: List[Dict[str, Any]] = []

    for item in subtitles:
        text_list = item.get("text", []) or []
        if isinstance(text_list, list):
            joined = " ".join(text_list)
        else:
            joined = str(text_list)

        no_ellipsis = replace_ellipsis(joined)
        norm_text = normalizer.norm(no_ellipsis)
        _check_lexical_invariant(joined, norm_text, stage="RUNorm", cue_index=item.get("index"))

        new_item = dict(item)
        new_item["_normalized_text"] = norm_text
        normalized_items.append(new_item)

    return normalized_items


_ELLIPSIS_RE = re.compile(r"(\.{3}|…)")
_LETTER_RE = re.compile(r"[^\W\d_]")


def accent_cue(accentizer: RUAccent, text: str) -> str:
    """RUAccent on one cue; the pieces between ellipses go alone: after "..." RUAccent stresses
    almost nothing ("Это... стр+ашно" came out "+Это... страшно", "около... тридцати часов")."""
    return "".join(accentizer.process_all(part) if _LETTER_RE.search(part) and not _ELLIPSIS_RE.fullmatch(part)
                   else part for part in _ELLIPSIS_RE.split(text))


_WORD_RE = re.compile(r"[А-Яа-яЁё+]+")
_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def fill_dictionary_stress(text: str, accents: Dict[str, str], omographs: Dict[str, Any]) -> str:
    """The dictionary stress for words RUAccent left unmarked (its stress-usage predictor drops it
    on words said weakly: "это", "или", "сразу", "слишком"), so no TTS engine guesses them.

    Homographs stay unmarked, and so does a word after a stressed one-syllable word: the stress
    moved there ("н+е было", "н+а пол").
    """
    out: List[str] = []
    last = 0
    previous = ""
    for match in _WORD_RE.finditer(text):
        word = match.group()
        out.append(text[last:match.start()])
        last = match.end()
        key = word.lower()
        form = accents.get(key) if sum(ch in _VOWELS for ch in word) >= 2 and "+" not in word \
            and "ё" not in key and key not in omographs else None
        moved = "+" in previous and sum(ch in _VOWELS for ch in previous) == 1
        if form and form.count("+") == 1 and form.replace("+", "") == key and not moved:
            position = form.index("+")
            word = word[:position] + "+" + word[position:]
        out.append(word)
        previous = word
    out.append(text[last:])
    return "".join(out)


def ruaccent_pass(
    subtitles: List[Dict[str, Any]],
    accentizer: RUAccent,
    batch_size: int = DEFAULT_BATCH_SIZE,
    per_cue: bool = False,
) -> List[Dict[str, Any]]:
    """Только RUAccent: берёт _normalized_text, ставит ударения и записывает в text.

    Ожидает, что в элементах есть поле _normalized_text (как после runorm_pass).

    By default a batch goes to RUAccent as one string (the list's repr, split at ``[ ] , "``).
    ``per_cue``: every cue on its own instead (``accent_cue``, also apart at ellipses). The joined
    string loses stress: RUAccent then reads the pieces between commas and quotes apart, and the
    first word of a cue is glued to the repr's quote ("'Это"); on S51E01 2.9% of the words with 2+
    vowels came out with no stress at all (the first word of most cues, names the stress check had
    corrected, "только", "племя"), per cue 1.6%, per cue and apart at ellipses 1.4% (mostly words
    RUAccent's stress-usage predictor leaves unstressed: "это", "или", "сразу"; for TTS
    ``fill_dictionary_stress`` gives those their dictionary stress: 0.3% left).
    """
    items = [dict(it) for it in subtitles]
    if any(not isinstance(it.get("_normalized_text"), str) for it in items):
        raise ValueError("RUAccent requires each cue's bound RUNorm text")

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_texts = [it.get("_normalized_text", "") for it in batch]

        if per_cue:
            processed_texts = [accent_cue(accentizer, t) for t in batch_texts]
        else:
            batch_string = str(batch_texts)
            processed_batch_string = accentizer.process_all(batch_string, skip_regex=r"[\[\],\"]")
            try:
                processed_texts = ast.literal_eval(processed_batch_string)
            except (SyntaxError, ValueError):
                processed_texts = [accentizer.process_all(t) for t in batch_texts]

        if (not isinstance(processed_texts, list) or len(processed_texts) != len(batch)
                or any(not isinstance(text, str) for text in processed_texts)):
            raise ValueError(f"RUAccent malformed batch output for cue indices {[it.get('index') for it in batch]}")

        for j, processed_text in enumerate(processed_texts):
            text_with_ellipsis = restore_ellipsis(processed_text)
            _check_lexical_invariant(batch_texts[j], text_with_ellipsis,
                                     stage="RUAccent", cue_index=batch[j].get("index"))
            batch[j]["text"] = [text_with_ellipsis]

    # чистим временное поле и возвращаем итоговый список
    result: List[Dict[str, Any]] = []
    for item in items:
        new_item = dict(item)
        new_item.pop("_normalized_text", None)
        result.append(new_item)

    return result


def normalize_and_accent_subtitles(
    subtitles: List[Dict[str, Any]],
    normalizer: RUNorm,
    accentizer: RUAccent,
    batch_size: int = DEFAULT_BATCH_SIZE,
    separator: str = DEFAULT_SEPARATOR,
    per_cue: bool = False,
) -> List[Dict[str, Any]]:
    """Полный пайплайн: RUNorm -> RUAccent, возвращает новый список сабов.

    text всегда заменяется на список из одной строки, остальные поля не трогаются.
    ``per_cue``: RUAccent on every cue alone (see ``ruaccent_pass``).
    """
    normalized_items = runorm_pass(subtitles, normalizer)
    return ruaccent_pass(normalized_items, accentizer, batch_size=batch_size, per_cue=per_cue)


def normalize_json_file(
    input_path: str,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    separator: str = DEFAULT_SEPARATOR,
    norm_device: str = "cpu",
    accent_device: str = "CUDA",
    per_cue: bool = False,
) -> None:
    with open(input_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    normalizer = _load_runorm(device=norm_device)
    accentizer = _load_ruaccent(device=accent_device)
    transformed = normalize_and_accent_subtitles(
        data,
        normalizer,
        accentizer,
        batch_size=batch_size,
        separator=separator,
        per_cue=per_cue,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize + accent Russian subtitles JSON (RUNorm + RUAccent)")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument("batch_size", nargs="?", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--separator", dest="separator", default=DEFAULT_SEPARATOR, help="Separator between subtitles inside batch")
    parser.add_argument(
        "--accent-device",
        dest="accent_device",
        default="cpu",
        choices=["CPU", "CUDA", "cuda", "cpu"],
        help="Device for RUAccent",
    )
    parser.add_argument(
        "--norm-device",
        dest="norm_device",
        default="cpu",
        choices=["CUDA", "cuda", "CPU", "cpu"],
        help="Device for RUNorm",
    )

    parser.add_argument(
        "--accent-per-cue",
        dest="per_cue",
        action="store_true",
        help="RUAccent on every cue alone (slower; the joined batch leaves some words without stress)",
    )

    args = parser.parse_args()

    normalize_json_file(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        separator=args.separator,
        norm_device=args.norm_device,
        accent_device=args.accent_device,
        per_cue=args.per_cue,
    )
