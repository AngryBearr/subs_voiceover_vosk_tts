import json
import ast
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


def _load_runorm(device: str = "cpu") -> RUNorm:
    normalizer = RUNorm()
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
    return accentizer


def replace_ellipsis(text: str) -> str:
    return text.replace("...", "..")


def restore_ellipsis(text: str) -> str:
    return text.replace("..", "...")


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

        for sub, norm_text in zip(chunk, parts):
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

        new_item = dict(item)
        new_item["_normalized_text"] = norm_text
        normalized_items.append(new_item)

    return normalized_items


def ruaccent_pass(
    subtitles: List[Dict[str, Any]],
    accentizer: RUAccent,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Только RUAccent: берёт _normalized_text, ставит ударения и записывает в text.

    Ожидает, что в элементах есть поле _normalized_text (как после runorm_pass).
    """
    items = [dict(it) for it in subtitles]

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_texts = [it.get("_normalized_text", "") for it in batch]

        batch_string = str(batch_texts)
        processed_batch_string = accentizer.process_all(batch_string, skip_regex=r"[\[\],\"]")

        try:
            processed_texts = ast.literal_eval(processed_batch_string)
        except (SyntaxError, ValueError):
            processed_texts = [accentizer.process_all(t) for t in batch_texts]

        for j, processed_text in enumerate(processed_texts):
            if j >= len(batch):
                break
            text_with_ellipsis = restore_ellipsis(processed_text)
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
) -> List[Dict[str, Any]]:
    """Полный пайплайн: RUNorm -> RUAccent, возвращает новый список сабов.

    text всегда заменяется на список из одной строки, остальные поля не трогаются.
    """
    normalized_items = runorm_pass(subtitles, normalizer)
    return ruaccent_pass(normalized_items, accentizer, batch_size=batch_size)


def normalize_json_file(
    input_path: str,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    separator: str = DEFAULT_SEPARATOR,
    norm_device: str = "cpu",
    accent_device: str = "CUDA",
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

    args = parser.parse_args()

    normalize_json_file(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        separator=args.separator,
        norm_device=args.norm_device,
        accent_device=args.accent_device,
    )

