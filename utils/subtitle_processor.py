import ast
from copy import deepcopy
from ruaccent import RUAccent
from runorm import RUNorm
from .text_normalizer import replace_ellipsis, restore_ellipsis

def process_subtitles_with_ruaccent(subtitle_list, batch_size=10, text_key='text', accentuator_instance=None, normalizer_instance=None):
    """
    Processes a list of subtitle objects in batches using RUAccent, preserving subtitle structure.
    Args:
        subtitle_list (list): List of subtitle dicts, each with a 'text' field (or configurable key).
        batch_size (int): Number of subtitles to process in a batch.
        text_key (str): Key to extract/replace text from subtitle dicts.
        accentuator_instance (RUAccent, optional): Pre-initialized RUAccent. If None, will initialize internally.
        normalizer_instance (RUNorm, optional): Pre-initialized RUNorm. If None, normalization is skipped.
    Returns:
        List of subtitle dicts with processed text.
    """
    if accentuator_instance is None:
        accentuator_instance = RUAccent()
        accentuator_instance.load(omograph_model_size='turbo3.1', use_dictionary=True, custom_dict={}, device="CPU", workdir='cache_models')

    processed_subtitles = []
    n = len(subtitle_list)
    for i in range(0, n, batch_size):
        batch = subtitle_list[i:i+batch_size]
        # Extract and pre-process texts
        texts = []
        for sub in batch:
            text = sub.get(text_key, '')
            if normalizer_instance:
                text = normalizer_instance.norm(text)
            text = replace_ellipsis(text)
            texts.append(text)
        # Stringify the batch
        stringified_batch = str(texts)
        skip_pattern = r'[\[\],\"]'  # skip list syntax
        try:
            processed_stringified = accentuator_instance.process_all(stringified_batch, skip_regex=skip_pattern)
            processed_texts = ast.literal_eval(processed_stringified)
            if not isinstance(processed_texts, list) or len(processed_texts) != len(texts):
                raise ValueError("Processed list structure or length mismatch.")
        except Exception as e:
            # Fallback: process texts individually
            processed_texts = []
            for text in texts:
                try:
                    processed_text = accentuator_instance.process_all(text)
                except Exception:
                    processed_text = text  # fallback to original
                processed_texts.append(processed_text)
        # Post-process and update subtitle objects
        for sub, processed_text in zip(batch, processed_texts):
            new_sub = deepcopy(sub)
            new_sub[text_key] = restore_ellipsis(processed_text)
            processed_subtitles.append(new_sub)
    return processed_subtitles
