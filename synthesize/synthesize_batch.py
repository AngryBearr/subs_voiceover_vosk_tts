import os
import json
from pathlib import Path
from vosk_tts.model import Model
from vosk_tts.synth import Synth
from utils.subs_utils import ensure_folder_exists

def patch_model_for_device(device="cpu"):
    """Monkey-patch Model to allow CUDA/CPU selection."""
    import vosk_tts.model as model_mod
    import onnxruntime
    def patched_init(self, model_path=None, model_name=None, lang=None):
        if model_path is None:
            model_path = self.get_model_path(model_name, lang)
        else:
            model_path = Path(model_path)
        sess_options = onnxruntime.SessionOptions()
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.onnx = onnxruntime.InferenceSession(str(model_path / "model.onnx"), sess_options=sess_options, providers=providers)
        self.dic = {}
        probs = {}
        for line in open(model_path / "dictionary", encoding='utf-8'):
            items = line.split()
            prob = float(items[1])
            if probs.get(items[0], 0) < prob:
                self.dic[items[0]] = " ".join(items[2:])
                probs[items[0]] = prob
        self.config = __import__('json').load(open(model_path / "config.json"))
        import os
        if os.path.exists(model_path / "bert/vocab.txt"):
            from tokenizers.implementations import BertWordPieceTokenizer
            self.tokenizer = BertWordPieceTokenizer(vocab=str(model_path / "bert/vocab.txt"), unk_token="[UNK]", lowercase=False)
            self.bert_onnx = onnxruntime.InferenceSession(str(model_path / "bert/model.onnx"), sess_options=sess_options, providers=providers)
        else:
            self.tokenizer = None
    model_mod.Model.__init__ = patched_init

def synthesize_json_lines(
    json_path,
    model_name=None,
    lang="en-us",
    output_folder="./tts_out",
    file_prefix="tts_",
    speaker_id=0,
    device="cpu",
    default_speech_rate=1.0,
    default_noise_level=None,
    default_duration_noise_level=None,
    default_scale=None,
    text_key="text",
    speech_rate_key=None,
    noise_level_key=None,
    duration_noise_level_key=None,
    scale_key=None,
):
    """
    Synthesize audio for each line in a JSON file (list of dicts or JSONL).
    Allows per-line override of synthesis parameters.
    """
    patch_model_for_device(device)
    ensure_folder_exists(output_folder)

    # Load model and synth ONCE
    model = Model(model_name=model_name, lang=lang)
    synth = Synth(model)

    # Read JSON file
    with open(json_path, encoding="utf-8") as f:
        if json_path.endswith(".jsonl"):
            entries = [json.loads(line) for line in f if line.strip()]
        else:
            entries = json.load(f)

    for idx, entry in enumerate(entries):
        text = entry[text_key]
        speech_rate = entry.get(speech_rate_key, default_speech_rate) if speech_rate_key else default_speech_rate
        noise_level = entry.get(noise_level_key, default_noise_level) if noise_level_key else default_noise_level
        duration_noise_level = entry.get(duration_noise_level_key, default_duration_noise_level) if duration_noise_level_key else default_duration_noise_level
        scale = entry.get(scale_key, default_scale) if scale_key else default_scale
        outname = os.path.join(output_folder, f"{file_prefix}{idx+1}.wav")
        synth.synth(
            text,
            outname,
            speaker_id=speaker_id,
            noise_level=noise_level,
            speech_rate=speech_rate,
            duration_noise_level=duration_noise_level,
            scale=scale,
        )
        print(f"Synthesized: {outname}")

if __name__ == "__main__":
    # Example usage
    synthesize_json_lines(
        json_path="input.json",  # Your JSON or JSONL file
        model_name="vosk-model-tts-ru-0.8-multi",
        lang="ru",
        output_folder="./tts_out",
        file_prefix="batch_",
        speaker_id=4,
        device="cuda",
        default_speech_rate=1,
        # Optionally, set keys for per-line overrides:
        # speech_rate_key="speech_rate",
        # noise_level_key="noise_level",
        # duration_noise_level_key="duration_noise_level",
        # scale_key="scale",
        text_key="text"
    )
