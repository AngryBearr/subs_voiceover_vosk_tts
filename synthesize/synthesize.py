import os
from pathlib import Path
from vosk_tts.model import Model
from vosk_tts.synth import Synth
from utils.subs_utils import ensure_folder_exists


def synthesize_text_to_audio(
    text,
    model_name=None,
    lang="en-us",
    output_folder=".",
    file_prefix="tts_",
    speaker_id=0,
    speech_rate=1.0,
    noise_level=None,
    duration_noise_level=None,
    scale=None,
    device="cpu"  # 'cpu' or 'cuda'
):
    """
    Synthesize text to audio using vosk_tts with customizable options.

    Args:
        text (str): Text to synthesize.
        model_name (str): Model name to use (e.g., 'vosk-model-tts-ru-0.8-multi').
        lang (str): Language code (default 'en-us').
        output_folder (str): Where to save the output file.
        file_prefix (str): Prefix for the output file.
        speaker_id (int): Speaker id for multispeaker models.
        speech_rate (float): Speed of speech (default 1.0).
        noise_level (float): Noise level for synthesis (default: model default).
        duration_noise_level (float): Duration noise (default: model default).
        scale (float): Volume scaling (default: model default).
        device (str): 'cpu' or 'cuda' (default 'cpu').
    Returns:
        str: Path to the generated wav file.
    """
    # Patch Model to allow CUDA if requested
    import vosk_tts.model as model_mod
    import onnxruntime
    orig_init = model_mod.Model.__init__
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

    # Prepare output path
    ensure_folder_exists(output_folder)
    outname = os.path.join(output_folder, f"{file_prefix}output.wav")

    model = Model(model_name=model_name, lang=lang)
    synth = Synth(model)
    synth.synth(
        text,
        outname,
        speaker_id=speaker_id,
        noise_level=noise_level,
        speech_rate=speech_rate,
        duration_noise_level=duration_noise_level,
        scale=scale,
    )
    return outname

# Example usage:
if __name__ == "__main__":
    wav = synthesize_text_to_audio(
        text="У Лукоморья дуб зелёный. Златая цепь на дубе том. И днём и ночью кот учёный всё ходит по цеп+и кругом.",
        model_name="vosk-model-tts-ru-0.8-multi",
        lang="ru",
        output_folder="./tts_out",
        file_prefix="test_",
        speaker_id=4,
        speech_rate=1,
        device="cuda",
    )
    print(f"Audio written to {wav}")
