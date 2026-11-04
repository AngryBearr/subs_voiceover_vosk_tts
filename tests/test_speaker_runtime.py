"""Process-local speaker selection must propagate without sharing model-dependent caches."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from stt import runner
from stt.line_gender import runtime_classifier
from stt.refine_speakers import model_available, model_cache_key, runtime_checkpoint
from stt.run import Episode
from stt.season_voices import embedding_path


def test_runtime_selection_and_cache_isolation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("STT_SPEAKER_CHECKPOINT", raising=False)
    assert runtime_checkpoint() is None
    assert model_cache_key("resnet293-ft") == "resnet293-ft"
    paths = [tmp_path / name for name in ("old.pt", "new.pt")]
    keys = []
    for path, value in zip(paths, (b"old weights", b"new weights")):
        path.write_bytes(value)
        monkeypatch.setenv("STT_SPEAKER_CHECKPOINT", str(path))
        assert runtime_checkpoint() == path
        assert model_available("resnet293-ft")
        key = model_cache_key("resnet293-ft")
        assert hashlib.sha256(value).hexdigest()[:16] in key
        assert embedding_path(tmp_path, "resnet293-ft").name == f"words.emb.{key}.npz"
        assert model_cache_key("resnet293") == "resnet293"
        keys.append(key)
    assert keys[0] != keys[1]
    result = subprocess.run([sys.executable, "-c", "import json; from stt.refine_speakers import "
                             "model_cache_key; print(json.dumps(model_cache_key('resnet293-ft')))"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == keys[-1]


def test_worker_environments_preserve_selection(tmp_path: Path, monkeypatch) -> None:
    weights, classifier = tmp_path / "model.pt", tmp_path / "gender.npz"
    weights.write_bytes(b"weights")
    np.savez(classifier, w=np.zeros(256))
    monkeypatch.setenv("STT_SPEAKER_CHECKPOINT", str(weights))
    monkeypatch.setenv("STT_GENDER_CLASSIFIER", str(classifier))
    assert runtime_classifier() == classifier
    with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        runner.run_module("subs_env", "stt.worker", ["--help"], tmp_path / "worker.log")
        assert run.call_args.kwargs["env"]["STT_SPEAKER_CHECKPOINT"] == str(weights)
        assert run.call_args.kwargs["env"]["STT_GENDER_CLASSIFIER"] == str(classifier)
    episode = object.__new__(Episode)
    episode.out = tmp_path
    with patch("stt.run.dotenv", return_value={}), \
            patch("stt.run.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
        episode._module("line_gender", "stt.line_gender", [str(tmp_path)])
        assert run.call_args.kwargs["env"]["STT_SPEAKER_CHECKPOINT"] == str(weights)
        assert run.call_args.kwargs["env"]["STT_GENDER_CLASSIFIER"] == str(classifier)


def test_cluster_and_review_embeddings_use_runtime_backbone(tmp_path: Path, monkeypatch) -> None:
    from stt.voice_bank import SegmentEmbedder

    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    monkeypatch.setenv("STT_SPEAKER_CHECKPOINT", str(path))
    with patch("stt.refine_speakers.WespeakerTorch") as model:
        model.return_value.return_value = np.ones(256)
        embed = SegmentEmbedder("cpu")
        assert embed(np.ones(24000, dtype=np.float32), 24000).shape == (256,)
        model.assert_called_once_with("resnet293-ft", "cpu")
        assert len(model.return_value.call_args.args[0]) == 16000


def test_full_mix_keeps_original_tail(tmp_path: Path) -> None:
    from stt import mix

    voice = np.ones(100, dtype=np.float32) * 0.1
    original = np.ones((200, 2), dtype=np.float32) * 0.1
    for full in (False, True):
        with patch.object(mix.sf, "read", return_value=(voice, 100)), \
                patch.object(mix, "load_audio", return_value=original) as load, \
                patch.object(mix.sf, "write") as write:
            mix.main(["source.mkv", "voice.wav", "--out", str(tmp_path / "mix.wav"),
                      "--sample-rate", "100", *(["--full-duration"] if full else [])])
            assert load.call_args.kwargs["duration"] == (None if full else 1.0)
            assert len(write.call_args.args[1]) == 200


def test_gender_backbone_mismatch_fails_before_inference(tmp_path: Path, monkeypatch) -> None:
    from stt.line_gender import voice_p_female

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("STT_SPEAKER_CHECKPOINT", str(weights))
    monkeypatch.delenv("STT_GENDER_CLASSIFIER", raising=False)
    offline = patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline"))
    with offline, pytest.raises(ValueError, match="requires a paired"):  # the local fallback pair would mismatch
        voice_p_female([], np.zeros(0), 0, None, "cpu")
    classifier = tmp_path / "gender.npz"
    np.savez(classifier, embedding_sha256="wrong-backbone")
    monkeypatch.setenv("STT_GENDER_CLASSIFIER", str(classifier))
    with pytest.raises(ValueError, match="does not match"):
        voice_p_female([], np.zeros(0), 0, None, "cpu")


def test_gender_files_download_the_pinned_published_pair(tmp_path: Path, monkeypatch) -> None:
    from stt import line_gender

    monkeypatch.delenv("STT_GENDER_CLASSIFIER", raising=False)
    monkeypatch.delenv("STT_SPEAKER_CHECKPOINT", raising=False)
    with patch("huggingface_hub.hf_hub_download", side_effect=lambda repo, name, revision: str(tmp_path / name)) as get:
        classifier, weights = line_gender.gender_files()
    assert (classifier.name, weights.name) == ("gender.npz", "avg_model.pt")
    assert {c.args[0] for c in get.call_args_list} == {line_gender.GENDER_REPO}
    assert {c.kwargs["revision"] for c in get.call_args_list} == {line_gender.GENDER_REVISION}


def test_gender_files_offline_fall_back_to_the_local_pair(monkeypatch) -> None:
    from stt import line_gender

    monkeypatch.delenv("STT_GENDER_CLASSIFIER", raising=False)
    monkeypatch.delenv("STT_SPEAKER_CHECKPOINT", raising=False)
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        assert line_gender.gender_files() == (line_gender.CLASSIFIER, line_gender.VOICE_WEIGHTS)
