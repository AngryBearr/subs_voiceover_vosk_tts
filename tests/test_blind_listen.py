"""Blind listening tool: discovery of the four pack layouts, blind isolation, Range, ratings and decoding."""

from __future__ import annotations

import builtins
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import threading
import urllib.error
import urllib.request
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ui import blind_decode, blind_listen
from ui.blind_listen import (
    BlindListenApp,
    ManifestGroup,
    ManifestVariant,
    analyze,
    discover,
    make_server,
    parse_range,
    write_pack_manifest,
)
from utils.ffmpeg import ffmpeg_exe

SECRETS = ("SECRET_alpha_model", "SECRET_beta_model", "SECRET_gamma_model", "SECRET_delta_model",
           "Секретная легенда", "SECRET_flagged_reason")
REAL_ROOT = Path("/media/angrybear/new_large_storage/Survivor/STT_listen_2026-11-25_sound")


# ------------------------------------------------------------------------------------------ fixtures


def make_wav(path: Path, seconds: float = 0.5, freq: float = 440.0, rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * n / rate)))
                      for n in range(int(seconds * rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def make_groups_pack(folder: Path) -> Path:
    """02/03-style: sub-folders with 00_ reference + A/B/C(/D) and a groups-list key."""
    for name, letters in (("01_S51E01_Devin_short", "ABC"), ("02_S01E01_Sue_long", "ABCD")):
        make_wav(folder / name / "00_actor_original_en.wav", 0.8, 300)
        for number, letter in enumerate(letters):
            make_wav(folder / name / f"{letter}.wav", 0.5, 400 + 50 * number)
    (folder / "README.txt").write_text(
        "Слепое сравнение: чистка референсов.\n\nСписок папок:\n"
        "  01_S51E01_Devin_short/ (3 вар. A–C) — Devin (S51E01): «Нельзя.»\n"
        "  02_S01E01_Sue_long/ (4 вар. A–D) — Sue (S01E01): «Думать надо!»\n", encoding="utf-8")
    write_json(folder / "key.json", {
        "blind_set": folder.name, "seed": 1,
        "variants_legend": {"SECRET_alpha_model": "Секретная легенда"},
        "groups": [
            {"folder": "01_S51E01_Devin_short", "episode": "S51E01", "speaker": "Devin", "kind": "short",
             "text": "Нельзя.", "variants": {"A": {"variant": "SECRET_alpha_model"},
                                              "B": {"variant": "SECRET_beta_model"},
                                              "C": {"variant": "SECRET_gamma_model"}}},
            {"folder": "02_S01E01_Sue_long", "episode": "S01E01", "speaker": "Sue", "kind": "long",
             "text": "Думать надо!", "variants": {"A": {"variant": "SECRET_beta_model", "nfe": 16},
                                                   "B": {"variant": "SECRET_alpha_model"},
                                                   "C": {"variant": "SECRET_delta_model"},
                                                   "D": {"variant": "SECRET_gamma_model"}}},
        ]})
    return folder


def make_clips_pack(folder: Path) -> Path:
    """04-style: flat clips + list.txt + clips key."""
    for number in range(1, 4):
        make_wav(folder / f"clip_{number:02d}.wav", 0.3 + 0.1 * number, 500)
    (folder / "list.txt").write_text(
        "Сначала прослушайте все файлы.\n\n"
        "clip_01.wav | Спикер: Kimmi | Текст: Отскочило.\n"
        "clip_02.wav | Спикер: Jeff | Текст: Так вот.\n"
        "clip_03.wav | Спикер: не указан | Текст: текст не указан\n", encoding="utf-8")
    write_json(folder / "key.json", {"seed": 2, "selection": "SECRET_flagged_reason", "clips": [
        {"file": "clip_01.wav", "status": "flagged", "reasons": ["tail_noise"]},
        {"file": "clip_02.wav", "status": "clean", "reasons": []},
        {"file": "clip_03.wav", "status": "flagged", "reasons": ["abrupt_end"]},
    ]})
    return folder


def make_multitrack_pack(folder: Path) -> Path:
    """01-style: an .mkv with a video, two variant tracks and an "Оригинал" track, plus a scenes key."""
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "01_S51E01_loud_challenge.mkv"
    command = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
               "-f", "lavfi", "-i", "sine=frequency=660:duration=2",
               "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
               "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
               "-metadata:s:a:0", "title=Вариант A", "-metadata:s:a:1", "title=Вариант B",
               "-metadata:s:a:2", "title=Оригинал", str(video)]
    subprocess.run(command, check=True)
    (folder / "README.txt").write_text(
        "Слепое сравнение звука.\n\n01_S51E01_loud_challenge.mkv — S51E01: испытание\n", encoding="utf-8")
    write_json(folder / "key.json", {"seed": 3, "scenes": [{
        "file": video.name,
        "scene": {"id": "S51E01_loud", "episode": "S51E01", "era": "S51 modern", "scene_type": "loud_challenge"},
        "mapping": {"A": {"variant_id": "SECRET_alpha_model", "description": "Секретная легенда"},
                    "B": {"variant_id": "SECRET_beta_model", "description": "beta"}}}]})
    return folder


def make_manifest_pack(folder: Path, sources: Path) -> Path:
    """A universal pack written by write_pack_manifest from files whose names reveal the model."""
    groups = []
    for number in range(2):
        variants = [ManifestVariant(name, make_wav(sources / f"{number}_{name}.wav", 0.4, 300 + 100 * i),
                                    description=f"desc {name}", facets={"kind": "short" if number else "long"})
                    for i, name in enumerate(("SECRET_alpha_model", "SECRET_beta_model", "SECRET_gamma_model"))]
        groups.append(ManifestGroup(f"g{number + 1:02d}", variants, title=f"Реплика {number + 1}", text="«Привет»",
                                    references=[("Образец голоса", make_wav(sources / f"ref{number}.wav", 0.6))],
                                    facets={"episode": f"S0{number + 1}E01"}))
    write_pack_manifest(folder, "Шаги синтеза", groups, description="Что слушать: мыло.", seed=7)
    return folder


@pytest.fixture()
def collection_root(tmp_path: Path) -> Path:
    root = tmp_path / "STT_listen_test"
    make_multitrack_pack(root / "01_mix")
    make_groups_pack(root / "02_groups")
    make_clips_pack(root / "04_clips")
    make_manifest_pack(root / "05_manifest", tmp_path / "sources")
    (root / "README.txt").write_text("Общий README.\n", encoding="utf-8")
    return root


class Server:
    def __init__(self, app: BlindListenApp) -> None:
        self.app = app
        self.httpd = make_server(app, "127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def request(self, path: str, body: Any = None, headers: dict[str, str] | None = None
                ) -> tuple[int, dict[str, str], bytes]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def get_json(self, path: str) -> Any:
        status, _, body = self.request(path)
        assert status == 200, body
        return json.loads(body)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server(collection_root: Path, tmp_path: Path) -> Iterator[Server]:
    app = BlindListenApp(collection_root, state_root=tmp_path / "state", cache_root=tmp_path / "cache")
    srv = Server(app)
    yield srv
    srv.close()


def pack_index(app: BlindListenApp, rel_id: str) -> int:
    return next(i for i, p in enumerate(app.collection.packs) if p.rel_id == rel_id)


# ------------------------------------------------------------------------------------------ discovery


def test_discover_groups_pack(tmp_path: Path) -> None:
    collection = discover(make_groups_pack(tmp_path / "02_groups"))
    assert not collection.is_collection
    pack = collection.packs[0]
    assert (pack.kind, pack.mode) == ("groups", "comparison")
    assert [g.id for g in pack.groups] == ["01_S51E01_Devin_short", "02_S01E01_Sue_long"]
    first, second = pack.groups
    assert [v.id for v in first.variants] == ["A", "B", "C"] and first.scale_max == 3
    assert [v.id for v in second.variants] == ["A", "B", "C", "D"] and second.scale_max == 4
    assert [r.label for r in first.references] == ["Ориентир: actor original en"]
    assert first.caption == "Devin (S51E01): «Нельзя.»"
    assert first.sync and all(v.sync for v in first.variants)
    assert not first.references[0].sync  # the reference is longer: played on its own


def test_discover_clips_pack(tmp_path: Path) -> None:
    pack = discover(make_clips_pack(tmp_path / "04_clips")).packs[0]
    assert (pack.kind, pack.mode) == ("clips", "single")
    assert len(pack.groups) == 1
    group = pack.groups[0]
    assert (group.scale_min, group.scale_max) == (1, 5)
    assert [v.id for v in group.variants] == ["clip_01", "clip_02", "clip_03"]
    assert group.variants[0].label == "clip_01 — Спикер: Kimmi | Текст: Отскочило."
    assert "Обрыв" in pack.tags and pack.scale_labels["1"] == "явный брак"


def test_discover_multitrack_pack_and_cache(tmp_path: Path) -> None:
    pack = discover(make_multitrack_pack(tmp_path / "01_mix")).packs[0]
    assert pack.kind == "multitrack"
    group = pack.groups[0]
    assert [v.id for v in group.variants] == ["A", "B"] and group.scale_max == 2
    assert [r.label for r in group.references] == ["Оригинал"]
    assert group.sync and group.video is not None
    assert group.caption == "S51E01: испытание"
    cache = tmp_path / "cache"
    blind_listen.build_multitrack_cache(group, cache)
    paths = [item.path for item in group.variants + group.references]
    assert all(p is not None and p.is_file() and p.suffix == ".m4a" for p in paths)
    assert group.video.path is not None and group.video.path.suffix == ".mp4"
    video_streams = blind_listen.probe_streams(group.video.path)
    assert [s.codec_type for s in video_streams] == ["video"]  # muted copy
    assert all(len(blind_listen.audio_streams(p)) == 1 for p in paths if p is not None)
    assert blind_listen.build_multitrack_cache(group, cache) == 0.0  # second time: from the cache
    assert not (tmp_path / "01_mix" / ".listen_cache").exists()


def test_manifest_pack_hides_names(tmp_path: Path) -> None:
    folder = make_manifest_pack(tmp_path / "pack", tmp_path / "sources")
    manifest_text = (folder / "blind_pack.json").read_text(encoding="utf-8")
    assert not any(secret in manifest_text for secret in SECRETS)
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    assert key["schema"] == "blind_key/1"
    assert sorted(v["name"] for v in key["groups"]["g01"]["variants"].values()) == [
        "SECRET_alpha_model", "SECRET_beta_model", "SECRET_gamma_model"]
    pack = discover(folder).packs[0]
    assert (pack.kind, pack.title) == ("manifest", "Шаги синтеза")
    group = pack.groups[0]
    assert [v.id for v in group.variants] == ["A", "B", "C"]
    assert all(v.path is not None and v.path.name == f"{v.id}.wav" for v in group.variants)
    assert group.references[0].label == "Образец голоса" and group.sync


def test_manifest_rejects_files_outside_pack(tmp_path: Path) -> None:
    folder = tmp_path / "pack"
    make_wav(folder / "A.wav")
    make_wav(tmp_path / "outside.wav")
    write_json(folder / "blind_pack.json", {"schema": "blind_pack/1", "title": "t", "groups": [
        {"id": "g", "variants": [{"id": "A", "file": "A.wav"}, {"id": "B", "file": "../outside.wav"}]}]})
    group = discover(folder).packs[0].groups[0]
    assert [v.id for v in group.variants] == ["A"]


def test_discover_collection(collection_root: Path) -> None:
    collection = discover(collection_root)
    assert collection.is_collection and collection.readme == "Общий README."
    assert [(p.rel_id, p.kind) for p in collection.packs] == [
        ("01_mix", "multitrack"), ("02_groups", "groups"), ("04_clips", "clips"), ("05_manifest", "manifest")]


def test_find_caption_variants() -> None:
    lines = ["  01_a/ (7 вар. A–G) — Carter: «Тут»", "01_ab — wrong", "clip_1.wav | Спикер: X | Текст: Y",
             "scene.mkv — S51: тихая сцена"]
    assert blind_listen.find_caption(lines, "01_a") == "Carter: «Тут»"
    assert blind_listen.find_caption(lines, "01_ab") == "wrong"
    assert blind_listen.find_caption(lines, "clip_1.wav") == "Спикер: X | Текст: Y"
    assert blind_listen.find_caption(lines, "scene.mkv", "scene") == "S51: тихая сцена"
    assert blind_listen.find_caption(lines, "missing") == ""


# ------------------------------------------------------------------------------------------ blind isolation


def test_no_key_data_before_decode(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    touched: list[str] = []
    real_open = builtins.open
    real_path_open = pathlib.Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)) and Path(file).name == "key.json":
            touched.append(str(file))
        return real_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "key.json":
            touched.append(str(self))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)

    bodies: list[bytes] = []
    for path in ("/", "/api/packs"):
        status, _, body = server.request(path)
        assert status == 200
        bodies.append(body)
    for index, pack in enumerate(server.app.collection.packs):
        view = server.get_json(f"/api/pack?pack={index}")
        bodies.append(json.dumps(view, ensure_ascii=False).encode("utf-8"))
        for group in view["groups"]:
            first = group["variants"][0]
            server.request("/api/ratings", {"pack": index, "group": group["id"],
                                             "state": {"scores": {first["id"]: 1}, "comment": "ok"}})
            urls = [item["url"] for item in group["variants"] + group["references"]]
            if group["video"]:
                urls.append(group["video"]["url"])
            for url in urls:
                status, _, body = server.request(url)
                assert status == 200
                bodies.append(body)
        status, _, body = server.request(f"/api/ratings?pack={index}")
        bodies.append(body)
        status, _, body = server.request(f"/api/results?pack={index}")
        assert status == 404
        bodies.append(body)
        status, _, body = server.request("/api/decode", {"pack": index})  # no confirmation
        assert status == 400
        bodies.append(body)

    assert touched == []
    for body in bodies:
        text = body.decode("utf-8", errors="ignore")
        assert not any(secret in text for secret in SECRETS)

    # explicit decode opens the key and shows the names
    index = pack_index(server.app, "02_groups")
    status, _, body = server.request("/api/decode", {"pack": index, "confirm": True})
    assert status == 200
    assert "SECRET_alpha_model" in body.decode("utf-8")
    assert touched  # only now
    assert server.get_json("/api/packs")["packs"][index]["decoded"] is True
    assert "SECRET_alpha_model" in server.get_json(f"/api/results?pack={index}")["markdown"]


def test_cross_site_post_rejected(server: Server) -> None:
    status, _, _ = server.request("/api/decode", {"pack": 0, "confirm": True},
                                  headers={"Origin": "http://evil.example"})
    assert status == 403


# ------------------------------------------------------------------------------------------ HTTP media


def test_parse_range() -> None:
    assert parse_range(None, 100) is None
    assert parse_range("bytes=0-9", 100) == (0, 9)
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=50-500", 100) == (50, 99)
    assert parse_range("bytes=0-1,5-6", 100) is None
    with pytest.raises(ValueError):
        parse_range("bytes=100-", 100)
    with pytest.raises(ValueError):
        parse_range("bytes=9-3", 100)


def test_range_requests(server: Server) -> None:
    index = pack_index(server.app, "02_groups")
    url = server.get_json(f"/api/pack?pack={index}")["groups"][0]["variants"][0]["url"]
    file_path = server.app.collection.packs[index].groups[0].variants[0].path
    assert file_path is not None
    data = file_path.read_bytes()
    status, headers, body = server.request(url)
    assert status == 200 and body == data and headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "audio/wav"
    status, headers, body = server.request(url, headers={"Range": "bytes=10-109"})
    assert status == 206 and body == data[10:110]
    assert headers["Content-Range"] == f"bytes 10-109/{len(data)}"
    status, headers, body = server.request(url, headers={"Range": "bytes=-16"})
    assert status == 206 and body == data[-16:]
    status, headers, _ = server.request(url, headers={"Range": f"bytes={len(data) + 5}-"})
    assert status == 416 and headers["Content-Range"] == f"bytes */{len(data)}"


def test_multitrack_media_served_from_cache(server: Server, tmp_path: Path) -> None:
    index = pack_index(server.app, "01_mix")
    group = server.get_json(f"/api/pack?pack={index}")["groups"][0]
    assert group["needs_prepare"] and group["sync"]
    assert server.request("/api/prepare", {"pack": index, "group": 0})[0] == 200
    status, headers, body = server.request(group["video"]["url"], headers={"Range": "bytes=0-99"})
    assert status == 206 and len(body) == 100 and headers["Content-Type"] == "video/mp4"
    status, headers, _ = server.request(group["references"][0]["url"])
    assert status == 200 and headers["Content-Type"] == "audio/mp4"
    assert list((tmp_path / "cache").rglob("*.m4a"))


# ------------------------------------------------------------------------------------------ ratings


def test_ratings_saved_and_cleaned(server: Server, tmp_path: Path) -> None:
    index = pack_index(server.app, "02_groups")
    state = {"scores": {"A": 3, "B": 9, "Z": 1, "C": True}, "notes": {"A": "чисто", "ref1": "тембр"},
             "tags": {"A": ["x"]}, "comment": "B шумит", "listened": ["A", "ref1", "nope"]}
    status, _, body = server.request("/api/ratings", {"pack": index, "group": "01_S51E01_Devin_short",
                                                      "state": state})
    assert status == 200
    reply = json.loads(body)
    assert reply["state"]["scores"] == {"A": 3}  # out of scale, unknown and bool scores dropped
    assert reply["state"]["notes"] == {"A": "чисто", "ref1": "тембр"}
    assert reply["state"]["listened"] == ["A", "ref1"]
    assert reply["progress"] == {"done": 0, "total": 2}
    saved_file = tmp_path / "state" / "02_groups" / "ratings.json"
    saved = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved["groups"]["01_S51E01_Devin_short"]["comment"] == "B шумит"
    full = {"scores": {"A": 3, "B": 1, "C": 2}}
    reply = json.loads(server.request("/api/ratings", {"pack": index, "group": "01_S51E01_Devin_short",
                                                       "state": full})[2])
    assert reply["progress"] == {"done": 1, "total": 2}
    assert server.request("/api/settings", {"pack": index, "unique_ranks": True})[0] == 200
    again = BlindListenApp(server.app.collection.root, state_root=tmp_path / "state")
    view = again.pack_view(index)
    assert view["unique_ranks"] is True
    assert view["ratings"]["01_S51E01_Devin_short"]["scores"] == {"A": 3, "B": 1, "C": 2}
    status, _, _ = server.request("/api/ratings", {"pack": index, "group": "nope", "state": {}})
    assert status == 404


# ------------------------------------------------------------------------------------------ decoding


def decode(pack: blind_listen.Pack, groups: dict[str, Any], out: Path) -> dict[str, Any]:
    ratings = blind_listen.empty_ratings(pack)
    ratings["groups"] = groups
    return blind_decode.decode_pack(pack, ratings, out)


def test_decode_groups_schema(tmp_path: Path) -> None:
    pack = discover(make_groups_pack(tmp_path / "02_groups")).packs[0]
    results = decode(pack, {
        "01_S51E01_Devin_short": {"scores": {"A": 3, "B": 1, "C": 3}, "notes": {"B": "шипит"},
                                  "comment": "A и C одинаковые"},
        "02_S01E01_Sue_long": {"scores": {"A": 4, "B": 2, "C": 1, "D": 3}},
    }, tmp_path / "out")
    stats = {s["name"]: s for s in results["summary"]["by_variant"]}
    alpha, beta, gamma, delta = (stats[f"SECRET_{n}_model"] for n in ("alpha", "beta", "gamma", "delta"))
    assert alpha["n"] == 2 and alpha["mean"] == 2.5 and alpha["median"] == 2.5
    assert alpha["top_1_count"] == 1 and alpha["mean_norm"] == pytest.approx((1 + 1 / 3) / 2, abs=1e-4)
    assert alpha["description"] == "Секретная легенда"
    assert beta["top_1_count"] == 1 and beta["last_count"] == 1 and beta["mean"] == 2.5
    assert gamma["top_1_count"] == 1  # tie for first place in group 1 counts for both
    assert delta["last_count"] == 1 and delta["n"] == 1
    assert gamma["mean_norm"] == pytest.approx((1 + 2 / 3) / 2, abs=1e-4)
    assert [s["name"] for s in results["summary"]["by_variant"]][:2] == ["SECRET_gamma_model", "SECRET_alpha_model"]
    by_facet = results["summary"]["by_facet"]
    assert set(by_facet) == {"episode", "kind"}
    assert by_facet["episode"]["S01E01"][0]["name"] == "SECRET_beta_model"
    first = results["groups"][0]
    assert first["winners"] == ["SECRET_alpha_model", "SECRET_gamma_model"]
    assert first["comment"] == "A и C одинаковые"
    assert results["notes"] == [{"group": "01_S51E01_Devin_short", "blind": "B", "name": "SECRET_beta_model",
                                 "note": "шипит"}]
    md = (tmp_path / "out" / "results_decoded.md").read_text(encoding="utf-8")
    assert "Общий рейтинг вариантов" in md and "| 1 | **SECRET_gamma_model** |" in md
    assert "A и C одинаковые" in md and "шипит" in md and "По категории «Серия»" in md
    saved = json.loads((tmp_path / "out" / "results_decoded.json").read_text(encoding="utf-8"))
    assert saved["key_schema"] == "groups" and saved["progress"] == {"done": 2, "total": 2}


def test_decode_scenes_schema(tmp_path: Path) -> None:
    pack = discover(make_multitrack_pack(tmp_path / "01_mix")).packs[0]
    results = decode(pack, {"01_S51E01_loud_challenge.mkv": {"scores": {"A": 1, "B": 2},
                                                             "notes": {"ref1": "оригинал громче"}}}, tmp_path)
    assert results["key_schema"] == "scenes"
    assert results["groups"][0]["winners"] == ["SECRET_beta_model"]
    assert results["groups"][0]["facets"] == {"episode": "S51E01", "era": "S51 modern",
                                              "scene_type": "loud_challenge"}
    assert results["summary"]["by_variant"][0]["name"] == "SECRET_beta_model"
    assert any(n["note"] == "оригинал громче" for n in results["notes"])


def test_decode_clips_schema(tmp_path: Path) -> None:
    pack = discover(make_clips_pack(tmp_path / "04_clips")).packs[0]
    results = decode(pack, {"all": {"scores": {"clip_01": 1, "clip_02": 5}, "tags": {"clip_03": ["Обрыв"]},
                                    "notes": {"clip_01": "хвост"}}}, tmp_path)
    assert results["key_schema"] == "clips" and results["mode"] == "single"
    crosstab = results["summary"]["ear_vs_key"]
    assert crosstab["flagged"]["defect"] == 2 and crosstab["clean"]["clean"] == 1
    reasons = results["summary"]["by_facet"]["reason"]
    assert reasons["tail_noise"][0]["mean"] == 1 and reasons["abrupt_end"][0]["defect"] == 1
    md = (tmp_path / "results_decoded.md").read_text(encoding="utf-8")
    assert "Сводка по ключу" in md and "| clip_01 | 1 |" in md and "хвост" in md


def test_decode_manifest_schema(tmp_path: Path) -> None:
    pack = discover(make_manifest_pack(tmp_path / "pack", tmp_path / "sources")).packs[0]
    key = json.loads((tmp_path / "pack" / "key.json").read_text(encoding="utf-8"))
    best = {gid: next(letter for letter, v in g["variants"].items() if v["name"] == "SECRET_gamma_model")
            for gid, g in key["groups"].items()}
    groups = {gid: {"scores": {letter: (3 if letter == best[gid] else 1) for letter in "ABC"}} for gid in best}
    results = decode(pack, groups, tmp_path / "out")
    assert results["key_schema"] == "blind_key/1"
    top = results["summary"]["by_variant"][0]
    assert top["name"] == "SECRET_gamma_model" and top["top_1_count"] == 2 and top["mean_norm"] == 1.0
    assert set(results["summary"]["by_facet"]) == {"episode", "kind"}


def test_analyze_cli_collection(collection_root: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    pack = discover(collection_root / "02_groups").packs[0]
    (state / "02_groups").mkdir(parents=True)
    write_json(state / "02_groups" / "ratings.json", {"groups": {"01_S51E01_Devin_short": {"scores": {"A": 3}}}})
    assert pack.groups
    assert blind_listen.main([str(collection_root), "--analyze", "--state-dir", str(state)]) == 0
    for rel in ("01_mix", "02_groups", "04_clips", "05_manifest"):
        assert (state / rel / "results_decoded.json").is_file()
        assert (state / rel / "results_decoded.md").is_file()
    combined = (state / "results_decoded.md").read_text(encoding="utf-8")
    assert combined.startswith("# Итоги слепого прослушивания: STT_listen_test")
    assert "## Итоги слепого прослушивания: 02_groups" in combined
    assert not (collection_root / "02_groups" / "results_decoded.md").exists()
    assert not any(collection_root.rglob("ratings.json"))


def test_analyze_default_writes_next_to_pack(tmp_path: Path) -> None:
    folder = make_groups_pack(tmp_path / "02_groups")
    written = analyze(folder)
    assert written == [folder / "results_decoded.md"] and written[0].is_file()


# ------------------------------------------------------------------------------------------ real pack (read-only)


def _snapshot(root: Path) -> dict[str, tuple[int, float]]:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime) for p in root.rglob("*")}


@pytest.mark.skipif(not REAL_ROOT.is_dir(), reason="today's listening folder is not mounted")
def test_real_listening_folder_read_only(tmp_path: Path) -> None:
    before = _snapshot(REAL_ROOT)
    collection = discover(REAL_ROOT)
    assert collection.is_collection
    kinds = {p.rel_id: (p.kind, p.mode, len(p.groups)) for p in collection.packs}
    assert kinds == {"01_mix_separators_and_dsp": ("multitrack", "comparison", 9),
                     "02_reference_cleaning": ("groups", "comparison", 12),
                     "03_espeech_epss_steps": ("groups", "comparison", 16),
                     "04_detected_artifacts": ("clips", "single", 1)}
    packs = {p.rel_id: p for p in collection.packs}
    assert all(len(g.variants) == 6 and [r.label for r in g.references] == ["Оригинал"]
               for g in packs["01_mix_separators_and_dsp"].groups)
    assert all(len(g.variants) in (6, 7) and len(g.references) == 1 for g in packs["02_reference_cleaning"].groups)
    assert all(len(g.variants) == 4 and len(g.references) == 1 for g in packs["03_espeech_epss_steps"].groups)
    assert len(packs["04_detected_artifacts"].groups[0].variants) == 16
    assert all(g.caption for p in collection.packs if p.kind != "clips" for g in p.groups)

    # the page data holds no key data
    app = BlindListenApp(REAL_ROOT, state_root=tmp_path / "state", cache_root=tmp_path / "cache")
    views = json.dumps([app.pack_view(i) for i in range(len(collection.packs))], ensure_ascii=False)
    for pack in collection.packs:
        key = blind_decode.load_key(pack)
        secrets = {text for group in key.groups.values() for v in group.variants.values()
                   for text in (v.name, v.description) if len(text) > 4}
        leaked = [text for text in secrets if re.search(rf"(?<![\w]){re.escape(text)}(?![\w])", views)]
        assert secrets and not leaked, pack.rel_id

    # split one real scene into the temporary cache
    app.prepare_group(0, 0)
    assert len(list((tmp_path / "cache").rglob("*.m4a"))) == 7

    # synthetic ratings in a temporary state folder decode against every real key
    for pack in collection.packs:
        groups = {g.id: {"scores": {v.id: 1 + n % g.scale_max for n, v in enumerate(g.variants)}}
                  for g in pack.groups}
        folder = tmp_path / "state" / pack.rel_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "ratings.json", {"groups": groups})
    analyze(REAL_ROOT, tmp_path / "state")
    for pack in collection.packs:
        results = json.loads((tmp_path / "state" / pack.rel_id / "results_decoded.json").read_text(encoding="utf-8"))
        names = [r["name"] for g in results["groups"] for r in g["variants"]]
        assert names and not any(name.startswith("нет в ключе") for name in names), pack.rel_id
        assert results["progress"]["done"] == results["progress"]["total"]
    assert _snapshot(REAL_ROOT) == before


def test_duplicate_detection_in_manifest_api_and_cli(tmp_path: Path) -> None:
    src = tmp_path / "src"
    same_a = make_wav(src / "m1.wav", 0.4, 440)
    same_b = make_wav(src / "m2.wav", 0.4, 440)
    diff_c = make_wav(src / "m3.wav", 0.4, 660)
    group = ManifestGroup("g01", [ManifestVariant("m1", same_a), ManifestVariant("m2", same_b),
                                  ManifestVariant("m3", diff_c)])
    with pytest.raises(ValueError, match="Duplicate audio variants"):
        write_pack_manifest(tmp_path / "reject_pack", "Dup test", [group], seed=1)

    pack_dir = tmp_path / "dup_pack"
    write_pack_manifest(pack_dir, "Dup allowed", [group], seed=1, allow_duplicates=True)
    app = BlindListenApp(pack_dir, state_root=tmp_path / "state")
    view = app.pack_view(0)
    assert view["groups"][0]["duplicates"] and len(view["groups"][0]["duplicates"][0]) == 2
    report = blind_listen.check_duplicates(pack_dir)
    assert report["ok"] is False and report["packs"][0]["issues"][0]["exact"]
    assert blind_listen.main([str(pack_dir), "--check-duplicates"]) == 1

    clean_pack = make_groups_pack(tmp_path / "clean_pack")
    clean_app = BlindListenApp(clean_pack, state_root=tmp_path / "state_clean")
    assert clean_app.pack_view(0)["groups"][0]["duplicates"] == []
    assert blind_listen.main([str(clean_pack), "--check-duplicates"]) == 0

