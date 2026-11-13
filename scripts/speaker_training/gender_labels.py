"""Gender of every person of the speaker training set (output/exp_spk/data_p2): from the season cast files where the pipeline made one,
else asked to gpt-6.1-sol through the subscription (the show, the season and the name the subtitles use). Writes people_gender.json (the input of train_gender.py).

    subs_env/bin/python scripts/speaker_training/gender_labels.py   # from the repository root"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from voiceover.stress_check import _client  # noqa: E402

classes = json.loads(Path("output/exp_spk/data_p2/index.json").read_text(encoding="utf-8"))["classes"]
SHOW = {"S": "Survivor (US) season {n}", "SAU": "Australian Survivor (the 2016 season, host Jonathan LaPaglia)", "ST": "The Amazing Race (US) season {n}"}


def season_name(code: str) -> str:
    if code.startswith("SAU"):
        return SHOW["SAU"]
    if code.startswith("ST"):
        return SHOW["ST"].format(n=int(code[2:]))
    return SHOW["S"].format(n=int(code[1:]))


people = [c for c in classes if c != "_HOST"]
SYSTEM = ("You know reality TV casts. For each item give the gender of the person the subtitles name (an upper-case first name, "
          "a nickname or a surname, sometimes 'FIRST LAST') in that show and season: contestant, host or crew. Answer JSON "
          '{"items": [{"id": <id>, "who": "<full name if you know it>", "gender": "male|female|unknown"}]}; "unknown" when you are not sure.')
llm = json.loads(Path("voiceover_config.json").read_text(encoding="utf-8"))["verify"]["llm"]
client = _client(llm, "opencode+openrouter:openai/gpt-6.1-sol", Path(".env"))
out = {"_HOST": "male"}
for start in range(0, len(people), 80):
    chunk = people[start:start + 80]
    items = [{"id": i, "show": season_name(c.split(":")[0]), "name": c.split(":", 1)[1]} for i, c in enumerate(chunk)]
    result = client.complete_json(SYSTEM, json.dumps({"items": items}, ensure_ascii=False), prompt_version="gender-v1", tag=f"gender:{start}")
    for a in result.content.get("items", []):
        if isinstance(a, dict) and isinstance(a.get("id"), int) and a["id"] < len(chunk):
            out[chunk[a["id"]]] = {"gender": a.get("gender"), "who": a.get("who")}
Path("output/exp_gender/people_gender.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
from collections import Counter
print(len(out), Counter((v if isinstance(v, str) else v.get("gender")) for v in out.values()), flush=True)
