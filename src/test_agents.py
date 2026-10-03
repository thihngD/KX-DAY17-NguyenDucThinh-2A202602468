from __future__ import annotations

import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig

REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_config(tmp_path: Path) -> LabConfig:
    """Isolated config: state lives in tmp_path, tiny threshold so compaction happens quickly."""

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    # read before any write: default profile, nothing on disk yet
    assert "## Profile facts" in store.read_text("dungct")
    assert store.file_size("dungct") == 0

    path = store.write_text("dungct", "# User.md - dungct\n\n## Profile facts\n- name: DũngCT\n")
    assert path == config.state_dir / "profiles" / "dungct" / "User.md"
    assert store.facts("dungct") == {"name": "DũngCT"}
    size_after_write = store.file_size("dungct")
    assert size_after_write > 0

    assert store.edit_text("dungct", "- name: DũngCT", "- name: DũngCT\n- location: Huế") is True
    assert store.edit_text("dungct", "text-that-does-not-exist", "x") is False
    assert store.facts("dungct")["location"] == "Huế"
    assert store.file_size("dungct") > size_after_write

    # The agent writes User.md itself from a natural message.
    agent = AdvancedAgent(config, force_offline=True)
    agent.reply("agent_user", "t1", "Chào bạn, mình tên là DũngCT.")
    assert "- name: DũngCT" in store.read_text("agent_user")
    # A question about a fact must not be stored as a fact.
    agent.reply("agent_user", "t1", "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.")
    assert "favorite_drink" not in store.facts("agent_user")

    # Parallel writers (live-mode tools run in threads) must not corrupt the file.
    from concurrent.futures import ThreadPoolExecutor

    keys = ["location", "profession", "favorite_drink", "favorite_food", "pet"] * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda k: store.apply_updates("race", {k: f"value-{k}"}), keys))
    race_text = store.read_text("race")
    assert race_text.startswith("# User.md - race")
    assert set(store.facts("race")) == set(keys)

    # LLM-tool guardrails: no overwrite of a known fact, no sentence-long list items.
    store.apply_updates("race", {"favorite_drink": "cà phê"}, overwrite_scalars=False)
    assert store.facts("race")["favorite_drink"] == "value-favorite_drink"
    store.apply_updates("race", {"interests": "Python, " + "một câu rất dài " * 5})
    assert store.facts("race")["interests"] == "Python"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    agent.reply("u", "short", "Mình tên là DũngCT.")
    assert agent.compaction_count("short") == 0  # short thread: no false trigger

    long_turn = "Mình đang đọc tin tức dài về Artemis III, X-59, El Nino và kế hoạch điện sạch. " * 3
    for _ in range(4):
        agent.reply("u", "long", long_turn)

    assert agent.compaction_count("long") >= 1
    context = agent.compact_memory.context("long")
    assert context["summary"]  # older messages folded into a summary
    assert len(context["messages"]) <= config.compact_keep_messages + 1


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.",
        "Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
    ]
    for turn in turns:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    question = "Mình tên gì, hiện ở đâu và làm nghề gì?"

    # Baseline remembers inside the same thread ...
    assert "DũngCT" in baseline.reply("dungct", "session-1", question)["response"]
    # ... but forgets everything in a new thread.
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]
    assert "DũngCT" not in baseline_answer
    assert "Huế" not in baseline_answer

    # Advanced recalls via User.md, keeps the correction and ignores the joke.
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer
    assert "Đà Nẵng" not in advanced_answer
    assert "backend engineer" in advanced_answer
    assert "product manager" not in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    stress = json.loads((REPO_DATA_DIR / "advanced_long_context.json").read_text(encoding="utf-8"))[0]
    for turn in stress["turns"]:
        baseline.reply(stress["user_id"], "long", turn)
        advanced.reply(stress["user_id"], "long", turn)

    assert baseline.compaction_count("long") == 0
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    # Compaction must not cost the long-term facts: they live in User.md.
    answer = advanced.reply(stress["user_id"], "new", stress["recall_questions"][0]["question"])["response"]
    for expected in stress["recall_questions"][0]["expected_contains"]:
        assert expected in answer
