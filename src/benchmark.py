from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import estimate_tokens

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        conversations = json.load(handle)
    for conv in conversations:
        missing = {"id", "user_id", "turns", "recall_questions"} - conv.keys()
        if missing:
            raise ValueError(f"{path}: conversation {conv.get('id')} missing {sorted(missing)}")
    return conversations


def _hits(answer: str, expected: list[str]) -> int:
    lowered = (answer or "").casefold()
    return sum(1 for item in expected if item.casefold() in lowered)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1], same formula for both agents.

    0.8 * fact coverage (fraction of expected strings present)
    + 0.2 * brevity (full score up to 60 tokens, linearly down to 0 at 180 tokens;
      the dataset user repeatedly asks for short answers).
    """

    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    tokens = estimate_tokens(answer)
    brevity = 1.0 if tokens <= 60 else max(0.0, 1.0 - (tokens - 60) / 120)
    return round(0.8 * coverage + 0.2 * brevity, 4)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    details: list[dict[str, Any]] | None = None,
) -> BenchmarkRow:
    """Feed every conversation, then ask its recall questions in a fresh thread.

    Token columns sum every thread the agent handled (chat + recall threads).
    Memory growth = User.md bytes after the run minus bytes before, per user.
    """

    users = sorted({conv["user_id"] for conv in conversations})
    memory_size = getattr(agent, "memory_file_size", lambda _user: 0)
    size_before = {user: memory_size(user) for user in users}

    thread_ids: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        chat_thread = f"{conv['id']}::chat"
        thread_ids.append(chat_thread)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], chat_thread, turn)

        # Same fresh-thread naming for both agents -> genuinely cross-session.
        recall_thread = f"{conv['id']}::recall"
        thread_ids.append(recall_thread)
        for item in conv["recall_questions"]:
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            points = recall_points(answer, item["expected_contains"])
            quality = heuristic_quality(answer, item["expected_contains"])
            recall_scores.append(points)
            quality_scores.append(quality)
            if details is not None:
                details.append(
                    {
                        "agent": agent_name,
                        "conversation": conv["id"],
                        "question": item["question"],
                        "expected": item["expected_contains"],
                        "answer": answer,
                        "recall": points,
                        "quality": quality,
                    }
                )

    growth = sum(memory_size(user) - size_before[user] for user in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in thread_ids),
        recall_score=round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=sum(agent.compaction_count(t) for t in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(
    suite_name: str, dataset: Path, config: LabConfig, force_offline: bool, verbose: bool
) -> list[BenchmarkRow]:
    """Run Baseline and Advanced on the same dataset with an isolated, freshly reset state dir."""

    suite_state = config.state_dir / "benchmark" / suite_name
    if suite_state.exists():
        shutil.rmtree(suite_state)
    suite_config = replace(config, state_dir=suite_state)
    conversations = load_conversations(dataset)

    details: list[dict[str, Any]] | None = [] if verbose else None
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline), conversations, suite_config, details),
        run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline), conversations, suite_config, details),
    ]
    if details:
        for item in details:
            print(f"[{item['agent']}] {item['conversation']} recall={item['recall']} quality={item['quality']:.2f}")
            print(f"  Q: {item['question']}")
            print(f"  expected: {item['expected']}")
            print("  A: " + item["answer"].replace("\n", "\n     "))
        print()
    return rows


def main() -> None:
    """Run the Standard benchmark and the Long-Context Stress benchmark."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced")
    parser.add_argument("--live", action="store_true", help="call the configured LLM instead of offline mode")
    parser.add_argument("--verbose", action="store_true", help="print every recall question and answer")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    force_offline = not args.live
    mode = "live" if args.live else "offline"

    suites = [
        ("Standard Benchmark", "standard", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", "long_context", config.data_dir / "advanced_long_context.json"),
    ]
    print(
        f"Mode: {mode} | compact_threshold_tokens={config.compact_threshold_tokens} "
        f"| compact_keep_messages={config.compact_keep_messages} "
        f"| profile_confidence_threshold={config.profile_confidence_threshold}\n"
    )
    for title, suite_name, dataset in suites:
        rows = run_suite(suite_name, dataset, config, force_offline, args.verbose)
        print(f"## {title} ({dataset.name})\n")
        print(format_rows(rows))
        print()


if __name__ == "__main__":
    main()
