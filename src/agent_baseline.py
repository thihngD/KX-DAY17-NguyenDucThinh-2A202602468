from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_facts, estimate_tokens, extract_profile_updates, is_recall_request
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Bạn chỉ nhớ những gì có trong cuộc trò chuyện hiện tại. "
    "Nếu không có thông tin trong cuộc trò chuyện này thì nói rõ là chưa có."
)
ACK_REPLY = "Đã ghi nhận."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def message_text(message: Any) -> str:
    """Normalize LangChain message content (str or list of content blocks) to text."""

    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
        return "".join(parts)
    return str(content)


class BaselineAgent:
    """Agent A: within-session memory only.

    - Sessions are keyed by `thread_id` (never `user_id`), so a new thread starts empty.
    - No `User.md`, no compaction: the whole thread is carried as prompt every turn.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # Prompt = the full thread history, re-sent every turn (no compaction).
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)

        response = ACK_REPLY
        if is_recall_request(message):
            # Within-session memory only: facts come from earlier user turns of THIS thread.
            facts: dict[str, str] = {}
            for previous in session.messages[:-1]:
                if previous["role"] == "user":
                    facts.update(extract_profile_updates(previous["content"]))
            response = answer_from_facts(message, facts) or ACK_REPLY

        response_tokens = estimate_tokens(response)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "offline",
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        ai_message = result["messages"][-1]
        response = message_text(ai_message)
        usage = getattr(ai_message, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or sum(estimate_tokens(m["content"]) for m in session.messages)
        response_tokens = usage.get("output_tokens") or estimate_tokens(response)

        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Build `create_agent` + `InMemorySaver` (thread-scoped memory) when a provider is configured."""

        if not self.config.model.is_live_ready():
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                build_chat_model(self.config.model),
                tools=[],
                system_prompt=BASELINE_SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
        except Exception as exc:  # missing SDK / bad credentials -> stay offline, but say so
            print(f"[BaselineAgent] live mode unavailable, using offline mode: {exc}", file=sys.stderr)
            return None
