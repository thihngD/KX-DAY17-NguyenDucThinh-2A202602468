from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from agent_baseline import ACK_REPLY, message_text
from config import LabConfig, load_config
from memory_store import (
    PROFILE_FIELDS,
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
    is_recall_request,
)
from model_provider import build_chat_model

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. Hồ sơ người dùng (User.md) và tóm tắt hội thoại cũ "
    "được cung cấp bên dưới; luôn ưu tiên fact mới nhất trong User.md. "
    "Dùng tool save_user_fact khi người dùng nêu một fact ổn định mới."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.confidence_threshold = getattr(self.config, "profile_confidence_threshold", 0.6)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._current_context: AgentContext | None = None
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, message: str) -> dict[str, str]:
        """Steps 1-2: extract confident facts and persist them into User.md."""

        updates = extract_profile_updates(message, min_confidence=self.confidence_threshold)
        if not updates:
            return {}
        return self.profile_store.apply_updates(user_id, updates)

    def _account(self, thread_id: str, prompt_tokens: int, response_tokens: int) -> None:
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        memory_updates = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message)
        response_tokens = estimate_tokens(response)
        self.compact_memory.append(thread_id, "assistant", response)
        self._account(thread_id, prompt_tokens, response_tokens)
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": memory_updates,
            "compactions": self.compaction_count(thread_id),
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Prompt = User.md + compact summary + recent kept messages."""

        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        return profile_tokens + self.compact_memory.context_tokens(thread_id)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from User.md; acknowledge plain statements."""

        if not is_recall_request(message):
            return ACK_REPLY
        return answer_from_facts(message, self.profile_store.facts(user_id)) or ACK_REPLY

    # ------------------------------------------------------------------ live

    def _system_prompt(self, user_id: str, thread_id: str) -> str:
        summary = str(self.compact_memory.context(thread_id)["summary"]) or "(chưa có)"
        return (
            f"{ADVANCED_SYSTEM_PROMPT}\n\n## User.md\n{self.profile_store.read_text(user_id)}\n"
            f"## Tóm tắt hội thoại cũ\n{summary}"
        )

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        memory_updates = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        self._current_context = AgentContext(user_id, str(self.profile_store.path_for(user_id)))

        recent = self.compact_memory.context(thread_id)["messages"]
        messages = [{"role": "system", "content": self._system_prompt(user_id, thread_id)}] + list(recent)
        result = self.langchain_agent.invoke({"messages": messages})

        new_messages = result["messages"][len(messages) :]
        response = message_text(result["messages"][-1])
        prompt_tokens = 0
        response_tokens = 0
        for msg in new_messages:
            usage = getattr(msg, "usage_metadata", None) or {}
            prompt_tokens += usage.get("input_tokens", 0)
            response_tokens += usage.get("output_tokens", 0)
        prompt_tokens = prompt_tokens or self._estimate_prompt_context_tokens(user_id, thread_id)
        response_tokens = response_tokens or estimate_tokens(response)

        self.compact_memory.append(thread_id, "assistant", response)
        self._account(thread_id, prompt_tokens, response_tokens)
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": memory_updates,
            "compactions": self.compaction_count(thread_id),
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + User.md tools.

        Short-term state and compaction are handled by `CompactMemoryManager` (the prompt is
        rebuilt each turn from User.md + summary + recent messages), so the compaction count
        reported by the benchmark is the real one in both modes.
        """

        if not self.config.model.is_live_ready():
            return None
        try:
            from langchain.agents import create_agent
            from langchain_core.tools import tool
        except Exception as exc:
            print(f"[AdvancedAgent] live mode unavailable, using offline mode: {exc}", file=sys.stderr)
            return None

        agent = self

        @tool
        def read_user_memory() -> str:
            """Read the current user's User.md profile."""

            if agent._current_context is None:
                return ""
            return agent.profile_store.read_text(agent._current_context.user_id)

        @tool
        def save_user_fact(key: str, value: str) -> str:
            """Save one NEW stable fact into User.md (short value, a few words, no sentences).
            key must be one of: name, location, profession, favorite_drink, favorite_food, pet,
            interests, response_style. Existing facts are not overwritten by this tool."""

            if agent._current_context is None:
                return "No active user."
            if key not in PROFILE_FIELDS:
                return f"Unknown key '{key}'."
            changed = agent.profile_store.apply_updates(
                agent._current_context.user_id, {key: value}, overwrite_scalars=False
            )
            return f"Saved {key}." if changed else "No change (already known or value rejected)."

        try:
            return create_agent(build_chat_model(self.config.model), tools=[read_user_memory, save_user_fact])
        except Exception as exc:
            print(f"[AdvancedAgent] live mode unavailable, using offline mode: {exc}", file=sys.stderr)
            return None
