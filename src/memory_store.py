from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token, 0 for empty text."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# Structured profile schema (entity extraction)
# ---------------------------------------------------------------------------

# Order matters: it is also the order facts are written to User.md and recalled.
PROFILE_FIELDS = (
    "name",
    "location",
    "profession",
    "favorite_drink",
    "favorite_food",
    "pet",
    "interests",
    "response_style",
)

# List fields accumulate values; scalar fields are replaced by the newest value
# (conflict handling: a correction overwrites the old fact instead of coexisting).
LIST_FIELDS = {"interests", "response_style"}

# Guardrails against memory bloat / junk values (found in live mode, where the LLM
# tried to store whole sentences as "interests"). A list keeps its newest items.
MAX_SCALAR_CHARS = 60
MAX_LIST_ITEM_CHARS = 40
MAX_LIST_ITEMS = 8

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm kỹ thuật",
    "response_style": "Style trả lời",
}


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user id."""

    root_dir: Path
    # Live-mode tools run in parallel threads; read-modify-write must be serialized.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^\w.-]+", "_", user_id or "").strip("._") or "anonymous"
        return Path(self.root_dir) / safe / "User.md"

    def default_text(self, user_id: str) -> str:
        return f"# User.md - {user_id}\n\n## Profile facts\n"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        with self._lock:
            if not path.exists():
                return self.default_text(user_id)
            return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: a reader never sees a truncated, half-written file.
            # Fixed newline so byte counts (Memory growth) are identical on every OS.
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(content, encoding="utf-8", newline="\n")
            os.replace(tmp, path)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        with self._lock:
            current = self.read_text(user_id)
            if not search_text or search_text not in current:
                return False
            updated = current.replace(search_text, replacement, 1)
            if updated == current:
                return False
            self.write_text(user_id, updated)
            return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z_]+): (.+)$", line.strip())
            if match:
                result[match.group(1)] = match.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one `- key: value` line. Returns True if the file changed."""

        value = _one_line(value)
        if not value:
            return False
        pattern = re.compile(rf"^- {re.escape(key)}: .*$", re.MULTILINE)
        new_line = f"- {key}: {value}"
        with self._lock:
            text = self.read_text(user_id)
            if pattern.search(text):
                updated = pattern.sub(lambda _: new_line, text, count=1)
            else:
                updated = text.rstrip("\n") + "\n" + new_line + "\n"
            if updated == text:
                return False
            self.write_text(user_id, updated)
            return True

    def apply_updates(
        self, user_id: str, updates: dict[str, str], overwrite_scalars: bool = True
    ) -> dict[str, str]:
        """Merge facts into User.md and return the facts that actually changed.

        `overwrite_scalars=False` is used for LLM tool writes: they may fill an empty
        field but never replace an existing fact (corrections go through the rule-based
        conflict handler in `extract_profile_updates`).
        """

        changed: dict[str, str] = {}
        with self._lock:
            current = self.facts(user_id)
            for key in PROFILE_FIELDS:
                if key not in updates:
                    continue
                value = _one_line(updates[key])
                if key in LIST_FIELDS:
                    merged = _split_list(current.get(key, ""))
                    for item in _split_list(value):
                        if len(item) <= MAX_LIST_ITEM_CHARS and item not in merged:
                            merged.append(item)
                    value = ", ".join(merged[-MAX_LIST_ITEMS:])
                elif len(value) > MAX_SCALAR_CHARS or (not overwrite_scalars and current.get(key)):
                    continue
                if value and self.upsert_fact(user_id, key, value):
                    changed[key] = value
        return changed


def _split_list(value: str) -> list[str]:
    return [item.strip(" .") for item in value.split(",") if item.strip(" .")]


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


# ---------------------------------------------------------------------------
# Fact extraction with confidence (bonus: confidence threshold + conflict handling)
# ---------------------------------------------------------------------------

LOCATIONS = (
    "Hồ Chí Minh",
    "Sài Gòn",
    "Hà Nội",
    "Đà Nẵng",
    "Huế",
    "Hải Phòng",
    "Cần Thơ",
    "Nha Trang",
    "Đà Lạt",
    "Vũng Tàu",
    "Quy Nhơn",
    "Hội An",
)
_LOCATION_RE = re.compile("|".join(re.escape(loc) for loc in LOCATIONS))
_LOCATION_TRIGGER = re.compile(r"(?:\bở|\btại|\bsang|nơi ở(?: hiện tại)?(?: của mình)? là)\s+$", re.IGNORECASE)

_PROFESSION_RE = re.compile(
    r"\b(?:backend|frontend|fullstack|full-stack|MLOps|DevOps|data|ML|AI|software|platform)\s+engineer\b"
    r"|\bproduct manager\b|\bdata scientist\b|\bdata analyst\b",
    re.IGNORECASE,
)
_PROFESSION_TRIGGER = re.compile(r"(?:\blàm|\blà|\bsang|\bnghề)\s+(?:một\s+)?$", re.IGNORECASE)

# A mention right after these words is a fact the user is *retracting*.
_NEGATION = re.compile(r"không còn|không phải|chứ không|đừng|chưa", re.IGNORECASE)

# Sentences that are hypothetical / jokes: facts inside them get low confidence.
_HEDGE = re.compile(r"\bnếu\b|\bđùa\b|\bgiả sử\b|\bhay là\b|\bví dụ cũ\b", re.IGNORECASE)

# Case-sensitive on purpose: "startup AI." must not look like the particle "ai".
_QUESTION_END = re.compile(r"(?:\bgì|\bkhông|\bđâu|\bnào|\bai|\bsao)\s*[.!]*\s*$")

_NAME_RE = re.compile(
    r"(?:\b(?:mình|tôi|em)\s+tên\s+là|\btên\s+(?:của\s+)?(?:mình|tôi|em)\s+là|(?:^|[:;]\s*)tên)\s+(?P<v>.+)",
    re.IGNORECASE,
)
_DRINK_RE = re.compile(r"đồ uống yêu thích(?:\s+của\s+(?:mình|tôi))?\s+là\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)
_FOOD_RE = re.compile(r"món ăn yêu thích(?:\s+của\s+(?:mình|tôi))?\s+là\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)
_PET_RE = re.compile(r"\bnuôi\s+(?:một\s+)?(?:bé\s+|con\s+|chú\s+)?(?P<v>[^.,!?;]+)", re.IGNORECASE)

_INTEREST_TRIGGER = re.compile(r"(?<!không )\b(?:thích|quan tâm|đang học|học thêm|ôn lại)\b", re.IGNORECASE)
INTEREST_KEYWORDS = ("Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "LangChain", "LangGraph")

_STYLE_TRIGGER = re.compile(r"trả lời|giải thích|trình bày|style", re.IGNORECASE)
STYLE_KEYWORDS = (
    "ngắn gọn",
    "3 bullet",
    "bullet ngắn",
    "rõ ý",
    "có cấu trúc",
    "ví dụ thực tế",
    "ví dụ thực chiến",
    "trade-off",
)

CONFIDENCE_EXPLICIT = 0.9
CONFIDENCE_HEDGED = 0.3


@dataclass
class FactCandidate:
    key: str
    value: str
    confidence: float
    sentence: str


def split_sentences(message: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", message or "") if part.strip()]


def is_question_sentence(sentence: str) -> bool:
    return "?" in sentence or bool(_QUESTION_END.search(sentence))


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+(?:nhé|nha|nữa|đó)$", "", value.strip())
    return value.strip(" .,:;")


def _leading_capitalized_words(text: str, limit: int = 3) -> str:
    words: list[str] = []
    for raw in text.split():
        word = raw.strip(".,!?;:\"'()")
        if not word or not word[0].isupper():
            break
        words.append(word)
        if len(words) >= limit or raw[-1] in ".,!?;:":
            break
    return " ".join(words)


def _positive_mentions(sentence: str, pattern: re.Pattern, trigger: re.Pattern) -> list[str]:
    """Mentions introduced by a residence/role trigger and not negated right before it."""

    found: list[str] = []
    last_end = 0
    for match in pattern.finditer(sentence):
        left = sentence[last_end : match.start()]
        last_end = match.end()
        if not trigger.search(left):
            continue
        if _NEGATION.search(left[-20:]):
            continue
        found.append(match.group(0))
    return found


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Return every fact candidate with a confidence score, in message order."""

    candidates: list[FactCandidate] = []
    for sentence in split_sentences(message):
        if is_question_sentence(sentence):
            continue
        hedged = bool(_HEDGE.search(sentence))
        confidence = CONFIDENCE_HEDGED if hedged else CONFIDENCE_EXPLICIT

        def add(key: str, value: str, conf: float = confidence) -> None:
            value = _clean_value(value)
            if value and value.lower() != "gì":
                candidates.append(FactCandidate(key, value, conf, sentence))

        name_match = _NAME_RE.search(sentence)
        if name_match:
            add("name", _leading_capitalized_words(name_match.group("v")))

        for location in _positive_mentions(sentence, _LOCATION_RE, _LOCATION_TRIGGER):
            add("location", location)

        for role in _positive_mentions(sentence, _PROFESSION_RE, _PROFESSION_TRIGGER):
            normalized = "MLOps engineer" if role.lower() == "mlops engineer" else role
            add("profession", normalized)

        for key, regex in (("favorite_drink", _DRINK_RE), ("favorite_food", _FOOD_RE), ("pet", _PET_RE)):
            match = regex.search(sentence)
            if match:
                add(key, match.group("v"))

        if _INTEREST_TRIGGER.search(sentence):
            interests = [kw for kw in INTEREST_KEYWORDS if kw.lower() in sentence.lower()]
            if interests:
                add("interests", ", ".join(interests))

        if _STYLE_TRIGGER.search(sentence):
            styles = [kw for kw in STYLE_KEYWORDS if kw.lower() in sentence.lower()]
            if styles:
                # Conditional instructions ("Nếu bạn giải thích, hãy ...") are still
                # preferences, so only jokes lower the confidence of style facts.
                style_conf = CONFIDENCE_HEDGED if re.search(r"\bđùa\b", sentence, re.I) else CONFIDENCE_EXPLICIT
                add("response_style", ", ".join(styles), style_conf)
    return candidates


def extract_profile_updates(message: str, min_confidence: float = 0.6) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    - Question sentences are skipped (asking about a fact is not stating it).
    - Mentions that are negated ("không còn ở Đà Nẵng") are never candidates.
    - Facts inside hypothetical/joke sentences get low confidence and are dropped
      by `min_confidence` (confidence threshold).
    - For scalar fields the last confident mention wins (conflict handling).
    """

    updates: dict[str, str] = {}
    for candidate in extract_profile_candidates(message):
        if candidate.confidence < min_confidence:
            continue
        if candidate.key in LIST_FIELDS and candidate.key in updates:
            merged = _split_list(updates[candidate.key])
            merged += [item for item in _split_list(candidate.value) if item not in merged]
            updates[candidate.key] = ", ".join(merged)
        else:
            updates[candidate.key] = candidate.value
    return updates


# ---------------------------------------------------------------------------
# Recall answering shared by both agents (offline mode)
# ---------------------------------------------------------------------------

_FIELD_QUESTION_PATTERNS = (
    ("name", re.compile(r"\btên\b|là ai", re.IGNORECASE)),
    ("profession", re.compile(r"nghề", re.IGNORECASE)),
    ("location", re.compile(r"ở đâu|nơi ở|còn ở|đang ở", re.IGNORECASE)),
    ("favorite_drink", re.compile(r"đồ uống", re.IGNORECASE)),
    ("favorite_food", re.compile(r"món ăn", re.IGNORECASE)),
    ("pet", re.compile(r"\bnuôi\b", re.IGNORECASE)),
    ("response_style", re.compile(r"style|kiểu trả lời|trả lời", re.IGNORECASE)),
    ("interests", re.compile(r"quan tâm|sở thích", re.IGNORECASE)),
)
_SUMMARY_REQUEST = re.compile(r"tóm tắt|là ai|mô tả", re.IGNORECASE)
# Imperative recall requests: "Nhắc lại giúp mình: ...", "..., nhắc lại giúp mình tên ...",
# "Tóm tắt ngắn về mình: ...". "mình nhắc lại ngắn về bốn tin" is NOT a request.
_RECALL_REQUEST = re.compile(
    r"^(?:nhắc lại|tóm tắt|mô tả)\b|(?:nhắc lại|tóm tắt|mô tả)\s+(?:giúp|cho)\s+(?:mình|tôi)",
    re.IGNORECASE,
)


def is_recall_request(message: str) -> bool:
    sentences = split_sentences(message)
    if any(is_question_sentence(s) for s in sentences):
        return True
    return any(_RECALL_REQUEST.search(s) for s in sentences)


def requested_fields(question: str) -> list[str]:
    fields = [key for key, pattern in _FIELD_QUESTION_PATTERNS if pattern.search(question)]
    if _SUMMARY_REQUEST.search(question):
        for key in ("name", "profession", "interests"):
            if key not in fields:
                fields.append(key)
    return [key for key in PROFILE_FIELDS if key in fields]


def answer_from_facts(question: str, facts: dict[str, str]) -> str | None:
    """Deterministic recall answer, or None when the message asks for no profile field."""

    fields = requested_fields(question)
    if not fields:
        return None
    lines = []
    for key in fields:
        value = facts.get(key)
        lines.append(f"- {FIELD_LABELS[key]}: {value if value else 'chưa có thông tin'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: first sentence of each message, truncated, last `max_items` lines."""

    lines = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        first = split_sentences(content)[0] if split_sentences(content) else content
        if len(first) > 100:
            first = first[:97].rstrip() + "..."
        lines.append(f"- {message.get('role', 'user')}: {first}")
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Keeps recent messages verbatim and folds older ones into a bounded summary."""

    threshold_tokens: int
    keep_messages: int
    max_summary_items: int = 6
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def context_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        return estimate_tokens(str(thread["summary"])) + sum(estimate_tokens(m["content"]) for m in messages)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        if self.context_tokens(thread_id) > self.threshold_tokens and len(messages) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        old_lines = [line for line in str(thread["summary"]).splitlines() if line.strip()]
        new_lines = summarize_messages(older, max_items=self.max_summary_items).splitlines()
        thread["summary"] = "\n".join((old_lines + new_lines)[-self.max_summary_items :])
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {
            "messages": list(thread["messages"]),  # type: ignore[arg-type]
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
