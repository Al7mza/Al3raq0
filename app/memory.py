from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypedDict
import os

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: MessageRole
    content: str


@dataclass
class InMemoryChatStore:
    max_history_messages: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "12")))
    _store: Dict[int, List[ChatMessage]] = field(default_factory=dict)

    def get_history(self, chat_id: int) -> List[ChatMessage]:
        return self._store.get(chat_id, [])

    def add_user_message(self, chat_id: int, content: str) -> None:
        self._append(chat_id, {"role": "user", "content": content})

    def add_assistant_message(self, chat_id: int, content: str) -> None:
        self._append(chat_id, {"role": "assistant", "content": content})

    def reset(self, chat_id: int) -> None:
        self._store.pop(chat_id, None)

    def _append(self, chat_id: int, message: ChatMessage) -> None:
        history = self._store.setdefault(chat_id, [])
        history.append(message)
        # Trim to keep only the most recent N messages (user+assistant), system is injected elsewhere
        if len(history) > self.max_history_messages:
            # keep the last N
            self._store[chat_id] = history[-self.max_history_messages :]