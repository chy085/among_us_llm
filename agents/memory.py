"""
Memory system for agents.
Implements rolling memory with LLM-based compression.
"""

from typing import List, Dict, Optional, Tuple
import json


class MemoryEntry:
    def __init__(self, timestep: int, observation: Dict, action: Dict, result: str):
        self.timestep = timestep
        self.observation = observation
        self.action = action
        self.result = result

    def to_text(self) -> str:
        obs = self.observation
        action_str = f"{self.action.get('type', 'WAIT')}"
        if self.action.get('destination'):
            action_str += f" to {self.action['destination']}"
        elif self.action.get('target'):
            action_str += f" on {self.action['target']}"
        elif self.action.get('task'):
            action_str += f": {self.action['task']}"
        elif self.action.get('speech'):
            action_str += f": \"{self.action['speech'][:80]}...\""
        elif self.action.get('vote'):
            action_str += f" for {self.action['vote']}"

        return (
            f"[T={self.timestep}] Location: {obs.get('my_location', '?')} | "
            f"With: {obs.get('co_located_players', [])} | "
            f"Action: {action_str} | Result: {self.result}"
        )


class AgentMemory:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.entries: List[MemoryEntry] = []
        self.compressed_summary: Optional[str] = None
        self._llm_caller = None  # injected after init

    def add(self, timestep: int, observation: Dict, action: Dict, result: str):
        if self.max_size == 0:
            return
        entry = MemoryEntry(timestep, observation, action, result)
        self.entries.append(entry)

        # Compress if over limit
        if len(self.entries) > self.max_size and self._llm_caller:
            self._compress()

    def _compress(self):
        """Compress oldest entries via LLM summarization (keep last 5 raw)."""
        to_compress = self.entries[:-5]
        recent = self.entries[-5:]

        old_text = "\n".join(e.to_text() for e in to_compress)
        if self.compressed_summary:
            old_text = f"Previous summary:\n{self.compressed_summary}\n\nNew events:\n{old_text}"

        summary = self._llm_caller(
            f"Summarize these Among Us game events concisely (2-4 sentences), "
            f"focusing on suspicious behavior, player locations, and key events:\n\n{old_text}"
        )
        self.compressed_summary = summary
        self.entries = recent

    def to_context_string(self) -> str:
        """Format memory for prompt injection."""
        if self.max_size == 0:
            return "Memory disabled."

        parts = []
        if self.compressed_summary:
            parts.append(f"[Summary of earlier events]:\n{self.compressed_summary}")

        if self.entries:
            parts.append("[Recent events]:")
            for e in self.entries[-5:]:
                parts.append(e.to_text())

        if not parts:
            return "No memory yet."

        return "\n".join(parts)
