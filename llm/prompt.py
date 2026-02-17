import logging
from collections import deque

logger = logging.getLogger("nanobot.llm.prompt")


class PromptManager:
    def __init__(self, config: dict):
        self.system_prompt = config.get("system_prompt", "You are a helpful robot.")
        max_turns = config.get("conversation_history_turns", 4)
        self.history: deque[dict] = deque(maxlen=max_turns * 2)  # *2 for user+assistant pairs

    def build_messages(self, user_input: str) -> list[dict]:
        """Build the full message list for the LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for msg in self.history:
            messages.append(msg)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    def add_turn(self, user_input: str, assistant_response: str):
        """Add a completed conversation turn to history."""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_response})

    def clear_history(self):
        """Reset conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
        logger.info("System prompt updated")
