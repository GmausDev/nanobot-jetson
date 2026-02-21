from __future__ import annotations

import re
import logging

logger = logging.getLogger("nanobot.llm.streamer")


class SentenceStreamer:
    """
    Accumulates streamed tokens and yields complete sentences.
    Useful for sending text to TTS as soon as a sentence is done,
    rather than waiting for the full LLM response.
    """

    # Sentence-ending patterns
    SENTENCE_END = re.compile(r'[.!?]+[\s"\')\]]*$')

    def __init__(self):
        self.buffer = ""

    def reset(self):
        self.buffer = ""

    def feed(self, token: str) -> list[str]:
        """
        Feed a token. Returns a list of complete sentences (usually 0 or 1).
        """
        self.buffer += token
        sentences = []

        # Try to extract complete sentences
        while True:
            match = self._find_sentence_break()
            if match is None:
                break
            sentence = self.buffer[:match].strip()
            self.buffer = self.buffer[match:].lstrip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def flush(self) -> str:
        """Flush remaining buffer as a final sentence."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining

    def _find_sentence_break(self) -> int | None:
        """
        Find the position of a sentence break in the buffer.
        Returns the index after the sentence-ending punctuation, or None.
        """
        # Look for sentence-ending punctuation followed by a space or end
        for i, char in enumerate(self.buffer):
            if char in '.!?' and i > 0:
                # Check if followed by a space or end of buffer (with some content after)
                next_pos = i + 1
                # Skip any closing quotes/parens
                while next_pos < len(self.buffer) and self.buffer[next_pos] in '"\')\] ':
                    next_pos += 1

                if next_pos < len(self.buffer) and self.buffer[next_pos].isupper():
                    return next_pos
                # Don't split on abbreviations like "Dr." or "U.S." or "3.5"
                if i > 0 and self.buffer[i-1].isdigit():
                    continue
                if i > 1 and self.buffer[i-1].isupper() and self.buffer[i-2] == '.':
                    continue

        return None
