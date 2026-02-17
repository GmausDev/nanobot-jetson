from enum import Enum, auto
from typing import Optional


class State(Enum):
    IDLE = auto()
    WAKE_DETECTED = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


# Valid transitions: (from_state, event) -> to_state
TRANSITIONS = {
    (State.IDLE, "wake_word"): State.WAKE_DETECTED,
    (State.WAKE_DETECTED, "ready"): State.LISTENING,
    (State.LISTENING, "silence"): State.THINKING,
    (State.LISTENING, "timeout"): State.IDLE,
    (State.THINKING, "response_ready"): State.SPEAKING,
    (State.SPEAKING, "done"): State.IDLE,
}

# Any state can go to ERROR, and ERROR always returns to IDLE
for s in State:
    if s != State.ERROR:
        TRANSITIONS[(s, "error")] = State.ERROR
TRANSITIONS[(State.ERROR, "recover")] = State.IDLE


def get_next_state(current: State, event: str) -> Optional[State]:
    """Get the next state given current state and event. Returns None if invalid."""
    return TRANSITIONS.get((current, event))


def is_valid_transition(current: State, event: str) -> bool:
    return (current, event) in TRANSITIONS
