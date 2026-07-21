"""Mock Adapter for offline testing, golden validation, and demonstration."""

import random
import time
from typing import Any
from fingerprint.types import AdapterResult


class MockAdapter:
    """Mock adapter producing realistic synthetic outputs based on model profile seed."""

    PROFILES = {
        "gpt-5": {
            "num100-random": ["7", "42", "73", "37", "88", "13", "55"],
            "num10-random": ["7", "3", "7", "5", "8", "7", "2"],
            "num-favorite": ["7", "7", "42", "7"],
            "letter-random": ["A", "B", "Z", "X", "M"],
            "word-random": ["apple", "ocean", "serendipity", "whisper", "galaxy"],
            "color-random": ["blue", "red", "green", "purple", "yellow"],
            "color-favorite": ["blue", "blue", "teal", "purple"],
            "animal-random": ["elephant", "lion", "dolphin", "tiger", "falcon"],
            "city-random": ["Paris", "Tokyo", "London", "New York", "Berlin"],
            "coin-flip": ["heads", "heads", "tails", "heads", "tails"],
        },
        "claude-3-7-sonnet": {
            "num100-random": ["42", "17", "84", "23", "99", "42", "7"],
            "num10-random": ["4", "7", "3", "9", "4", "6", "8"],
            "num-favorite": ["42", "42", "17", "42"],
            "letter-random": ["E", "T", "A", "O", "I"],
            "word-random": ["luminous", "cascade", "prism", "ephemeral", "solitude"],
            "color-random": ["green", "blue", "violet", "amber", "crimson"],
            "color-favorite": ["green", "blue", "emerald", "indigo"],
            "animal-random": ["owl", "fox", "raven", "octopus", "wolf"],
            "city-random": ["Kyoto", "Vienna", "Prague", "Sydney", "Cairo"],
            "coin-flip": ["tails", "heads", "tails", "tails", "heads"],
        },
        "gemini-2.5-pro": {
            "num100-random": ["37", "73", "42", "19", "88", "7"],
            "num10-random": ["3", "7", "8", "5", "9", "2"],
            "num-favorite": ["37", "37", "73", "37"],
            "letter-random": ["G", "M", "N", "S", "R"],
            "word-random": ["harmony", "quantum", "solace", "nexus", "horizon"],
            "color-random": ["amber", "cyan", "sapphire", "indigo", "violet"],
            "color-favorite": ["sapphire", "amber", "cyan", "indigo"],
            "animal-random": ["panther", "eagle", "cheetah", "orca", "falcon"],
            "city-random": ["Tokyo", "Zurich", "Singapore", "Toronto", "Seoul"],
            "coin-flip": ["heads", "tails", "heads", "heads", "tails"],
        },
        "gpt-4o": {
            "num100-random": ["7", "42", "73", "37", "88", "13", "55"],
            "num10-random": ["7", "3", "7", "5", "8", "7", "2"],
            "num-favorite": ["7", "7", "42", "7"],
            "letter-random": ["A", "B", "Z", "X", "M"],
            "word-random": ["apple", "ocean", "serendipity", "whisper", "galaxy"],
            "color-random": ["blue", "red", "green", "purple", "yellow"],
            "color-favorite": ["blue", "blue", "teal", "purple"],
            "animal-random": ["elephant", "lion", "dolphin", "tiger", "falcon"],
            "city-random": ["Paris", "Tokyo", "London", "New York", "Berlin"],
            "coin-flip": ["heads", "heads", "tails", "heads", "tails"],
        },
        "claude-3-5-sonnet": {
            "num100-random": ["42", "17", "84", "23", "99", "42", "7"],
            "num10-random": ["4", "7", "3", "9", "4", "6", "8"],
            "num-favorite": ["42", "42", "17", "42"],
            "letter-random": ["E", "T", "A", "O", "I"],
            "word-random": ["luminous", "cascade", "prism", "ephemeral", "solitude"],
            "color-random": ["green", "blue", "violet", "amber", "crimson"],
            "color-favorite": ["green", "blue", "emerald", "indigo"],
            "animal-random": ["owl", "fox", "raven", "octopus", "wolf"],
            "city-random": ["Kyoto", "Vienna", "Prague", "Sydney", "Cairo"],
            "coin-flip": ["tails", "heads", "tails", "tails", "heads"],
        },
        "devin-auto-mock": {
            # Router mixture of gpt-5 and claude-3-7-sonnet
            "num100-random": ["7", "42", "73", "17", "84", "37"],
            "num10-random": ["7", "4", "3", "7", "9", "5"],
            "num-favorite": ["7", "42", "7", "42"],
            "color-random": ["blue", "green", "red", "violet", "purple"],
            "coin-flip": ["heads", "tails", "heads", "tails", "heads"],
        },
    }

    def __init__(
        self,
        name: str = "mock",
        target_profile: str = "gpt-5",
        environment: str = "mock-cli",
        failure_rate: float = 0.0,
        seed: int = 42,
    ):
        self.name = name
        self.target_profile = target_profile
        self.environment = environment
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)

    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult:
        start_time = time.perf_counter()

        if self.rng.random() < self.failure_rate:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AdapterResult(
                raw_text="I cannot complete this request as specified.",
                extracted_token=None,
                valid=False,
                invalid_reason="Refusal / mock failure",
                latency_ms=round(elapsed_ms, 2),
                exit_code=0,
                stdout="I cannot complete this request as specified.",
                stderr="",
                meta={"adapter": self.name, "environment": self.environment},
            )

        profile = self.PROFILES.get(self.target_profile, self.PROFILES["gpt-4o"])

        # Detect probe task from prompt text
        chosen_word = "7"
        prompt_lower = prompt.lower()
        if "between 1 and 100" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("num100-random", ["7", "42"]))
        elif "between 1 and 10" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("num10-random", ["7", "3"]))
        elif "favorite number" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("num-favorite", ["7"]))
        elif "letter of the alphabet" in prompt_lower or "letter" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("letter-random", ["A"]))
        elif "color" in prompt_lower:
            if "favorite" in prompt_lower:
                chosen_word = self.rng.choice(profile.get("color-favorite", ["blue"]))
            else:
                chosen_word = self.rng.choice(profile.get("color-random", ["blue", "red"]))
        elif "animal" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("animal-random", ["elephant"]))
        elif "city" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("city-random", ["Paris"]))
        elif "coin" in prompt_lower:
            chosen_word = self.rng.choice(profile.get("coin-flip", ["heads"]))
        else:
            words = profile.get("word-random", ["apple"])
            chosen_word = self.rng.choice(words)

        elapsed_ms = (time.perf_counter() - start_time + 0.005) * 1000.0

        return AdapterResult(
            raw_text=chosen_word,
            extracted_token=chosen_word,
            valid=True,
            invalid_reason=None,
            latency_ms=round(elapsed_ms, 2),
            exit_code=0,
            stdout=chosen_word,
            stderr="",
            meta={
                "adapter": self.name,
                "environment": self.environment,
                "profile": self.target_profile,
            },
        )
