"""Wrap any model so that a share of its calls fail, to measure how a loop copes (Lab 1).

    model = Flaky(OpenAICompatModel(), fail_rate=0.2, seed=1)

A failed call raises ModelError("injected failure") before the wrapped model is called, so it costs no tokens.
run_suite.py exposes this as --fail-rate.
"""

from __future__ import annotations

import random
from typing import Any

from models.scripted import ModelError
from nanoos.types import Model, ModelResponse


class Flaky:
    def __init__(self, model: Model, fail_rate: float, seed: int | None = None):
        self.model = model
        self.fail_rate = fail_rate
        self.name = f"flaky({fail_rate}):{model.name}"
        self.failures = 0
        self._rng = random.Random(seed)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        if self._rng.random() < self.fail_rate:
            self.failures += 1
            raise ModelError("injected failure")
        return self.model.complete(messages, tools)
