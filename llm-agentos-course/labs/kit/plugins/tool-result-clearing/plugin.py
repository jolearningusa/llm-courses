"""tool-result-clearing: fills the context-engine slot with nanoos.context.ToolResultClearing (Lab 3)."""

from __future__ import annotations


def register(api):
    from nanoos.context import ToolResultClearing
    api.provide("context-engine", ToolResultClearing(api.config["keep_last_results"], api.config["min_chars"]))
