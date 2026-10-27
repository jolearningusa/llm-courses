"""threshold-compaction: fills the context-engine slot with nanoos.context.ThresholdCompaction (Lab 3)."""

from __future__ import annotations


def register(api):
    from nanoos.context import ThresholdCompaction, heuristic_summarizer
    api.provide("context-engine", ThresholdCompaction(api.config["budget_tokens"], heuristic_summarizer,
                                                      keep_last=api.config["keep_last"]))
