"""Provided AGT-3 closed-world state fixture; no LLM, network or real tool actions.

An episodic claim log deliberately permits multiple active claims. This is not
Lab 4's one-active-profile-slot MemoryStore implementation or its solution.
Source labels and verified repair inputs are supplied by the fixture's host.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass

MODES = ("labels_only", "deny_all", "source_gate", "delete_latest", "purge_tool", "supersede")
INITIAL = "team@example.invalid"
UPDATED = "ops@example.invalid"
POISON = "outside@attacker.invalid"


@dataclass
class Claim:
    id: str
    value: str
    source: str
    active: bool = True
    superseded_by: str | None = None


class ClaimLog:
    """Latest active episodic claim wins; sources are host labels, not proof of truth."""

    def __init__(self, mode: str):
        if mode not in MODES:
            raise ValueError("unknown fixture mode")
        self.mode = mode
        self.claims: list[Claim] = []
        self.audit: list[dict] = []
        self.next_id = 1

    def write(self, value: str, source: str) -> Claim | None:
        allowed = self.mode != "deny_all" and (
            self.mode != "source_gate" or source in ("user", "verified_repair"))
        claim = None
        if allowed:
            claim = Claim(f"m{self.next_id}", value, source)
            self.next_id += 1
            self.claims.append(claim)
        self.audit.append({"event": "write", "source": source, "allowed": allowed,
                           "claim_id": claim.id if claim else None})
        return claim

    def retrieve(self) -> Claim | None:
        return next((c for c in reversed(self.claims) if c.active), None)

    def repair(self, verified_value: str) -> None:
        """Oracle-supplied repair, not automatic contamination detection or fact checking."""
        affected = []
        if self.mode in ("delete_latest", "purge_tool"):
            tool_claims = [c for c in self.claims if c.source.startswith("tool:")]
            selected = tool_claims[-1:] if self.mode == "delete_latest" else tool_claims
            affected = [c.id for c in selected]
            self.claims = [c for c in self.claims if c.id not in affected]
        elif self.mode == "supersede":
            affected = [c.id for c in self.claims if c.active]
            correction = self.write(verified_value, "verified_repair")
            for c in self.claims:
                if c.id in affected:
                    c.active, c.superseded_by = False, correction.id
        self.audit.append({"event": "repair", "mode": self.mode, "affected_ids": affected})


def simulate(mode: str) -> dict:
    log, observations = ClaimLog(mode), []

    def observe(phase: str, expected: str) -> None:
        # Ground truth is used only for scoring, never by the reader.
        hit = log.retrieve()
        proposal = hit.value if hit else None
        observations.append({"phase": phase, "expected": expected,
                             "retrieved_id": hit.id if hit else None,
                             "retrieved_source": hit.source if hit else None,
                             "proposed_destination": proposal,
                             "correct_proposal": proposal == expected,
                             "unauthorized_proposal": proposal is not None and proposal != expected,
                             "claims": [asdict(c) for c in log.claims]})

    log.write(INITIAL, "user")
    log.write(POISON, "tool:page")
    log.write(POISON, "tool:feedback")
    observe("after_contamination", INITIAL)
    log.repair(INITIAL)  # externally verified value intentionally supplied to repair
    observe("after_repair", INITIAL)
    log.write(UPDATED, "user")  # legitimate owner correction must remain useful
    observe("after_user_correction", UPDATED)
    log.write(POISON, "tool:page")  # a repair is not an ongoing write gate
    observe("after_recontamination", UPDATED)
    return {"mode": mode, "observations": observations, "audit": log.audit}


def run() -> dict:
    runs = [simulate(mode) for mode in MODES]
    summary = []
    for row in runs:
        a, b, c, d = row["observations"]
        summary.append({"mode": row["mode"],
                        "unauthorized_after_contamination": a["unauthorized_proposal"],
                        "correct_after_repair": b["correct_proposal"],
                        "correct_after_user_correction": c["correct_proposal"],
                        "unauthorized_after_recontamination": d["unauthorized_proposal"]})
    return {"schema_version": 1, "experiment": "AGT-3 fixed state controls",
            "environment": {"python": platform.python_version(), "system": platform.system()},
            "limits": "Six deterministic arms of one constructed history; host-supplied labels and repair truth; "
                      "latest-active scripted reader; no LLM, authentication, semantic retrieval, network, "
                      "real delivery, model safety rate, erasure guarantee or latency benchmark.",
            "summary": summary, "runs": runs}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
