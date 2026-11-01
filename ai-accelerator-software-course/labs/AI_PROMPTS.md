# Bounded AI-assisted lab tasks

Keep the contract and independent evaluator fixed. Record assistant/model identity if known, prompt, generated diff, human edits, tested source hash and complete outcomes. Do not upload confidential SDK code or private device logs to an external service without authorization.

## Candidate implementation prompt

“Read ABI.md and the selected function. Propose one small implementation change improving clarity or bounded resource behavior. Preserve all public contracts, including pinned ownership, timeout behavior and reset generations. Explain the invariant and an adverse case. Return a diff; do not edit tests, tolerances, report scope or release policy.”

## Independent test-design prompt

“Given the ABI contract, propose inputs/event orderings that distinguish a correct implementation from early release, stale completion or incorrect cancellation. Include a valid case that must pass. Explain expected results without relying on the candidate's own helper functions. Do not weaken the contract to match a result.”

## Triage prompt

“Given the frozen contract and a failing trace, reduce the sequence while preserving the failure. Classify an implementation bug, unsupported behavior, incorrect test expectation or environment failure. State the evidence needed to distinguish them. Do not label simulated ticks as hardware time.”

The lab does not call a hosted LLM or execute arbitrary generated files automatically. Students review a patch, run it in their local teaching environment and keep the evaluator separate. A proposed change that permits free while queued/inflight should be rejected by the independent lifetime tests and bounded counterexample. A test that rejects every implementation is also inadequate.
