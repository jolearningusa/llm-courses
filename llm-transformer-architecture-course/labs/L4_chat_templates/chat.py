"""Lab L4: from a conversation to one token stream, and the masks used to train on it.

Pure Python plus NumPy. Two chat templates are written out by hand from their
documented formats (deck 11):

- Llama 3 Instruct: <|begin_of_text|>, then for each message
  <|start_header_id|>role<|end_header_id|>\\n\\ncontent<|eot_id|>
- Qwen2.5 Instruct (ChatML style): <|im_start|>role\\ncontent<|im_end|>\\n,
  with a default system message when none is given.

Then the training side: packing several documents into one sequence, the
document mask that stops attention across them (Llama 3), and a loss mask
that trains only on assistant tokens.
"""
from __future__ import annotations

import numpy as np

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def render_llama3(messages: list[dict], add_generation_prompt: bool = True) -> str:
    out = "<|begin_of_text|>"
    for m in messages:
        out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    if add_generation_prompt:
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out


def render_qwen25(messages: list[dict], add_generation_prompt: bool = True) -> str:
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": QWEN_DEFAULT_SYSTEM}] + list(messages)
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


def segments(messages: list[dict], render) -> list[tuple[str, bool]]:
    """Split a rendered chat into (text, is_assistant_content) pieces, for a loss mask.

    Works for both templates: the assistant content (and its end-of-turn marker) is
    what the model must learn to produce; everything else is context.
    """
    pieces, done = [], ""
    for k, m in enumerate(messages):
        upto = render(messages[:k + 1], add_generation_prompt=False)
        assert upto.startswith(done), "template must render each prefix of the chat as a prefix"
        if m["role"] == "assistant":
            head = render(messages[:k], add_generation_prompt=True)   # ends with the open assistant header
            pieces += [(head[len(done):], False), (upto[len(head):], True)]
        else:
            pieces.append((upto[len(done):], False))
        done = upto
    return pieces


def pack(doc_lengths: list[int]) -> np.ndarray:
    """Document id of every position when documents are packed back to back."""
    return np.repeat(np.arange(len(doc_lengths)), doc_lengths)


def causal_mask(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n), dtype=bool))


def document_mask(doc_ids: np.ndarray) -> np.ndarray:
    """Causal mask that also forbids attention between different packed documents."""
    same = doc_ids[:, None] == doc_ids[None, :]
    return causal_mask(len(doc_ids)) & same


def positions_restart(doc_ids: np.ndarray) -> np.ndarray:
    """Position ids that restart at 0 in each packed document."""
    pos = np.arange(len(doc_ids))
    starts = np.r_[0, np.flatnonzero(np.diff(doc_ids)) + 1]
    first = starts[np.searchsorted(starts, pos, side="right") - 1]
    return pos - first
