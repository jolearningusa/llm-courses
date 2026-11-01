"""Checks for Lab L4. Run: python -m unittest test_chat.py"""
import unittest

import numpy as np

from chat import (document_mask, pack, positions_restart, render_llama3, render_qwen25, segments)

CHAT = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is GQA?"},
]

# Expected strings written from the published formats (deck 11): Meta's Llama 3 prompt-format page and the
# chat_template in Qwen/Qwen2.5-7B-Instruct's tokenizer_config.json.
LLAMA3_EXPECTED = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are concise.<|eot_id|>"
                   "<|start_header_id|>user<|end_header_id|>\n\nWhat is GQA?<|eot_id|>"
                   "<|start_header_id|>assistant<|end_header_id|>\n\n")
QWEN_EXPECTED = ("<|im_start|>system\nYou are concise.<|im_end|>\n"
                 "<|im_start|>user\nWhat is GQA?<|im_end|>\n"
                 "<|im_start|>assistant\n")


class Templates(unittest.TestCase):
    def test_llama3(self):
        self.assertEqual(render_llama3(CHAT), LLAMA3_EXPECTED)

    def test_qwen25(self):
        self.assertEqual(render_qwen25(CHAT), QWEN_EXPECTED)

    def test_qwen_default_system_prompt(self):
        out = render_qwen25(CHAT[1:])
        self.assertTrue(out.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud."))

    def test_loss_mask_covers_only_assistant_content(self):
        chat = CHAT + [{"role": "assistant", "content": "Shared K/V heads."},
                       {"role": "user", "content": "And MQA?"},
                       {"role": "assistant", "content": "One K/V head."}]
        for render, end in ((render_llama3, "<|eot_id|>"), (render_qwen25, "<|im_end|>\n")):
            with self.subTest(render.__name__):
                pieces = segments(chat, render)
                self.assertEqual("".join(p for p, _ in pieces), render(chat, add_generation_prompt=False))
                trained = [p for p, is_asst in pieces if is_asst]
                self.assertEqual(trained, ["Shared K/V heads." + end, "One K/V head." + end])


class Packing(unittest.TestCase):
    def test_document_mask_blocks_cross_document_attention(self):
        ids = pack([3, 2, 3])
        mask = document_mask(ids)
        causal = np.tril(np.ones((8, 8), dtype=bool))
        self.assertEqual(int(causal.sum() - mask.sum()), 21)      # deck 11, question 3
        self.assertFalse(mask[3, 2])                                # doc 1 cannot see doc 0
        self.assertTrue(mask[4, 3])                                 # but sees itself

    def test_positions_restart_per_document(self):
        np.testing.assert_array_equal(positions_restart(pack([3, 2, 3])), [0, 1, 2, 0, 1, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
