# Lab L2: byte-level BPE

Used with deck 07. CPU only, standard library only.

`bpe.py` trains a byte-level BPE tokenizer: Sennrich et al.'s merge loop (deck 07) run on UTF-8 bytes, with
GPT-2's rule that merges never cross a letter/digit/punctuation boundary. `corpus/` holds the same short
passage about tokenizers written in English, German, Spanish and Chinese.

## Run

```bash
python bpe.py --merges 300          # trains on corpus/, prints characters per token for each language
python -m unittest test_bpe.py
```

## Tasks

1. Read `train()` and match each line to Algorithm 1 of Sennrich et al. What replaces the end-of-word symbol?
2. Print the first 30 merges. Which language's pieces appear first, and why?
3. The four files say the same thing. Count tokens for each at 100, 300 and 1,000 merges. Which language
   gains least from more merges, and what does that mean for its share of a 128K-token context window?
4. Remove `pre_split` (encode the whole text as one chunk). What kinds of tokens appear, and why does GPT-2
   forbid them?
5. Add a fifth language file of your own and retrain. How does adding it change the English token count?

## Checks

Encoding and decoding are lossless for every corpus file and for unseen text (emoji, rare scripts, tabs);
every ID is a byte or a learned merge, so there is no unknown token; merges at least halve the English byte
count; a frequent word becomes one token; and the same content costs more tokens in Chinese than in English.
