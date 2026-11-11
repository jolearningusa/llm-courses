"""Byte/chunk accounting fixtures; no BLT/H-Net model or GPU is executed.

The prefix-count entropy estimator is an original, deliberately weak teaching
rule, not BLT's trained entropy model. Segmentation counts are not FLOPs or speed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform

from bpe import ByteBPE

SAMPLES = {'repeat': 'banana banana! banana?\n',
           'multilingual': '缓存 cache 🙂\n',
           'code': 'def f(x): return x + 1\n'}


def prefix_entropy(data, threshold=7.99, max_patch=8):
    """Start BEFORE byte i using counts from data[:i] only; reset per document.

    Laplace-one distribution over 256 byte values; entropy in bits. EOS closes
    the final patch but is not read to select an earlier boundary.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 8 or type(max_patch) is not int or max_patch < 1:
        raise ValueError('finite threshold in [0,8] and positive integer patch cap required')
    counts, boundaries, entropies, age = Counter(), [], [], 0
    for i, byte in enumerate(data):
        total = i + 256
        entropy = -sum(((counts[b] + 1) / total) * math.log2((counts[b] + 1) / total)
                       for b in range(256))
        start = i == 0 or entropy > threshold or age >= max_patch
        boundaries.append(start)
        entropies.append(entropy)
        age = 1 if start else age + 1
        counts[byte] += 1
    return boundaries, entropies


def partition(data, boundaries):
    if len(data) != len(boundaries) or (data and not boundaries[0]):
        raise ValueError('one boundary flag per byte, first byte starts a patch')
    starts = [i for i, flag in enumerate(boundaries) if flag]
    return [data[a:b] for a, b in zip(starts, starts[1:] + [len(data)])]


def dechunk_scalar(length, starts, values, probabilities):
    """Literal EMA at selected positions, then causal forward fill.

    Isolates H-Net paper equations 5/8 with supplied scalar values/probabilities.
    No learned router, straight-through gradient, scan kernel or residual branch.
    """
    if (length < 1 or not starts or starts[0] != 0 or starts != sorted(set(starts))
            or starts[-1] >= length or len(starts) != len(values)
            or len(starts) != len(probabilities)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)):
        raise ValueError('ordered starts in range and aligned probabilities required')
    out, state, selected = [], 0.0, 0
    for i in range(length):
        if selected < len(starts) and i == starts[selected]:
            p = probabilities[selected]
            state = p * values[selected] + (1 - p) * state
            selected += 1
        out.append(state)
    return out


def position_ledger(n, m, hierarchical=False, entropy=False):
    """Declared architecture positions, not executed neural operations.

    Distinct coefficients/widths/layers make adding these columns meaningless
    as a FLOP total. Tokenization, packing, cross-attention, training, memory and
    kernel costs still require a qualified model implementation and profiler.
    """
    return dict(global_positions=m, local_encoder_positions=n if hierarchical else 0,
                local_decoder_positions=n if hierarchical else 0,
                router_positions=n if entropy else 0,
                byte_output_positions=n if hierarchical else 0,
                token_or_byte_head_positions=m if not hierarchical else 0,
                latency='not_measured', flops='not_measured', memory='not_measured',
                model_execution='not_run')


def report():
    root = Path(__file__).resolve().parent
    corpora = {p.name: p.read_bytes() for p in sorted((root / 'corpus').glob('*.txt'))}
    training = '\n'.join(v.decode('utf-8') for v in corpora.values())
    bpe = ByteBPE().train(training, 64)
    rows = []
    for name, text in SAMPLES.items():
        raw = text.encode('utf-8')
        boundaries, entropies = prefix_entropy(raw)
        ids = bpe.encode(text)
        pieces = [bpe.vocab[i] for i in ids]
        partitions = {'bytes': [bytes([b]) for b in raw], 'bpe64': pieces,
                      'fixed8': [raw[i:i+8] for i in range(0, len(raw), 8)],
                      'prefix_entropy_toy': partition(raw, boundaries)}
        for method, chunks in partitions.items():
            # Reconstruct bytes first: an individual piece may split a UTF-8 character.
            rebuilt = b''.join(chunks)
            if rebuilt != raw:
                raise AssertionError('byte conservation failed')
            rows.append(dict(sample=name, method=method, input_hex=raw.hex(),
                             input_sha256=hashlib.sha256(raw).hexdigest(),
                             bytes=len(raw), characters=len(text),
                             piece_hex=[p.hex() for p in chunks], pieces=len(chunks),
                             round_trip=rebuilt.decode('utf-8') == text,
                             positions=position_ledger(len(raw), len(chunks),
                                                       method in ('fixed8', 'prefix_entropy_toy'),
                                                       method == 'prefix_entropy_toy'),
                             entropy_details=dict(boundaries=boundaries, entropy_bits=entropies,
                                                  distribution_evaluations=len(raw),
                                                  probability_bins=len(raw)*256)
                             if method == 'prefix_entropy_toy' else None))
    source_names = ('chunk_accounting.py', 'bpe.py')
    return dict(schema=1, python=platform.python_version(),
                source_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source_names},
                corpus_sha256={n: hashlib.sha256(b).hexdigest() for n,b in corpora.items()},
                config=dict(bpe_merges=64, entropy_threshold_bits=7.99, max_patch_bytes=8,
                            entropy_prior='Laplace one, 256 bytes, document-local prefix counts'),
                actual_merges=len(bpe.merges), merge_pairs=bpe.merges, rows=rows,
                scalar_dechunk=dict(starts=[0,3], values=[10,30], probabilities=[1,.5],
                                    output=dechunk_scalar(5,[0,3],[10,30],[1,.5])),
                limits='Only tokenization, boundaries, byte round trips and scalar algebra executed. '
                       'Architecture position ledger is declared, not profiled. No trained '
                       'BLT/H-Net, model quality, BPB, FLOPs, memory or device timing measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(report(), stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
