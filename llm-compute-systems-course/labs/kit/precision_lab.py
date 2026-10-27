"""CPU precision teaching reference; packed format is NOT TorchAO/vendor compatible."""
import argparse
import json
import math
import platform
import struct
from dataclasses import dataclass


@dataclass
class Packed:
    payload: bytes
    scales: bytes
    length: int
    bits: int
    group_size: int

    @property
    def array_bytes(self):
        return len(self.payload) + len(self.scales)


def quantize(values, bits=4, group_size=32):
    if bits not in (4, 8) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError('bits must be 4/8 and group_size a positive integer')
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('nonempty finite values required')
    limit = (1 << (bits - 1)) - 1
    codes, scales = [], []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        scale = max(map(abs, group)) / limit or 1.0
        try:
            scale = struct.unpack('<f', struct.pack('<f', scale))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError('scale not representable as FP32') from exc
        if scale == 0 or not math.isfinite(scale):
            raise ValueError('scale not representable as positive FP32')
        scales.append(scale)
        codes.extend(max(-limit, min(limit, round(x / scale))) & ((1 << bits) - 1) for x in group)
    payload = bytes(codes) if bits == 8 else bytes(codes[i] | ((codes[i+1] if i+1 < len(codes) else 0) << 4) for i in range(0, len(codes), 2))
    return Packed(payload, struct.pack('<' + 'f' * len(scales), *scales), len(values), bits, group_size)


def dequantize(packed):
    scales = struct.unpack('<' + 'f' * (len(packed.scales) // 4), packed.scales)
    result = []
    for i in range(packed.length):
        raw = packed.payload[i] if packed.bits == 8 else (packed.payload[i // 2] >> (4 * (i % 2))) & 15
        signed = raw - (1 << packed.bits) if raw >= (1 << (packed.bits - 1)) else raw
        result.append(signed * scales[i // packed.group_size])
    return result


def autocast_comparison():
    import torch
    torch.manual_seed(17)
    torch.set_num_threads(1)
    x, target = torch.randn(32, 16), torch.randn(32, 8)
    base = torch.nn.Linear(16, 8)
    state = base.state_dict()
    records, gradients = {}, {}
    for name, enabled in [('fp32', False), ('bf16_autocast', True)]:
        model = torch.nn.Linear(16, 8)
        model.load_state_dict(state)
        opt = torch.optim.SGD(model.parameters(), lr=0.02)
        losses = []
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=enabled):
                output = model(x)
                loss = (output.float() - target).square().mean()
            loss.backward()
            gradients[name] = torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
            if not torch.isfinite(gradients[name]).all():
                raise RuntimeError('nonfinite gradient')
            opt.step()
            losses.append(loss.item())
        records[name] = {'losses': losses, 'output_dtype': str(output.dtype)}
    records['last_gradient_max_abs_difference'] = (gradients['fp32'] - gradients['bf16_autocast']).abs().max().item()
    records['torch'] = torch.__version__
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autocast', action='store_true')
    args = parser.parse_args()
    values = [math.sin(i) for i in range(1025)]
    rows = []
    for bits in (4, 8):
        packed = quantize(values, bits)
        rows.append({'bits': bits, 'elements': len(values), 'payload_bytes': len(packed.payload), 'scale_bytes': len(packed.scales), 'max_abs_error': max(abs(x-y) for x,y in zip(values, dequantize(packed)))})
    result = {'lab': 'L6', 'provenance': 'CPU synthetic numerical experiment, not accelerated inference', 'python': platform.python_version(), 'quantization': rows}
    if args.autocast:
        result['autocast'] = autocast_comparison()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
