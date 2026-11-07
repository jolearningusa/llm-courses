"""Float64 gradient-check diagnostics; CPU/NumPy, not a model benchmark.

python gradient_diagnostics.py
python gradient_diagnostics.py --plot-dir figures  # optional matplotlib
"""
import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
from autograd import Tensor


def centered_difference(fn, x, h):
    if not math.isfinite(h) or h <= 0 or not math.isfinite(x):
        raise ValueError('finite x and positive finite h required')
    x, h = np.float64(x), np.float64(h)
    return float((fn(x + h) - fn(x - h)) / (2 * h))


def error_metrics(reference, candidate):
    if not all(math.isfinite(v) for v in (reference, candidate)):
        raise ValueError('finite gradients required')
    absolute = abs(reference - candidate)
    scale = max(abs(reference), abs(candidate))
    return {'absolute': absolute, 'raw_relative': absolute / scale if scale else 0.0}


def matches(reference, candidate, atol=1e-8, rtol=1e-6):
    """Fixture-specific mixed tolerance, not a universal gradient-check threshold."""
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('finite nonnegative tolerances required')
    return (all(math.isfinite(v) for v in (reference, candidate))
            and abs(candidate - reference) <= atol + rtol * abs(reference))


def cubic(x):
    return x * x * x


def autograd_cubic(x):
    t = Tensor(np.array([x], dtype=np.float64))
    (t * t * t).sum().backward()
    return float(t.grad[0])


def report():
    analytic = autograd_cubic(1.0)
    sweep = []
    for power in range(1, 21):
        h = 10.0 ** -power
        numeric = centered_difference(cubic, 1.0, h)
        sweep.append(dict(h=h, numeric=numeric, **error_metrics(analytic, numeric)))
    numeric = centered_difference(cubic, 1.0, 1e-5)
    controls = [{'name': name, 'candidate': value, 'accepted_on_fixture': matches(numeric, value)}
                for name, value in [('autograd', analytic), ('wrong_sign', -analytic),
                                    ('missing_contribution', 2.0)]]
    near_zero = []
    for true in (1.0, 1e-3, 1e-6, 1e-9, 0.0):
        candidate = true + 1e-12
        near_zero.append(dict(reference=true, candidate=candidate,
                             **error_metrics(true, candidate)))
    relu = []
    for x, h in [(-1e-4, 1e-3), (-1e-4, 1e-5), (0.0, 1e-5)]:
        t = Tensor(np.array([x], dtype=np.float64))
        t.relu().sum().backward()
        relu.append(dict(x=x, h=h, numeric=centered_difference(lambda z: max(0.0, z), x, h),
                         autograd=float(t.grad[0]), classical_derivative=None if x == 0 else float(x > 0),
                         touches_kink=bool(x-h <= 0 <= x+h)))
    return {'provenance': 'executed fixed CPU float64 examples; not a trained-model result',
            'python': platform.python_version(), 'numpy': np.__version__, 'dtype': 'float64',
            'smooth': {'function': 'x^3', 'x': 1.0, 'analytic': analytic, 'sweep': sweep},
            'contract': {'atol': 1e-8, 'rtol': 1e-6, 'h': 1e-5, 'scope': 'smooth cubic controls only'},
            'controls': controls, 'near_zero': near_zero, 'relu': relu}


def plot(data, directory):
    """Optional static scientific figures, derived from the same report data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'fnd1-gradient-diagnostics',
                         'font.size': 15, 'text.color': '#eef4ff', 'axes.labelcolor': '#eef4ff',
                         'xtick.color': '#b7c7df', 'ytick.color': '#b7c7df',
                         'axes.edgecolor': '#7185a4', 'axes.facecolor': '#101927',
                         'figure.facecolor': '#101927'})
    def style(ax):
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    rows = data['smooth']['sweep']
    fig, ax = plt.subplots(figsize=(11, 3.6), layout='constrained')
    ax.loglog([r['h'] for r in rows], [max(r['absolute'], 1e-18) for r in rows], 'o-', color='#8de0cf')
    ax.set_xlabel('finite-difference step h'); ax.set_ylabel('absolute gradient error')
    ax.set_title('Executed float64 example: f(x) = x³ at x = 1', color='#eef4ff'); style(ax)
    fig.savefig(directory/'smooth_step.svg', metadata={'Date': None}); plt.close(fig)
    rows = data['near_zero']; labels = ['1', '1e−3', '1e−6', '1e−9', '0']
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained')
    for ax, key, title, color in zip(axs, ['absolute', 'raw_relative'],
                                    ['Absolute error', 'Raw relative error'], ['#8de0cf', '#f1c979']):
        ax.semilogy(range(5), [r[key] for r in rows], 'o-', color=color)
        ax.set_xticks(range(5), labels); ax.set_xlabel('reference gradient')
        ax.set_title(title, color='#eef4ff'); style(ax)
        if key == 'absolute':
            ax.set_ylim(1e-13, 1e-11)
        else:
            ax.set_ylim(1e-13, 2)
    fig.savefig(directory/'near_zero.svg', metadata={'Date': None}); plt.close(fig)
    for name in ('smooth_step.svg', 'near_zero.svg'):
        path = directory / name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON file; refuses to overwrite')
    parser.add_argument('--plot-dir', type=Path, help='Optional matplotlib SVG exports')
    args = parser.parse_args()
    data = report()
    if args.plot_dir:
        plot(data, args.plot_dir)
    encoded = json.dumps(data, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
