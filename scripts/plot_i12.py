"""Render the I12 sampling, noise, image and measured-calibration figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from run_i12 import AMP_METHODS, METHODS, image_target, read_json
from tmbench.solvers.prvamp.utils import align_phase


LABELS = dict(prgamp='prGAMP', prsamp='prSAMP', prvbem='prVBEM', prvamp='prVAMP',
              gs50='GS-50', gs100='GS-100', wf50='WF-50', wf500='WF-500',
              phaselift='PhaseLift', phasemax='PhaseMax')


def draw_sweep(ax, groups, settings, xlabel, title):
    floor = 1e-10
    for index, method in enumerate(METHODS):
        rows = [groups.get((key, method, 'oracle' if method in AMP_METHODS else 'none')) for _, key in settings]
        values = [row.get('phase_nmse', {}) if row else {} for row in rows]
        coordinates = [position for position, _ in settings]
        bands = [np.maximum([item.get(stat, np.nan) for item in values], floor)
                 for stat in ('q25', 'median', 'q75')]
        color = plt.get_cmap('tab10')(index)
        ax.plot(coordinates, bands[1], color=color, marker='o', markersize=3, label=LABELS[method])
        ax.fill_between(coordinates, bands[0], bands[2], color=color, alpha=.08)
    ax.set(xlabel=xlabel, ylabel='Phase-aligned coefficient NMSE', title=title, yscale='log')
    ax.grid(alpha=.22)


def save(pdf, figure, note):
    figure.text(.5, .027, note, ha='center', size=8)
    pdf.savefig(figure)
    plt.close(figure)


def plot_images(pdf, groups, output, cfg, regime):
    target = image_target(cfg['image_modes'])
    shape = (int(np.sqrt(target.size)),) * 2
    extent = np.max(target)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), layout='constrained')
    fig.suptitle(f'{regime.capitalize()} probes · representative image trials')
    pixels = dict(cmap='RdBu_r', vmin=-extent, vmax=extent)
    axes.flat[0].imshow(target.reshape(shape), **pixels)
    axes.flat[0].set_title('Target', size=10)
    for number, method in enumerate(METHODS, start=1):
        ax = axes.flat[number]
        policy = 'oracle' if method in AMP_METHODS else 'none'
        group = groups[(f'image_{regime}', method, policy)]
        trial = group.get('median_error_image_trial')
        if trial is None:
            ax.text(.5, .5, 'No finite estimate', va='center', ha='center', transform=ax.transAxes)
            ax.set_title(LABELS[method], size=9)
            continue
        path = output/'simulation'/f'image_{regime}'/f'{trial:02}'
        record = next(record for record in read_json(path.with_suffix('.json'))['records']
                      if record['method'] == method)
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
            estimate = align_phase(data[f'{method}_oracle'], target)
        ax.imshow(estimate.real.reshape(shape), **pixels)
        ax.set_title(f"{LABELS[method]} · NMSE {record['phase_nmse']:.3g}", size=9)
    for ax in axes.flat:
        ax.set_axis_off()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = read_json(args.summary)
    if any(group['recorded'] != group['attempted'] for group in summary['completion'].values()):
        parser.error('The figure set requires a complete campaign')
    cfg = summary['metadata']['config']
    groups = {(row['study'], row['method'], row['variance_policy']): row for row in summary['simulation']}
    legend = dict(loc='upper center', ncols=5, bbox_to_anchor=(.5, .15), fontsize=8, frameon=False)
    with PdfPages(args.output, metadata={'Title': 'Transmission-matrix retrieval benchmark I12',
                                       'CreationDate': None, 'ModDate': None}) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        fig.subplots_adjust(bottom=.24)
        draw_sweep(ax, groups, [(ratio, f'ratio_gaussian_{ratio:02}') for ratio in cfg['oversampling_ratios']],
                   'Measurements per input mode (P/N)', 'Complex Gaussian probes · sampling ratio')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, **legend)
        save(pdf, fig, 'Lines show trial medians, shading shows quartiles. Plot floor 1e-10. Failed fits count toward coverage.')

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
        fig.subplots_adjust(bottom=.24, wspace=.3)
        for ax, regime in zip(axes, ('amplitude', 'phase')):
            draw_sweep(ax, groups, [(value, f'noise_{regime}_{value:g}') for value in cfg['noise_levels']],
                       'Complex field-noise scale σ', f'Binary {regime} probes')
            ax.set_xscale('log')
        fig.legend(handles, labels, **legend)
        save(pdf, fig, 'Lines show trial medians, shading shows quartiles. Plot floor 1e-10. Failed fits count toward coverage.')

        for regime in ('gaussian', 'amplitude', 'phase'):
            plot_images(pdf, groups, args.run, cfg, regime)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=.48, bottom=.12)
        for ax, arm in zip(axes, ('nominal', 'matched')):
            rows = [row for row in summary['rice'] if row['arm'] == arm]
            x = np.arange(len(rows))
            for offset, split in [(-.19, 'validation'), (.19, 'test')]:
                scores = [row.get(split, {}).get('relative_residual', np.nan) for row in rows]
                ax.bar(x + offset, scores, width=.36, label=split.capitalize())
            ax.set_xticks(x, [f"{LABELS[row['method']]}\n{row['finite']}/{row['attempted']}" for row in rows],
                          fontsize=8)
            ax.set(ylabel='Relative amplitude residual',
                   title=f"Rice measured data · {arm} · {cfg[f'rice_{arm}_cpu_limit_seconds']} CPU seconds per detector")
            ax.legend(frameon=False)
        save(pdf, fig, 'Detector counts under each method include finite estimates. The supplied test split has five patterns.')


if __name__ == '__main__':
    main()
