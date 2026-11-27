"""Build the figure-by-figure ICCP 2017 reproduction report from the paper and I12."""

import argparse
import csv
from functools import lru_cache
import hashlib
import io
import json
import logging
from pathlib import Path
from xml.sax.saxutils import escape

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image as Raster
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from run_i12 import image_target
from tmbench.benchmark import AMP_METHODS
from tmbench.solvers.prvamp.utils import align_phase


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs/results'
PAPER_DOI = 'https://doi.org/10.1109/ICCPHOT.2017.7951483'
RELEASE = 'https://dsp.rice.edu/research/transmissionmatrices/'
LABELS = dict(prgamp='prGAMP', prsamp='prSAMP', prvbem='prVBEM', prvamp='prVAMP',
              gs50='GS-50', gs100='GS-100', wf50='WF-50', wf500='WF-500',
              phaselift='PhaseLift', phasemax='PhaseMax')
DIMENSIONS = (64, 256, 1024)
PAPER_TIMINGS = {
    'prgamp': (.07, .39, 9.64), 'prsamp': (3.52, 54.80, 826.79),
    'prvbem': (.22, 2.66, 105.74), 'prvamp': (.44, .48, 6.79),
    'gs50': (.15, 1.78, 30.56), 'gs100': (.26, 3.41, 60.26),
    'wf50': (.09, 1.50, 30.30), 'wf500': (.69, 14.22, 244.54),
    'phaselift': (5.00, 201.00, 12321), 'phasemax': (.48, 2.85, 104.46),
}
PAPER_PREPARATION = {'prvamp': (.52, .50, 19.92), 'gs50': (.59, .62, 24.80)}
# Read at the twelve integer ticks of the embedded Figure 3 raster. Boundary
# values are visible levels or bounds, as annotated in the CSV and report.
PAPER_RATIO = {
    'gs50':      (1, 1, 1, 1, 1, 1, 1, .93, .54, .09, .02, 0),
    'gs100':     (1, 1, 1, 1, 1, 1, .72, 0, 0, 0, 0, 0),
    'prgamp':    (1, 1, .43, .06, 0, 0, 0, 0, 0, 0, 0, 0),
    'prsamp':    (1, 1, .04, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'prvbem':    (1, .88, .65, .02, 0, 0, 0, 0, 0, 0, 0, 0),
    'phaselift': (1, .94, .79, .12, .01, 0, 0, 0, 0, 0, 0, 0),
    'phasemax':  (1, 1, .77, .50, .26, .12, .03, 0, 0, 0, 0, 0),
    'wf50':      (1, 1, .84, .62, .43, .31, .24, .20, .16, .14, .12, .10),
    'wf500':     (1, .92, .26, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'prvamp':    (1, .80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
}
NOISE_METHODS = ('prvbem', 'prsamp', 'prgamp', 'prvamp', 'gs50', 'gs100',
                 'phasemax', 'wf500', 'phaselift')
NOISE_LEVELS = (.1, 1, 5, 15)
FIGURE_PAGES = {1: 1, 2: 5, 3: 6, 4: 7, 5: 7, 6: 9, 7: 10, 8: 10,
                9: 11, 10: 12, 11: 13, 12: 13, 13: 14, 14: 15}
RICE_DIR = ROOT / 'data/raw/phase_slm_40x40'
INK, ACCENT, MUTED = '#192c3b', '#085f66', '#4d6476'
PLOT_PAPER, PLOT_LOCAL = '#38649b', '#008179'
FIGURE_COLORS = ('#9852b0', '#633779', '#c48300', '#2a7d47', '#008ca2',
                 '#c12e3c', '#1549a5', '#8b621b', '#bd5f12', '#182c3b')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            sha.update(block)
    return sha.hexdigest()


class Evidence:
    def __init__(self, paper, run):
        self.paper = paper
        self.run = run
        self.pdf = PdfReader(paper)
        self.summary = read_json(RESULTS / 'i12_summary.json')
        self.sim = {(r['study'], r['method'], r['variance_policy']): r for r in self.summary['simulation']}
        self.timing = {t['modes']: {r['method']: r for r in t['records']}
                       for t in read_json(RESULTS / 'i12_timing.json')['timing']}
        self.manifest = read_json(ROOT / 'data/manifests/rice_phase_slm_40x40.json')

    @lru_cache(maxsize=None)
    def image(self, page, number):
        return self.pdf.pages[page - 1].images[f'/Im{number}'].image.convert('RGB')

    def group(self, study, method, policy=None):
        if policy is None:
            policy = 'oracle' if method in AMP_METHODS else 'none'
        return self.sim[(study, method, policy)]

    @lru_cache(maxsize=None)
    def noise(self, number):
        paper = np.asarray(self.image(9 if number == 15 else 13, number), dtype=int)
        blue = (paper[:, :, 2] - paper[:, :, 0] > 45) & (paper[:, :, 2] < 175) & (paper[:, :, 1] < 100)
        readings = []
        for x, top, bottom in ((106, 37, 388), (916, 37, 388), (106, 487, 837), (916, 487, 837)):
            values = []
            for center in x + 80*np.arange(len(NOISE_METHODS)):
                bar = np.flatnonzero((blue[top:bottom, center-2:center+3]).sum(axis=1) >= 3)
                height = 0. if not len(bar) else 1. if bar[0] <= 1 else 1 - bar[0]/(bottom-top)
                values.append(round(height, 3))
            readings.append(values)
        return readings

    def rice_data(self, split, indices=None):
        probes, capture = f'XH_{split}', f'YH_squared_{split}'
        if split == 'test':
            for name in (probes, capture):
                path = RICE_DIR / f'{name}.mat'
                if file_hash(path) != self.manifest['file_layout'][path.name]['sha256']:
                    raise ValueError(f'Rice test-file hash mismatch: {path.name}')
        what = slice(None) if indices is None else indices
        with h5py.File(RICE_DIR / f'{probes}.mat') as handle:
            x = handle[probes][:, what].T / 255.
        with h5py.File(RICE_DIR / f'{capture}.mat') as handle:
            intensity = handle[capture][:, what].T
        return x, intensity

    @lru_cache(maxsize=1)
    def figure13(self):
        probes, captures = self.rice_data('test')
        matches = []
        for image_id, speckle_id, name in ((36, 37, 'Smiley'), (40, 41, 'Digit 4'), (44, 45, 'Letter M')):
            ref = np.asarray(self.image(14, image_id).convert('L')) > 127
            distances = [np.mean((row.reshape(40, 40, order='F') > 0) != ref) for row in probes]
            index = int(np.argmin(distances))
            if distances[index] != 0 or distances.count(0) != 1:
                raise ValueError(f'No unique exact Rice pattern match for {name}')
            frame = captures[index].reshape(256, 256, order='F')
            raster = np.asarray(self.image(14, speckle_id).convert('L'))
            correlation = float(np.corrcoef(frame.ravel(), raster.ravel())[0, 1])
            if correlation < .98:
                raise ValueError(f'Figure 13 camera frame does not match Rice release: {name}')
            matches.append((name, index, correlation, probes[index].reshape(40, 40, order='F'), frame))
        return matches


def raster_grid(images, columns, labels=None, *, cell=(330, 285)):
    """Lay out original raster panels without changing their data values."""
    from PIL import ImageDraw, ImageFont, ImageOps

    width, height = cell
    grid = Raster.new('RGB', (columns*width, ((len(images)+columns-1)//columns)*height), 'white')
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 18)
    draw = ImageDraw.Draw(grid)
    for position, im in enumerate(images):
        x, y = position % columns * width, position // columns * height
        thumb = ImageOps.contain(im, (width - 14, height - (32 if labels else 10)))
        grid.paste(thumb, (x+(width-thumb.width)//2, y))
        if labels:
            draw.text((x+width//2, y+height-22), labels[position], fill=INK, font=font, anchor='mm')
    return grid


def pil_plot(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    return Raster.open(buffer).convert('RGB')


def paper_qualifier(value):
    return 'axis ceiling' if value == 1 else 'baseline or visually obscured' if value == 0 else 'digitized estimate'


def ratio_plot(data):
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 5.2), sharex=True)
    for color, (method, original) in zip(FIGURE_COLORS, PAPER_RATIO.items()):
        ratios = np.arange(1, 13)
        dash = '--' if method in ('gs50', 'wf50') else '-'
        axes[0].plot(ratios, original, dash, color=color, linewidth=1.7, label=LABELS[method])
        score = np.array([data.group(f'ratio_gaussian_{r:02}', method)['phase_nmse']['median'] for r in ratios])
        axes[1].plot(ratios, np.minimum(score, 1), dash, color=color, linewidth=1.7)
        outside = score > 1
        axes[1].scatter(ratios[outside], np.ones(outside.sum()), s=17, color=color, marker='^', zorder=4)
    for ax, title in zip(axes, ('Published figure, redrawn from the visible traces', 'I12, paired simulations')):
        ax.set(title=title, ylabel='Median NMSE', ylim=(-.025, 1.06))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.25)
    axes[1].set(xlabel='Measurements per input mode (P/N)', xticks=np.arange(1, 13))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=.20, hspace=.31)
    return pil_plot(fig)


def noise_plot(data, regime):
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.25))
    source = data.noise(15 if regime == 'amplitude' else 33)
    short = ('VBEM', 'SAMP', 'GAMP', 'VAMP', 'GS-50', 'GS-100', 'P-Max', 'WF-500', 'P-Lift')
    x = np.arange(len(NOISE_METHODS))
    for ax, sigma, original in zip(axes.flat, NOISE_LEVELS, source):
        rows = [data.group(f'noise_{regime}_{sigma:g}', method) for method in NOISE_METHODS]
        values = np.array([r.get('phase_nmse', {}).get('median', np.nan) for r in rows])
        ax.bar(x-.19, original, width=.37, color=PLOT_PAPER)
        ax.bar(x+.19, np.minimum(values, 1.), width=.37, color=PLOT_LOCAL)
        for mask, y, marker, color in ((values > 1., 1.02, '^', PLOT_LOCAL),
                                      (~np.isfinite(values), .043, 'x', '#a73236')):
            ax.scatter(x[mask]+.19, [y]*np.count_nonzero(mask), marker=marker, s=21, color=color, zorder=4)
        ax.set(ylim=(0, 1.1), title=f'σ = {sigma:g}', xticks=x)
        ax.set_xticklabels(short, rotation=35, ha='right', fontsize=7.3)
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Median NMSE')
    axes[1, 0].set_ylabel('Median NMSE')
    legend = [Patch(color=PLOT_PAPER, label='Paper, digitized'), Patch(color=PLOT_LOCAL, label='I12'),
              Line2D([], [], color=PLOT_LOCAL, marker='^', linestyle='none', label='I12 > 1'),
              Line2D([], [], color='#a73236', marker='x', linestyle='none', label='I12 no finite estimate')]
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=7.7, frameon=False)
    fig.subplots_adjust(bottom=.15, hspace=.48, wspace=.2)
    return pil_plot(fig)


def simulated_image_grid(data, regime):
    target = image_target(1024)
    extent = float(target.max())
    order = ('prvbem', 'prsamp', 'prgamp', 'prvamp', 'gs50', 'gs100',
             'phasemax', 'wf500', 'phaselift', 'wf50')
    fig, axes = plt.subplots(4, 3, figsize=(5.1, 6.3), layout='constrained')
    axes.flat[0].imshow(target.reshape(32, 32), cmap='RdBu_r', vmin=-extent, vmax=extent, interpolation='nearest')
    axes.flat[0].set_title('I12 target', fontsize=8)
    for ax, method in zip(list(axes.flat)[1:], order):
        row = data.group(f'image_{regime}', method)
        trial = row.get('median_error_image_trial')
        if trial is None:
            ax.set_title(LABELS[method], fontsize=8)
            ax.text(.5, .5, 'No finite\nestimate', va='center', ha='center', fontsize=8)
            continue
        path = data.run / 'simulation' / f'image_{regime}' / f'{trial:02}'
        score = next(r['phase_nmse'] for r in read_json(path.with_suffix('.json'))['records'] if r['method'] == method)
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
            x = align_phase(arrays[f'{method}_oracle'], target).real.reshape(32, 32)
        ax.imshow(x, cmap='RdBu_r', vmin=-extent, vmax=extent, interpolation='nearest')
        ax.set_title(f'{LABELS[method]}  {score:.3g}', fontsize=7.7)
    for ax in axes.flat:
        ax.set_axis_off()
    return pil_plot(fig)


def measured_pairs(patterns, intensities, titles):
    fig, axes = plt.subplots(2, len(titles), figsize=(7.2, 4.0), squeeze=False, layout='constrained')
    for i, title in enumerate(titles):
        axes[0, i].imshow(patterns[i].reshape(40, 40, order='F'), cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        axes[0, i].set_title(title, fontsize=9)
        captured = intensities[i].reshape(256, 256, order='F')
        axes[1, i].imshow(captured, cmap='gray', vmin=0, vmax=np.percentile(captured, 99.5))
        axes[1, i].set_title('Camera intensity', fontsize=8)
    for ax in axes.flat:
        ax.set_axis_off()
    return pil_plot(fig)


def measured_error_plot(data):
    x, captured = data.rice_data('test')
    with np.load(RESULTS/'i12_rice_estimates.npz', allow_pickle=False) as archive:
        ids = archive['detector_ids']
        rows = {(str(arm), str(method)): A for arm, method, A in
                zip(archive['arms'], archive['methods'], archive['A'])}
    observed = np.sqrt(captured[:, ids].astype(float))
    valid = captured[:, ids] < 255
    methods = ('wf500', 'gs100', 'phaselift', 'phasemax', 'prvamp')
    scores = {}
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout='constrained')
    for method in methods:
        predicted = np.abs(x @ rows['nominal', method].T)
        sse = np.sum(np.where(valid, (predicted - observed)**2, 0), axis=1)
        energy = np.sum(np.where(valid, observed**2, 0), axis=1)
        saved = next(row['test']['relative_residual'] for row in data.summary['rice']
                     if row['arm'] == 'nominal' and row['method'] == method)
        np.testing.assert_allclose(np.sqrt(sse.sum()/energy.sum()), saved, rtol=1e-10)
        scores[method] = np.sqrt(sse / energy)
        ax.plot(np.arange(len(x)), scores[method], marker='o', label=LABELS[method])
    ax.set(xticks=range(len(x)), xlabel='Held-out release pattern ID (zero-based)', ylabel='Relative amplitude residual')
    ax.legend(frameon=False, ncol=3, fontsize=7.5)
    ax.grid(alpha=.22)
    return pil_plot(fig), scores


def coefficient_slice():
    with np.load(RESULTS/'i12_rice_estimates.npz', allow_pickle=False) as archive:
        group = next(i for i, (a, m) in enumerate(zip(archive['arms'], archive['methods']))
                     if a == 'nominal' and m == 'wf500')
        x = archive['A'][group, 0].copy()
        detector = int(archive['detector_ids'][0])
    reference = int(np.argmax(np.abs(x)))
    x *= np.exp(-1j * np.angle(x[reference]))
    matrix = x.reshape(40, 40, order='F')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), layout='constrained')
    for ax, values, title, cmap, limits in (
        (axes[0], np.abs(matrix), 'Coefficient magnitude', 'viridis', (0, np.percentile(np.abs(matrix), 99))),
        (axes[1], np.angle(matrix), 'Relative phase (rad)', 'twilight', (-np.pi, np.pi)),
    ):
        raster = ax.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        fig.colorbar(raster, ax=ax, fraction=.05, pad=.03)
    return pil_plot(fig), detector


class Report:
    def __init__(self, output):
        self.output = output
        self.story = []
        fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
        for name, suffix in [('ReportSans', ''), ('ReportBold', '-Bold'), ('ReportItalic', '-Oblique')]:
            pdfmetrics.registerFont(TTFont(name, str(fonts/f'DejaVuSans{suffix}.ttf')))
        pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportBold',
                                      italic='ReportItalic', boldItalic='ReportBold')
        self.styles = {
            'kicker': ParagraphStyle('kicker', fontName='ReportBold', fontSize=8, leading=13,
                                     textColor=colors.HexColor(ACCENT), spaceAfter=6),
            'title': ParagraphStyle('title', fontName='ReportBold', fontSize=23, leading=28,
                                     textColor=colors.HexColor(INK), spaceAfter=10),
            'section': ParagraphStyle('section', fontName='ReportBold', fontSize=11, leading=15,
                                       textColor=colors.HexColor(INK), spaceAfter=4, spaceBefore=5),
            'body': ParagraphStyle('body', fontName='ReportSans', fontSize=9.4, leading=14,
                                    textColor=colors.HexColor(INK), spaceAfter=8),
            'small': ParagraphStyle('small', fontName='ReportSans', fontSize=8, leading=11.4,
                                     textColor=colors.HexColor(MUTED), spaceAfter=6),
            'cell': ParagraphStyle('cell', fontName='ReportSans', fontSize=8.1, leading=11.8,
                                    textColor=colors.HexColor(INK), alignment=TA_LEFT),
            'head': ParagraphStyle('head', fontName='ReportBold', fontSize=8.1, leading=11.8,
                                    textColor=colors.white, alignment=TA_LEFT),
        }

    def text(self, value, style='body'):
        self.story.append(Paragraph(value, self.styles[style]))

    def page(self, label, title, deck=None):
        if self.story:
            self.story.append(PageBreak())
        self.text(label, 'kicker')
        self.text(title, 'title')
        if deck:
            self.text(deck)

    def table(self, headings, rows, widths):
        rendered = [[Paragraph(str(value), self.styles['head']) for value in headings]]
        rendered += [[Paragraph(str(value), self.styles['cell']) for value in row] for row in rows]
        table = Table(rendered, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(INK)),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f5f8'), colors.white]),
            ('LINEBELOW', (0, 0), (-1, 0), .8, colors.HexColor('#b9d2da')),
            ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#b9d2da')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        self.story.extend((table, Spacer(1, 11)))

    @staticmethod
    def flow_image(im, width, height):
        factor = min(width/im.width, height/im.height)
        stream = io.BytesIO()
        im.convert('RGB').save(stream, format='PNG')
        stream.seek(0)
        return Image(stream, width=im.width*factor, height=im.height*factor, hAlign='CENTER')

    def picture(self, im, caption, *, width=499, height=340):
        self.story.extend((self.flow_image(im, width, height), Spacer(1, 6)))
        self.text(caption, 'small')

    def pair(self, left, right, left_title, right_title, height=320):
        cells = [[Paragraph(left_title, self.styles['cell']), Paragraph(right_title, self.styles['cell'])],
                 [self.flow_image(left, 236, height), self.flow_image(right, 236, height)]]
        table = Table(cells, colWidths=[249, 249], hAlign='CENTER')
        table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eaf2f6')),
                                   ('TOPPADDING', (0, 0), (-1, -1), 7),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
        self.story.extend((table, Spacer(1, 7)))

    def figure(self, number, title, deck=None):
        self.page(f'PAPER FIGURE {number:02}  /  PDF PAGE {FIGURE_PAGES[number]}', title, deck)

    def build(self):
        def footer(canvas, doc):
            canvas.setStrokeColor(colors.HexColor('#d0dce3'))
            canvas.line(47, 38, A4[0]-47, 38)
            canvas.setFont('ReportSans', 8)
            canvas.setFillColor(colors.HexColor(MUTED))
            canvas.drawString(47, 25, 'TM Retrieval  /  ICCP 2017 benchmark reproduction')
            canvas.drawRightString(A4[0]-47, 25, str(doc.page))

        pdf = SimpleDocTemplate(str(self.output), pagesize=A4, leftMargin=42, rightMargin=42,
                                topMargin=42, bottomMargin=48,
                                title='Coherent inverse scattering | Benchmark reproduction report',
                                author='TM Retrieval project')
        pdf.build(self.story, onFirstPage=footer, onLaterPages=footer)


def overview(report, data):
    s = data.summary['completion']
    report.page('BENCHMARK REPRODUCTION  /  ICCP 2017', 'Coherent inverse scattering via transmission matrices',
                'Figure-by-figure and table-by-table review of Metzler, Sharma, Nagesh, Baraniuk, Cossairt, '
                'and Veeraraghavan. The comparison uses the attached 16-page paper, the Rice release, '
                'and the verified TM Retrieval I12 results.')
    report.table(('Part', 'What this report reproduces'), [
        ('Tables 1–3', 'Published timing numbers, preparation costs, and solver settings alongside the matching I12 records.'),
        ('Figures 3, 6 and 11', 'Recreated numerical charts from the visible paper traces and bars, paired with I12 values.'),
        ('Figures 2, 7 and 10', 'Original simulation montages and new panels regenerated from the saved I12 coefficient arrays.'),
        ('Figures 1, 4, 5, 8, 9, 12, 13 and 14',
         'The published optical evidence, with new Rice data panels where the released measurements can be traced.'),
        ('Additional analysis', 'Complete measured fits, pilot and constraint failures, train-only variances, and per-pattern sensitivity.'),
    ], [132, 367])
    report.text('Main findings', 'section')
    report.text(f"I12 recorded <b>{s['simulation']['recorded']:,} simulated fits</b> and "
                f"<b>{s['rice']['recorded']} measured detector fits</b>. The simulations recover the paper’s "
                'main qualitative contrast between zero-mean phase probes and non-zero-mean binary amplitude '
                'probes. Several quantitative curves differ, especially for PhaseLift and PhaseMax.')
    report.text('Three 40 × 40 input patterns in Figure 13 match the released test arrays exactly. '
                'The corresponding 256 × 256 camera frames also closely match the paper’s displayed speckle. '
                'The raw captures and the paper’s subsequent reconstructions have distinct roles in the report.')
    report.text('How to read the evidence', 'section')
    report.text('<b>Published</b> indicates a table transcription or an original paper panel. '
                '<b>Digitized</b> indicates an estimate from a printed graph, with boundaries flagged. '
                '<b>I12</b> indicates our independently checked run. The raw detector data and '
                'the phase-aligned coefficient error of a known simulation target are different measures. '
                'I12 evaluates measured predictions by held-out amplitude error.')
    report.text(f'<link href="{PAPER_DOI}">Paper DOI 10.1109/ICCPHOT.2017.7951483</link>  ·  '
                f'<link href="{RELEASE}">Rice public release</link>', 'small')


def timing_tables(report, data):
    report.page('PAPER TABLE 1  /  PDF PAGE 4', 'Isolated solver timing',
                'One phase retrieval fit with 12N circular complex Gaussian measurements. '
                'The three input grids are 8 × 8, 16 × 16 and 32 × 32. Times below are seconds per fit.')
    rows = []
    for method, values in PAPER_TIMINGS.items():
        local = []
        for n in DIMENSIONS:
            fit = data.timing[n][method]
            marker = ' *' if fit['status'] == 'timed_out' else ''
            local.append(f"{fit['solve_wall_seconds']:.3f}{marker}")
        rows.append((LABELS[method], *(f'{value:,.2f}' for value in values), *local))
    report.table(('Method', 'Paper<br/>64', 'Paper<br/>256', 'Paper<br/>1,024',
                  'I12<br/>64', 'I12<br/>256', 'I12<br/>1,024'), rows, [94, 61, 62, 83, 62, 62, 75])
    p1 = data.timing[1024]['phaselift']
    p2 = data.timing[1024]['prsamp']
    report.text('What the numbers support', 'section')
    report.text(f"The published PhaseLift run at 1,024 modes took 12,321 s, about 3 h 25 min. "
                f"I12 stopped at {p1['iterations']} PhaseLift updates and observed "
                f"{p1['solve_wall_seconds']:.2f} s wall time. I12 prSAMP completed "
                f"{p2['completed_sweeps']} sweeps and part of the next sweep after its VBEM warm start. "
                '* marks budget stops. Use these rows to see '
                'resource limits. They cannot establish speedups for equal work.')
    report.text('I12 measured a single isolated, one-thread timing input at each size. Its solve time '
                'includes detector-specific initialization and prSAMP’s VBEM warm start. '
                'Shared matrix preparation is separate. Effective versions, backend details, '
                'and timing aggregation for the original table are unspecified. WF-500 stopped on '
                'the local convergence rule after 307, 301, and 314 updates at the three sizes.')

    report.page('PAPER TABLE 2  /  PDF PAGE 4', 'Reusable matrix preparation',
                'The paper separated the economy SVD used by prVAMP and the pseudoinverse used by GS '
                'from fitting. Both operate on the known probe matrix and are reusable across detectors.')
    rows = []
    for method, name in [('prvamp', 'Economy SVD'), ('gs50', 'Pseudoinverse')]:
        for n, paper in zip(DIMENSIONS, PAPER_PREPARATION[method]):
            rec = data.timing[n][method]
            rows.append((name, f'{12*n:,} × {n:,}', f'{paper:.2f}', f"{rec['prepare_wall_seconds']:.3f}",
                         f"{rec['prepare_cpu_seconds']:.3f}"))
    report.table(('Step', 'Probe matrix P × N', 'Paper seconds', 'I12 wall s', 'I12 CPU s'),
                 rows, [118, 118, 95, 85, 83])
    report.text('How to account for preparation', 'section')
    report.text('A repeat fit on the same design can reuse the factorization. The cost per fitted detector '
                'depends on how many detectors share it. Table 1 reports per-fit solver work. '
                'Table 2 gives the corresponding shared preparation charge. A 32-detector wall-clock '
                'total also depends on work-group scheduling and host load.')
    report.text('The I12 eight-second comparison charged per-detector initialization, including spectral '
                'initialization and VBEM warm starts, to each fit. It recorded design preparation and the '
                'shared VBEM pilot separately. Short CPU measurements can round to zero on the local '
                'Windows counter. The wall-time column resolves those short preparation steps.')


def settings_table(report):
    report.page('PAPER TABLE 3  /  PDF PAGE 10', 'Solver settings and implementation differences')
    report.table(('Method', 'Published settings', 'I12 implementation'), [
        ('prGAMP', '50 iterations, random start, signal and noise priors, damping 0.8.',
         '50 iterations, seeded Gaussian start. Interprets 0.8 as the old-state weight.'),
        ('prSAMP', '50 iterations, prVBEM initialization, signal and noise priors, damping 0.9.',
         'Up to 50 VBEM warm-start iterations followed by up to 50 sweeps. Uses 0.1 new-state weight.'),
        ('prVBEM', '50 iterations, built-in initialization, noise setting supplied, otherwise defaults.',
         'Seeded local implementation. Holds simulation variances fixed and learns both on measured data.'),
        ('prVAMP', '50 iterations, random start, signal and noise priors, damping 0.8.',
         '50 iterations, seeded Gaussian start and 0.2 weight on new natural parameters.'),
        ('GS', '50 or 100 iterations, random start, defaults.',
         'Seeded starts, two budgets, shared economy-SVD pseudoinverse.'),
        ('WF', '50 or 500 iterations, spectral initialization, defaults.',
         '50 power iterations for a spectral start, original ramp and local gradient stopping rule.'),
        ('PhaseLift', '50 iterations, random start, defaults.',
         '50 proximal updates of the implicit trace-penalized formulation, seeded lifted initialization.'),
        ('PhaseMax', '50 iterations, random start, defaults.',
         '50 ADMM updates. Random simulation anchor and training spectral anchor on Rice.'),
    ], [64, 186, 249])
    report.text('What the Bayesian methods share', 'section')
    report.text('The paper used zero-mean signal priors and oracle variances in simulations. '
                'For measured rows, it shared signal and noise variance estimates from one training-only '
                'prVBEM pilot across prGAMP, prVAMP and prSAMP. I12 follows that transfer rule. '
                'Standalone measured prVBEM fits its own variances, and prSAMP’s per-row VBEM warm start '
                'uses the transferred variances. The exact historical code that produced the PhaseLift '
                'and PhaseMax tables and figures is unspecified.')


def first_figure(report, data):
    report.figure(1, 'From the displayed object to a recovered image',
                  'The optical chain sends coherent light through an input pattern and a scatterer. '
                  'The camera measures speckle. The printed example shows a smiley target, its speckle, '
                  'and a reconstructed object.')
    left, right = data.image(1, 1), data.image(1, 2)
    diagram = Raster.new('RGB', (left.width+right.width, left.height), 'white')
    diagram.paste(left, (0, 0))
    diagram.paste(right, (left.width, 0))
    report.picture(diagram, 'Published experiment sketch from the supplied PDF.', height=260)
    grid = raster_grid([data.image(1, n) for n in (3, 4, 5)], 3,
                       ['64 × 64 source', '256 × 256 speckle', '64 × 64 reconstruction'], cell=(320, 255))
    report.picture(grid, 'Published image sequence, arranged in acquisition order.', height=148)
    report.text('Calibration and imaging answer different questions', 'section')
    report.text('During <b>calibration</b>, the input pattern is known and the transmission matrix is unknown. '
                'During <b>imaging</b>, a previously calibrated matrix is available and the input image is unknown. '
                'I12 measured calibration quality by predicting held-out camera amplitudes. '
                'This opening figure illustrates the later imaging step and includes no numerical accuracy metric.')


def simulation_figure(report, data, number, regime, embedded, assessment):
    regime_label = {'gaussian': 'complex Gaussian', 'amplitude': 'binary amplitude', 'phase': 'binary phase'}[regime]
    report.figure(number, f'Image recovery with {regime_label} probes',
                  'A 32 × 32 simulated recovery problem with 1,024 input modes and 12,288 measurements. '
                  'Ten I12 trials share the same procedural smiley and use paired probe realizations across methods.')
    report.pair(data.image(FIGURE_PAGES[number], embedded), simulated_image_grid(data, regime),
                '<b>Published</b>  ·  the paper’s reconstruction montage',
                '<b>I12</b>  ·  regenerated from saved trial coefficients', height=335)
    report.text('Reading the comparison', 'section')
    report.text(assessment)
    report.text('The I12 panel shows the real part of each phase-aligned estimate on a shared signed display '
                'scale. Its titles report the corresponding phase-only NMSE. The representative trial is the '
                'lower of the two middle errors when there are ten finite fits. When the paper and I12 use '
                'different target drawings or solver versions, compare recovery behavior across probe types '
                'and budgets.', 'small')


def oversampling_figure(report, data):
    report.figure(3, 'How many Gaussian measurements are needed?',
                  'Both panels compare 256 unknown coefficients across 12 measurement ratios. Each marker '
                  'represents the median of 50 trials under the corresponding protocol.')
    report.picture(ratio_plot(data), 'Figure 3 redrawn above and reproduced with I12 below. Upward triangles mark '
                   'I12 medians above the paper’s displayed 0–1 range. Full values are in the companion CSV.', height=355)
    report.text('What stayed consistent', 'section')
    report.text('prVAMP recovers early in both experiments. Near 3N the paper’s trace is at the visible '
                'baseline and I12 gives median NMSE 4.23 × 10<super>−4</super>. I12 WF-500 also improves '
                'sharply between 3N and 4N. A solver still has to be judged at its named iteration budget.')
    report.text('What changed', 'section')
    report.text('The paper’s PhaseLift and PhaseMax traces approach the baseline at high ratios. '
                'At 12N, I12 medians are approximately 0.931 and 0.939 with the local trace-penalized '
                'and random-anchor ADMM implementations. This discrepancy remains visible in the reproduced '
                'curve. Finite coefficients can still have poor normalized error.')
    report.text('Digitized traces are estimates from the embedded chart. Points at zero or one reproduce the visible '
                'baseline or ceiling. Overlapping traces can hide separate values at those boundaries.', 'small')


def calibration_figure(report, data):
    report.figure(4, 'What a calibration pair contains',
                  'A calibration pair is a known SLM pattern and the camera frame recorded during its exposure. '
                  'A phase pattern assigns either 0 or π to each SLM cell.')
    left, right = data.image(7, 8), data.image(7, 9)
    amplitude = Raster.new('RGB', (left.width+right.width, left.height), 'white')
    amplitude.paste(left, (0, 0))
    amplitude.paste(right, (left.width, 0))
    images = raster_grid([amplitude, data.image(7, 10)], 1,
                         ['Published binary amplitude examples', 'Published binary phase examples'],
                         cell=(900, 230))
    report.picture(images, 'Original calibration pattern and speckle montages. The colored phase display encodes radians.',
                   height=205)
    frozen = read_json(ROOT/'data/manifests/rice_phase_slm_40x40_split.json')
    indices = sorted(frozen['splits']['train'])[:3]
    x, camera = data.rice_data('train', indices)
    report.picture(measured_pairs(x, camera, [f'Training pattern {i}' for i in indices]),
                   'New panels rebuilt from three recorded Rice phase-training pairs, using MATLAB column-order reshaping. '
                   'Displayed contrast is normalized per frame. The measured values remain unchanged for analysis.', height=215)
    report.text('Benchmark scope', 'section')
    report.text('The raw phase release contains 19,200 training-pool pairs and five provided test pairs. '
                'I12 took 17,280 training patterns and 1,920 validation patterns from the pool, retained the five '
                'test pairs, and evaluated 32 of the 65,536 camera pixels. Each fitted row keeps all 1,600 input modes.')


def amplitude_image_example(report, data):
    report.figure(5, 'A single 64 × 64 amplitude imaging result',
                  'The paper uses a digit 4 to show how one recorded speckle pattern is turned into an image. '
                  'prVAMP is the method reported for calibrating the transmission matrix used here.')
    grid = raster_grid([data.image(7, n) for n in range(11, 15)], 2,
                       ['(a) Source pattern', '(b) Captured speckle',
                        '(c) Reconstruction', '(d) After BM3D'], cell=(350, 300))
    report.picture(grid, 'Published image panels with their original order. The labels distinguish the raw '
                   'camera observation, inverse reconstruction and separate denoising stage.', height=353)
    report.text('Evidence available for this panel', 'section')
    report.text('The pre-denoising image shows recognizable structure along with visible speckle-like artifacts. '
                'BM3D makes the digit easier to see. The panel has no coefficient ground truth or numerical '
                'image score. I12 did not fit this 64 × 64 amplitude dataset or apply post-imaging BM3D. '
                'The adjacent I12 amplitude comparisons elsewhere in the report use simulations.')


def noise_figure(report, data, number, regime, assessment):
    label = 'Binary amplitude {0, 1}' if regime == 'amplitude' else 'Binary phase {−1, +1}'
    report.figure(number, f'Noise sweep with {label.lower()}',
                  'The experiment uses 256 input modes and 3,072 measurements. It reports median '
                  'phase-aligned coefficient NMSE over 50 trials at each of four plotted noise settings.')
    report.picture(noise_plot(data, regime), 'Published values are digitized from the original four panels. '
                   'I12 display bars stop at 1 when their median is larger. A cross marks no finite I12 estimate. '
                   'Method labels follow the paper’s nine plotted variants.', height=350)
    report.text('What the comparison shows', 'section')
    report.text(assessment)
    report.text('I12 interprets σ as the RMS of circular complex field noise added before magnitude '
                'detection, with total variance σ². The source figure gives the labels 0.1, 1, 5 and 15. '
                'The exact mapping from those labels to the authors’ noise generator is unspecified. '
                'A zero-height source bar means the visible baseline. A source bar at 1.0 is at the plot ceiling.', 'small')


def optical_setup(report, data, number, phase=False):
    if phase:
        report.figure(number, 'Reflective phase-SLM optical setup',
                      'The paper switches to a reflective SLM and uses phase shifts of 0 and π. '
                      'That produces signed known probe coefficients while keeping both commanded states illuminated.')
        photo, sketch = data.image(13, 34), data.image(13, 35)
    else:
        report.figure(number, 'Transmissive amplitude-SLM optical setup',
                      'A 632.8 nm laser illuminates a transmissive SLM from Holoeye. '
                      'The camera observes the field after a diffuser. Binary amplitude probe cells are off or on.')
        left, right = data.image(10, 17), data.image(10, 18)
        photo = Raster.new('RGB', (left.width+right.width, left.height), 'white')
        photo.paste(left, (0, 0))
        photo.paste(right, (left.width, 0))
        sketch = data.image(10, 19)
    report.picture(photo, 'Published apparatus photograph.', height=163)
    report.picture(sketch, 'Published light-path diagram.', height=221)
    report.text('Why the setup matters for the benchmark', 'section')
    if phase:
        report.text('With equally probable binary states the signed probe ensemble has mean zero. '
                    'The paper uses this change to make inexpensive AMP-based calibration, including '
                    'prGAMP, more practical. The two simulation noise sweeps test the corresponding '
                    'difference in probe ensembles. The I12 camera runs use this reflected phase-SLM release.')
    else:
        report.text('The paper reports an LC 2012 transmissive SLM, a 150 mm focusing lens, a 5° '
                    'holographic diffuser, a 10× objective with NA 0.25, and a Grasshopper2 camera. '
                    'The 64 × 64 calibration experiment uses 49,152 probe patterns. The binary {0,1} '
                    'probe ensemble has a positive mean, a meaningful stress test for several AMP and spectral methods.')


def amplitude_montage(report, data):
    report.figure(9, 'Measured amplitude-SLM imaging',
                  'Three published 64 × 64 imaging examples. Each row shows a source pattern, its 256 × 256 '
                  'camera speckle, a reconstruction using the prVAMP-calibrated matrix, and BM3D output.')
    rows = ('Symbol', 'Resolution target', 'RICE lettering')
    columns = ('Source', 'Speckle', 'Reconstruction', 'BM3D')
    grid = raster_grid([data.image(11, n) for n in range(20, 32)], 4,
                       [f'{row} · {column}' for row in rows for column in columns], cell=(300, 270))
    report.picture(grid, 'All twelve source panels are reproduced from the supplied paper. BM3D is an '
                   'image post-processing step.', height=330)
    report.text('Reported filtering and resources', 'section')
    report.text('The authors discuss excluding detector rows above a residual of 0.4 before imaging and '
                'report roughly 98% of rows below that threshold. They used 80 compute nodes with '
                '16 cores and 32–128 GB RAM per node for the parallel TM work. These are experimental '
                'calibration resources. The paper gives no matched per-method image score for the montage.')
    report.text('I12 retained every attempted detector in its measured denominator. '
                'Its held-out errors compare predicted and observed amplitudes at those detectors. '
                'The protocol keeps this calibration metric separate from the appearance of a BM3D image.')


def phase_montage(report, data):
    report.figure(13, 'Measured phase-SLM imaging and the released test pairs',
                  'The published 40 × 40 reconstructions use a prGAMP-calibrated TM. '
                  'The release permits a direct check of the displayed input patterns and recorded speckle.')
    labels = ('Input', 'Camera', 'Reconstruction', 'BM3D')
    grid = raster_grid([data.image(14, n) for n in range(36, 48)], 4, list(labels)*3, cell=(285, 247))
    report.picture(grid, 'Published panels in rows for the smiley, the digit 4, and the letter M. '
                   'Columns 3 and 4 are the authors’ outputs.', height=237)
    pairs = data.figure13()
    report.picture(measured_pairs(np.array([row[3].ravel(order='F') for row in pairs]),
                                  np.array([row[4].ravel(order='F') for row in pairs]),
                                  [f'{name} · test {index}' for name, index, _, _, _ in pairs]),
                   'Columns 1 and 2 regenerated from the pinned test MAT files. Image display limits are set per camera frame.',
                   height=166)
    report.table(('Figure 13 pattern', 'Zero-based test ID', 'SLM cells matching', 'Speckle r'), [
        (name, index, '1,600 / 1,600', f'{correlation:.5f}') for name, index, correlation, _, _ in pairs
    ], [141, 107, 139, 112])
    report.text('Source speckle r is the Pearson correlation between flattened raw intensity and the embedded '
                'grayscale camera panel after column-major reshaping. Display contrast and raster encoding '
                'can change pixel values. The I12 calibration subset covers 32 outputs, so the two image '
                'reconstruction columns are shown as published evidence.', 'small')


def transmission_montage(report, data):
    report.figure(14, 'Transmission-matrix magnitude and phase',
                  'A column of the paper’s matrix corresponds to one input mode and all output pixels. '
                  'The left inset reshapes that column to the camera grid. Larger panels visualize the matrix.')
    pair = raster_grid([data.image(15, 48), data.image(15, 49)], 2,
                       ['Published magnitude', 'Published phase'], cell=(575, 355))
    report.picture(pair, 'Published coefficient views. The authors call out structure in the magnitude map.', height=181)
    local, detector = coefficient_slice()
    report.picture(local, f'I12 illustration of one estimated row for detector {detector}, nominal WF-500. '
                   'All 1,600 input modes are shown. Magnitudes above the 99th percentile share the top display color.', height=166)
    report.text('Which dimension is being displayed', 'section')
    report.text('The reproduced paper inset is a <b>column</b> across camera outputs. The I12 illustration is '
                'a <b>row</b> across input modes for one measured detector. The row’s phase is expressed '
                'relative to its largest-magnitude coefficient. Signed real probes permit an overall '
                'phase and conjugation ambiguity. The released fitted matrices and these visual structures '
                'provide no independent ground-truth coefficient error.')


def rice_table(report, data, arm):
    matches = [row for row in data.summary['rice'] if row['arm'] == arm]
    limit = data.summary['metadata']['config'][f'rice_{arm}_cpu_limit_seconds']
    title = 'Nominal iteration limits' if arm == 'nominal' else 'Eight-second CPU comparison'
    report.page('ADDITIONAL ANALYSIS  /  MEASURED RICE', title,
                f'The same 32 detectors and the same 17,280, 1,920 and five pattern splits across training, '
                f'validation and test. Each detector fit has a {limit}-second process CPU ceiling. '
                'The table retains time-limited and failed outcomes.')
    status_labels = dict(timed_out='CPU limit', max_iter_reached='Iteration limit',
                         timed_out_warm_start='Warm start consumed budget')
    rows = []
    for r in matches:
        values = []
        for split in ('validation', 'test'):
            scores = r.get(split, {})
            values.extend(f'{scores[k]:.3f}' if k in scores else '—' for k in ('relative_residual', 'amplitude_mse'))
        iters = r.get('iterations', {}).get('median')
        cpu = r.get('solve_cpu_seconds', {}).get('median')
        outcome = '<br/>'.join(f'{status_labels.get(k, k)} ({n})' for k, n in sorted(r['statuses'].items()))
        rows.append((LABELS[r['method']], f"{r['finite']}/32", *values,
                     f'{iters:g}' if iters is not None else '—', f'{cpu:.2f}' if cpu is not None else '—', outcome))
    report.table(('Method', 'Finite', 'Val.<br/>rel.', 'Val.<br/>MSE', 'Test<br/>rel.', 'Test<br/>MSE',
                  'Median<br/>updates', 'Median<br/>CPU s', 'Statuses'),
                 rows, [64, 41, 42, 45, 45, 45, 56, 55, 106])
    report.text('Interpretation', 'section')
    if arm == 'nominal':
        report.text('WF-500 has the lowest validation and test residual in this arm, 0.486 and 0.428, '
                    'with a median 91.5 updates before the CPU limit. GS-100 has test residual 0.465 '
                    'and completes its full budget. These two results have full detector coverage. '
                    'Neither arm has any detector meeting a solver convergence criterion.')
    else:
        report.text('Gerchberg–Saxton has the lowest test residual among estimates in this arm, 0.483. '
                    'prSAMP spends its budget on the per-detector prVBEM warm start for all 32 rows. '
                    'PhaseMax has test residual 0.529 as an approximate prediction and a median of one '
                    'update after initialization. Every PhaseMax row exceeds the amplitude constraint tolerance.')
    report.text('Relative residual is the square root of summed squared amplitude error divided by '
                'summed measured amplitude energy. Each split pools all valid scored observations. '
                'Finite coverage uses attempted detectors as the denominator. The five test patterns '
                'were previously exposed in integration reports and provide limited evidence for ranking.', 'small')


def supplemental_analysis(report, data):
    report.page('ADDITIONAL ANALYSIS  /  BAYESIAN SETTINGS', 'What changed when variances came from data?',
                'The Gaussian 12N experiment is paired across variance policies. '
                'It uses 256 modes, 3,072 probes and 50 trials with the same realizations for both policies.')
    rows = []
    for method in ('prgamp', 'prvamp', 'prvbem', 'prsamp'):
        oracle = data.group('ratio_gaussian_12', method, 'oracle')['phase_nmse']
        local = data.group('ratio_gaussian_12', method, 'training_energy')['phase_nmse']
        rows.append((LABELS[method], f"{oracle['median']:.3g}", f"{oracle['q25']:.3g} – {oracle['q75']:.3g}",
                     f"{local['median']:.3g}", f"{local['q25']:.3g} – {local['q75']:.3g}"))
    report.table(('Method', 'Oracle median', 'Oracle Q25–Q75', 'Training median', 'Training Q25–Q75'),
                 rows, [78, 78, 116, 95, 132])
    report.text('Simulation comparison', 'section')
    report.text('Oracle settings use the generated coefficient power and known field-noise variance, '
                'with a small likelihood floor for the noiseless problem. The training-energy rule assumes '
                'that 1% of the mean training intensity is noise energy and allocates the remaining '
                'energy to the prior. This auxiliary run measures sensitivity to the availability of oracle values.')
    pilot = data.summary['rice_pilot']
    report.text('Measured pilot and warm starts', 'section')
    report.table(('Quantity', 'Rice pilot at detector 8208'), [
        ('Initial → learned field-noise variance', f"{pilot['initial_noise']:.5f} → {pilot['noise_variance']:.5f}"),
        ('Initial → learned coefficient prior variance', f"{pilot['initial_prior']:.6f} → {pilot['prior_variance']:.3g}"),
        ('Coefficient norm after the pilot', f"{pilot['estimate_norm']:.3g}"),
        ('Training amplitude relative residual', f"{pilot['training_relative_residual']:.3f}"),
    ], [269, 230])
    report.text('The model attributed almost all measured energy to noise and shrank the fitted row toward zero. '
                'Transferring those settings explains the near-unit accuracy residuals of the measured AMP rows. '
                'The shared pilot estimates two settings once. prSAMP’s warm start estimates coefficients '
                'separately for each detector and consumes part of that detector’s solve budget. '
                'I11 separately observed training residuals of 0.417 when only noise was learned '
                'and 0.376 when both variances stayed fixed on the same pilot case.')

    report.page('ADDITIONAL ANALYSIS  /  TEST PATTERN SENSITIVITY', 'How much does the chosen pattern matter?',
                'Each point combines the amplitude prediction errors of the 32 selected detectors for '
                'one measured test pattern. These are the same archived coefficients as in I12.')
    plot, scores = measured_error_plot(data)
    report.picture(plot, 'Per-pattern relative amplitude residual. The main Rice results pool the error and '
                   'measurement energies before computing the ratio.', height=225)
    report.table(('Nominal run', '0 · Smiley', '1 · M', '2 · Digit 4', '3', '4'),
                 [(LABELS[method], *(f'{value:.3f}' for value in vals)) for method, vals in scores.items()],
                 [109, 78, 78, 88, 73, 73])
    report.text('Inference limits', 'section')
    report.text('Changing the displayed pattern changes measured speckle energy and inference difficulty. '
                'With five supplied patterns, the pooled residual can depend on which patterns were selected. '
                'Use this plot to locate pattern dependence in the frozen release. '
                'An independent held-out recording would support a confirmatory comparison.')


def evaluation_notes(report, data):
    report.page('ADDITIONAL ANALYSIS  /  METRICS AND COVERAGE', 'How to interpret the three error measures')
    report.table(('Measure', 'What gets compared', 'How to read the score'), [
        ('Phase-aligned NMSE', 'Known simulated coefficient vector and fitted vector. '
         'Align the global phase, then divide squared coefficient error by true coefficient energy.',
         'Zero denotes exact coefficient recovery up to global phase. '
         'Values above 1 can occur. The source plots show a 0–1 window.'),
        ('Relative amplitude residual', 'At each held-out pattern and scored detector, compare the predicted '
         'field magnitude with the square root of the recorded intensity.', 'Square root of total squared '
         'amplitude error divided by total measured amplitude energy. A zero prediction gives 1 when '
         'the observed energy is positive.'),
        ('Amplitude MSE', 'The same measured amplitude prediction errors as in relative residual.',
         'Total squared amplitude error divided by the number of valid scored observations. '
         'Its units follow the released intensity counts.'),
    ], [130, 187, 182])
    report.text('How a measured fit produces a score', 'section')
    report.text('A known SLM probe x contains 1,600 signed input coefficients. For one fitted detector row a, '
                'the predicted complex field is x · a with ordinary transpose. Its magnitude gives a predicted '
                'amplitude. In the validation and test splits the estimate stays frozen. The scorer '
                'pools errors and observed energy across valid observations before taking the ratio.')
    report.text('Coverage and model constraints', 'section')
    s = data.summary['completion']
    report.text(f"I12 retained <b>{s['simulation']['finite']:,} of {s['simulation']['attempted']:,}</b> simulated "
                f"estimates and <b>{s['rice']['finite']} of {s['rice']['attempted']}</b> measured estimates. "
                'Finite-only medians and quartiles describe the returned estimates. The full attempted count '
                'still includes failed and uncovered cases. Convergence, amplitude constraint compliance '
                'and prediction error are recorded separately. In the measured PhaseMax cases, for example, '
                'every scored estimate was outside its declared optimization tolerance.')
    report.text('For real probes, complex conjugation can leave the magnitudes unchanged. I12 saves that '
                'ambiguity-aware simulation score separately from the phase-only score used in the '
                'paper comparison. The camera data have no independently measured coefficient reference.')


def sources_page(report, data):
    report.page('REPRODUCIBILITY  /  SOURCES AND FILES', 'How this report was built')
    report.text(f'<b>Paper.</b> Metzler et al., ICCP 2017. DOI <link href="{PAPER_DOI}">'
                '10.1109/ICCPHOT.2017.7951483</link>. Page numbers in this report refer to the '
                'attached 16-page PDF. Figure visuals taken from that PDF are identified as published.')
    report.text('<b>Tables.</b> Paper Tables 1 and 2 were transcribed as published. Table 3 '
                'is reproduced in paraphrase. I12 solve timing uses wall seconds. Matrix '
                'preparation is retained as both wall and process CPU seconds when available.')
    report.text('<b>Digitization.</b> Figure 3 readings were taken at the twelve integer ticks of the embedded '
                'raster and rounded to about two decimal places. Visible baseline and axis-ceiling values '
                'are flagged. Figures 6 and 11 use the first dark-blue bar row at each bar center, '
                'divided by plot height. Raster precision is approximately one to two pixels and '
                'the plot boundaries can conceal distinct underlying values.')
    report.text('<b>Benchmark.</b> The I12 implementation uses shared seeded inputs, known-target '
                'phase-aligned NMSE for simulations, and held-out amplitude error for measured rows. '
                'The reports retain timeouts and oversized finite errors. The paper does not disclose '
                'its exact seeds, backend defaults or target arrays for many simulations. '
                'This report reproduces the design and compares it with the accessible numerical evidence.')
    report.text(f'<b>Data.</b> <link href="{RELEASE}">Rice public release</link>. '
                'Figure 13 pattern matches were checked against the SHA-256-pinned test files. '
                'The comparison uses exact binary mask equality and image correlation for the camera panels.')
    report.table(('Local artifact', 'Use'), [
        ('iccp2017_reproduction_data.csv', 'Paper readings, local statistics and boundary labels for reproduced numerical graphics and tables.'),
        ('i12_summary.json', 'Aggregate and coverage metrics, full solver statuses, recorded CPU budgets and data provenance.'),
        ('i12_trials.jsonl.gz', 'Per-fit records, actual iteration counts and errors.'),
        ('i12_rice_estimates.npz', 'Stored measured transmission row estimates for both budget arms.'),
        ('scripts/build_iccp_report.py', 'Report source. Takes the supplied PDF and the local I12 trial directory.'),
    ], [213, 286])
    report.text('Attached PDF SHA-256', 'section')
    report.text(file_hash(data.paper), 'small')


def write_csv(path, data):
    fields = ('paper_item', 'study', 'method', 'modes', 'patterns', 'noise_sigma',
              'metric', 'paper_value', 'paper_reading', 'i12_value', 'i12_q25', 'i12_q75',
              'i12_finite', 'i12_attempted', 'i12_statuses')

    def comparison(record):
        stats = record.get('phase_nmse', {})
        return dict(i12_value=stats.get('median'), i12_q25=stats.get('q25'), i12_q75=stats.get('q75'),
                    i12_finite=record['finite'], i12_attempted=record['attempted'],
                    i12_statuses=json.dumps(record['statuses'], sort_keys=True))

    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for title, values, metric in [('Table 1', PAPER_TIMINGS, 'solve_wall_seconds'),
                                      ('Table 2', PAPER_PREPARATION, 'prepare_wall_seconds')]:
            for method, nums in values.items():
                for n, num in zip(DIMENSIONS, nums):
                    r = data.timing[n][method]
                    writer.writerow(dict(paper_item=title, study='gaussian_timing', method=LABELS[method],
                                         modes=n, patterns=12*n, noise_sigma=0, metric=metric,
                                         paper_value=num, paper_reading='printed table', i12_value=r[metric],
                                         i12_finite=int(r['finite']), i12_attempted=1,
                                         i12_statuses=json.dumps({r['status']: 1})))
        for method, values in PAPER_RATIO.items():
            for ratio, value in enumerate(values, 1):
                row = data.group(f'ratio_gaussian_{ratio:02}', method)
                writer.writerow(dict(paper_item='Figure 3', study=f'gaussian_{ratio}N', method=LABELS[method],
                                     modes=256, patterns=256*ratio, noise_sigma=0,
                                     metric='median_phase_nmse', paper_value=value,
                                     paper_reading=paper_qualifier(value), **comparison(row)))
        for number, regime, image in [(6, 'amplitude', 15), (11, 'phase', 33)]:
            for sigma, values in zip(NOISE_LEVELS, data.noise(image)):
                for method, value in zip(NOISE_METHODS, values):
                    row = data.group(f'noise_{regime}_{sigma:g}', method)
                    writer.writerow(dict(paper_item=f'Figure {number}', study=f'binary_{regime}', method=LABELS[method],
                                         modes=256, patterns=3072, noise_sigma=sigma, metric='median_phase_nmse',
                                         paper_value=value, paper_reading=paper_qualifier(value), **comparison(row)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/i12')
    parser.add_argument('--output', type=Path, default=RESULTS/'iccp2017_benchmark_reproduction.pdf')
    parser.add_argument('--csv', type=Path, default=RESULTS/'iccp2017_reproduction_data.csv')
    args = parser.parse_args()
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    data = Evidence(args.paper, args.run)
    report = Report(args.output)
    overview(report, data)
    timing_tables(report, data)
    settings_table(report)
    first_figure(report, data)
    simulation_figure(report, data, 2, 'gaussian', 6,
                      'The published montage depicts recognizable smileys across the nine shown variants. '
                      'I12 also reconstructs a clear shape with prVAMP, prGAMP and long-budget WF on this probe family. '
                      'At 1,024 modes, several I12 runs reach the time budget. PhaseLift and the random-anchor '
                      'PhaseMax implementation retain much larger coefficient errors than the clear source images suggest.')
    oversampling_figure(report, data)
    calibration_figure(report, data)
    amplitude_image_example(report, data)
    noise_figure(report, data, 6, 'amplitude',
                 'At σ = 1 the paper shows prVAMP and prSAMP at the visible baseline, compared with about '
                 '0.50 for PhaseMax and 0.64 for PhaseLift. I12 gives around 0.553 and 0.663 for prVAMP '
                 'and prSAMP. WF failed on every binary-amplitude case. I12 prGAMP’s median at this '
                 'noise level is about 4.4 × 10<super>69</super>, despite its vectors staying finite. '
                 'That behavior illustrates the nonzero-mean probe difficulty also visible in Figure 7.')
    simulation_figure(report, data, 7, 'amplitude', 16,
                      'The paper’s montage shows prVAMP and prSAMP recovering the object clearly, with '
                      'no visible recovery from original WF and a noisy prGAMP panel. I12 prVAMP also '
                      'recovers the object (median NMSE about 7.9 × 10<super>−5</super>). '
                      'Both I12 WF budgets fail, and prGAMP’s representative estimate has extremely large error. '
                      'The I12 prSAMP image is degraded under its active limits.')
    optical_setup(report, data, 8)
    amplitude_montage(report, data)
    simulation_figure(report, data, 10, 'phase', 32,
                      'The paper shows cleaner recoveries after switching to signed probes. '
                      'I12 prGAMP and prSAMP improve substantially over the binary amplitude image case. '
                      'The signed-probe observation also admits a conjugate coefficient solution. '
                      'I12 retains both the paper-style phase-only score and a separate conjugation-aware score.')
    noise_figure(report, data, 11, 'phase',
                 'At σ = 1, the original first six bars sit at the plot baseline. Their I12 counterparts '
                 'include prVAMP 0.00083, prGAMP 0.00136 and GS-100 0.00089. PhaseMax is around 0.94 '
                 'and PhaseLift about 0.14 in I12, with different backends and initialization. At σ = 15 '
                 'the paper shows growing errors across techniques. The separate I12 curves preserve '
                 'both the scale of that degradation and trial-level uncertainty.')
    optical_setup(report, data, 12, phase=True)
    phase_montage(report, data)
    transmission_montage(report, data)
    rice_table(report, data, 'nominal')
    rice_table(report, data, 'matched')
    supplemental_analysis(report, data)
    evaluation_notes(report, data)
    sources_page(report, data)
    report.build()
    write_csv(args.csv, data)
    print(args.output)
    print(args.csv)


if __name__ == '__main__':
    main()
