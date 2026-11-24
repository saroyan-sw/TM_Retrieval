"""Amplitude prediction metrics with explicit denominators and full-row coverage."""
import numpy as np


def evaluate(split, estimates, statuses):
    rows = []
    for j, detector in enumerate(split.detector_ids):
        record = dict(detector_id=int(detector), solver_status=statuses[j], evaluated=False)
        if estimates[j] is not None and np.isfinite(estimates[j]).all():
            x = np.asarray(split.X, dtype=float if not np.iscomplexobj(split.X) else complex)
            a = estimates[j]
            predicted = np.abs(x @ a.real + 1j * (x @ a.imag)) if not np.iscomplexobj(x) else np.abs(x @ a)
            mask = split.valid[:, j]
            y = split.amplitudes[:, j][mask]
            pred = predicted[mask]
            if len(y) and np.isfinite(pred).all():
                sse = float(np.sum((pred - y)**2))
                energy = float(y @ y)
                record.update(evaluated=True, valid_count=len(y), squared_error=sse,
                              amplitude_energy=energy, amplitude_mse=sse / len(y),
                              relative_amplitude_residual=float(np.sqrt(sse / energy)) if energy > 0 else None)
        rows.append(record)
    good = [r for r in rows if r['evaluated']]
    count = sum(r['valid_count'] for r in good)
    sse = sum(r['squared_error'] for r in good)
    energy = sum(r['amplitude_energy'] for r in good)
    return dict(attempted_rows=len(rows), evaluated_rows=len(good), coverage=len(good)/len(rows),
                valid_count=count, squared_error=sse, amplitude_energy=energy,
                amplitude_mse=sse/count if count else None,
                relative_amplitude_residual=float(np.sqrt(sse/energy)) if energy else None,
                aggregation='pooled valid observations among evaluated rows; coverage uses all attempted rows', rows=rows)
