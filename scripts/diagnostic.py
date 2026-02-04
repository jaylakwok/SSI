import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
import config
from speech_detection import SpeechActivityDetector


def diagnose_file(filepath, detector, ax_row=None):
    """
    Run the detector's actual internal pipeline on one file and return
    the stats + the threshold it actually picked.  Optionally plots into
    a pre-made axes row [signal, envelope, derivative].
    """
    raw = np.load(filepath).squeeze().astype(float)

    # --- mirror detect() exactly so we see the same numbers ---
    clean = detector._preprocess(raw)
    rms_env = detector.compute_rms_envelope(clean)
    smooth_env = detector.smooth_envelope(rms_env)

    # derivative (same as detect_activity does internally)
    deriv_abs = np.abs(np.diff(smooth_env))

    # zero out the ignore window (same as detect_activity)
    ignore_windows = int(detector.ignore_start_ms / config.HOP)
    if ignore_windows < len(deriv_abs):
        deriv_abs[:ignore_windows] = 0.0

    # the threshold the detector actually picks for THIS file
    initial_thresh = detector._get_initial_threshold(deriv_abs)

    # run the full decay loop the same way detect_activity does
    current_threshold = initial_thresh
    for _ in range(100):
        hits = np.where(deriv_abs > current_threshold)[0]
        if len(hits) > 0:
            break
        current_threshold *= detector.threshold_decay
        if current_threshold < detector.absolute_min:
            break

    # also grab the full detection result for segment overlay
    results = detector.detect(raw, return_metadata=True)

    stats = {
        'file': Path(filepath).name,
        'initial_threshold': initial_thresh,
        'final_threshold': current_threshold,
        'decay_steps': 0 if current_threshold == initial_thresh else
                       int(np.round(np.log(current_threshold / initial_thresh)
                                    / np.log(detector.threshold_decay))),
        'n_segments': results['n_segments'],
        'deriv_max': float(np.max(deriv_abs)),
        'deriv_mean': float(np.mean(deriv_abs)),
        'deriv_std': float(np.std(deriv_abs)),
        'p10': float(np.percentile(deriv_abs, 10)),
        'p95': float(np.percentile(deriv_abs, 95)),
        'p99': float(np.percentile(deriv_abs, 99)),
        'segments': results['segments'],
        'smooth_env': smooth_env,
        'deriv_abs': deriv_abs,
        'clean': clean,
    }

    # --- optional plotting into caller-provided axes ---
    if ax_row is not None:
        time = np.arange(len(clean)) / config.SAMPLING_RATE
        env_time = np.arange(len(smooth_env)) * config.HOP / 1000.0
        deriv_time = np.arange(len(deriv_abs)) * config.HOP / 1000.0
        segments = results['segments']

        # A) filtered signal + segment shading
        ax_row[0].plot(time, clean, color='#333333', lw=0.4)
        for on, off in segments:
            ax_row[0].axvspan(time[on], time[min(off, len(time)-1)],
                              color='#e6550d', alpha=0.35, edgecolor='none')
        ax_row[0].set_ylabel('Signal', fontsize=8)
        ax_row[0].set_title(Path(filepath).name, fontsize=8, loc='left')
        ax_row[0].tick_params(labelsize=6)

        # B) smoothed envelope
        ax_row[1].plot(env_time, smooth_env, color='#2166ac', lw=0.8)
        for on, off in segments:
            ax_row[1].axvspan(time[on], time[min(off, len(time)-1)],
                              color='#e6550d', alpha=0.35, edgecolor='none')
        ax_row[1].set_ylabel('Envelope', fontsize=8)
        ax_row[1].tick_params(labelsize=6)

        # C) derivative + threshold lines
        ax_row[2].plot(deriv_time, deriv_abs, color='#756bb1', lw=0.8)
        ax_row[2].axhline(initial_thresh, color='red', ls='--', lw=1,
                          label=f'Initial: {initial_thresh:.6f}')
        if current_threshold != initial_thresh:
            ax_row[2].axhline(current_threshold, color='orange', ls='-', lw=1,
                              label=f'Final (after {stats["decay_steps"]} decays): {current_threshold:.6f}')
        for on, off in segments:
            ax_row[2].axvspan(time[on], time[min(off, len(time)-1)],
                              color='#e6550d', alpha=0.35, edgecolor='none')
        ax_row[2].set_ylabel('|Derivative|', fontsize=8)
        ax_row[2].set_xlabel('Time (s)', fontsize=8)
        ax_row[2].legend(fontsize=6, loc='upper right')
        ax_row[2].tick_params(labelsize=6)

    return stats


def run_batch_diagnostic(file_list, max_plots=6):
    """
    Run diagnostics on every file, print a summary table, and produce
    a compact multi-file plot for the first `max_plots` files.
    """
    detector = SpeechActivityDetector(method='spc')

    # --- collect stats for all files ---
    all_stats = []
    for i, pattern in enumerate(file_list):
        filepath = project_root / config.RAW_DATA_DIR / pattern
        if not filepath.exists():
            continue
        print(f"[{i+1}/{len(file_list)}] {pattern} ... ", end='', flush=True)
        stats = diagnose_file(filepath, detector)
        all_stats.append(stats)
        decayed = f" ({stats['decay_steps']} decays)" if stats['decay_steps'] > 0 else ""
        print(f"thresh={stats['final_threshold']:.6f}{decayed}  segs={stats['n_segments']}")

    # --- summary table ---
    print(f"\n{'='*90}")
    print(f"{'File':<45} {'Initial':>10} {'Final':>10} {'Decays':>7} {'Segs':>5}")
    print(f"{'-'*90}")
    for s in all_stats:
        print(f"{s['file']:<45} {s['initial_threshold']:>10.6f} "
              f"{s['final_threshold']:>10.6f} {s['decay_steps']:>7} {s['n_segments']:>5}")

    # highlight any that needed decay
    decayed_files = [s for s in all_stats if s['decay_steps'] > 0]
    if decayed_files:
        print(f"\n⚠  {len(decayed_files)} file(s) needed threshold decay:")
        for s in decayed_files:
            print(f"   {s['file']}  — {s['decay_steps']} steps, "
                  f"{s['initial_threshold']:.6f} → {s['final_threshold']:.6f}")

    # --- plot the first max_plots files ---
    n_plot = min(max_plots, len(all_stats))
    fig, axes = plt.subplots(n_plot, 3, figsize=(18, 3.5 * n_plot),
                             sharex=False)
    if n_plot == 1:
        axes = [axes]   # make it consistently indexable

    for idx in range(n_plot):
        filepath = project_root / config.RAW_DATA_DIR / file_list[idx]
        diagnose_file(filepath, detector, ax_row=axes[idx])

    plt.tight_layout()
    reports_dir = project_root / config.REPORTS_DIR
    reports_dir.mkdir(exist_ok=True, parents=True)
    save_path = reports_dir / "threshold_diagnostic_batch.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to: {save_path}")
    plt.show()

    return all_stats


if __name__ == "__main__":
    # Run on all files in config
    run_batch_diagnostic(config.FILE_PATTERNS, max_plots=6)