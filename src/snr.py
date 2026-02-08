''' 
SNR calculation 
- Uses bottom 25% of signal as baseline noise -> from here - https://www.jssm.org/mobile/ShowFigure.php?jid=jssm-09-620.xml&FigureId=fig001
- Calculates mean and standard deviation of baseline
- Visualise all snr 
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calc_snr(signal, detection_results, visualize=False, save_path=None, sampling_rate=2000):
    segments = detection_results['segments']
    n_samples = len(signal)
    # directly from speech detection file
    smooth_envelope = detection_results.get('smooth_envelope', None)
    
    if smooth_envelope is None:
        print('Warning: smooth_envelope not found')
        from speech_detection import SpeechActivityDetector
        detector = SpeechActivityDetector()
        clean_signal = detector._preprocess(signal)
        rms_env = detector.compute_rms_envelope_filtered(clean_signal)
        smooth_envelope = detector.smooth_envelope(rms_env)
    
    # find baseline rms as noise 
    baseline_rms = detection_results.get('baseline_value', None)
    if baseline_rms is None:
        baseline_rms = np.percentile(smooth_envelope, 25)
    
    rms_threshold = np.percentile(smooth_envelope, 25) 
    quiet_rms_values = smooth_envelope[smooth_envelope <= rms_threshold]
    
    # snr = 10 log (power of signal/power noise)
    baseline_noise_power = np.mean(quiet_rms_values**2)
    baseline_noise_power_std = np.std(quiet_rms_values**2)  
    
    
    print(f'Baseline RMS (mean): {np.mean(rms_threshold):.4e}')
    print(f'Baseline RMS (std):  {np.std(rms_threshold):.4e}')
    print(f'Baseline power (mean): {baseline_noise_power:.4e}')
    print(f'Baseline power (std): {baseline_noise_power_std:.4e}')
    
    # Calculate SNR for each segment
    segment_snrs = []
    segment_powers = []
    segment_rms_values = []
    
    for onset, offset in segments:
        safe_onset = max(0, onset)
        safe_offset = min(n_samples, offset)
        
        word_samples = signal[safe_onset:safe_offset]
        word_power = np.mean(np.square(word_samples))
        word_rms = np.sqrt(word_power)
        
        if baseline_noise_power > 0:
            snr = 10 * np.log10(word_power / baseline_noise_power)
        else:
            snr = 0.0
        
        segment_powers.append(word_power)
        segment_rms_values.append(word_rms)
        segment_snrs.append(snr)
    
    # Calculate SNR statistics
    average_snr = np.mean(segment_snrs) if len(segment_snrs) > 0 else 0.0
    snr_std = np.std(segment_snrs) if len(segment_snrs) > 0 else 0.0
    
    results = {
        'noise_power': baseline_noise_power,
        'noise_std': baseline_noise_power_std,
        'noise_rms_mean': np.mean(rms_threshold),
        'noise_rms_std': np.std(rms_threshold),  
        'noise_threshold': rms_threshold,
        'segment_powers': segment_powers,
        'segment_rms_values': segment_rms_values,
        'segment_snrs': segment_snrs,
        'average_snr': float(average_snr),
        'snr_std': float(snr_std),
        'method': 'rms_baseline'
    }
    
    if visualize:
        visualize_snr_rms(signal, smooth_envelope, results, segments, save_path, sampling_rate)
    
    return results

def visualize_snr_rms(signal, rms_env, snr_results, segments, save_path=None, sampling_rate=2000):

    from speech_detection import SpeechActivityDetector
    import config
    
    time = np.arange(len(signal)) / sampling_rate
    hop = getattr(config, 'HOP', 50)  
    rms_time = np.arange(len(rms_env)) * hop / 1000.0
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    fig.patch.set_facecolor('white')
    
    # Get metrics
    baseline_rms = snr_results['noise_rms_mean']
    baseline_rms_std = snr_results['noise_rms_std']
    rms_threshold = snr_results['noise_threshold']
    segment_snrs = snr_results['segment_snrs']
    mean_snr = snr_results['average_snr']
    snr_std = snr_results['snr_std']
    
    # Color map for segments
    cmap = plt.get_cmap('tab20')
    palette = [cmap(i) for i in range(cmap.N)]
    
    axes[0].plot(time, signal, color='#333333', lw=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude', fontweight='bold')
    axes[0].set_title('A) Signal with detected segments', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Highlight segments
    for i, (onset, offset) in enumerate(segments):
        if onset < len(time):
            color = palette[i % len(palette)]
            axes[0].axvspan(time[onset], time[min(offset, len(time)-1)],facecolor=color, alpha=0.3, edgecolor='black', linewidth=1.5,label=f'Seg {i+1}' if i < 5 else None)
    
    if len(segments) <= 5:
        axes[0].legend(loc='upper right', fontsize=8)
    
    axes[0].set_xlabel("Time (s)", fontweight='bold')
    

    axes[1].plot(rms_time, rms_env, color='#4dac26', lw=1.5, label='Filtered RMS envelope')
    axes[1].axhline(rms_threshold, color='red', ls='--', lw=2, label=f'RMS Threshold \n= {rms_threshold:.4e}', alpha=0.8, zorder=10)
    
    # filling in bwtn mean + std 
    axes[1].fill_between([rms_time[0], rms_time[-1]], baseline_rms - baseline_rms_std, baseline_rms + baseline_rms_std,color='orange', alpha=0.1)
    
    # colour coding everything below baseline
    quiet_mask = rms_env <= rms_threshold
    axes[1].fill_between(rms_time, 0, rms_env, where=quiet_mask,color='green', alpha=0.2, label='Baseline')
    
    axes[1].set_ylabel('RMS amplitude', fontweight='bold')
    axes[1].set_xlabel('Time (s)', fontweight='bold')
    axes[1].set_title('B) Filtered RMS envelope with baseline noise', fontweight='bold', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=7)
    axes[1].grid(True, alpha=0.3)
    
    segment_indices = np.arange(1, len(segments) + 1)
    
    bars = axes[2].bar(segment_indices, segment_snrs, width=0.7,color=palette[:len(segments)], edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Add SNR values on top of bars
    for idx, (seg_idx, snr_val) in enumerate(zip(segment_indices, segment_snrs)):
        axes[2].text(seg_idx, snr_val, f'{snr_val:.1f}dB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Mean SNR line
    axes[2].axhline(mean_snr, color='red', ls='--', lw=2,label=f'Mean SNR = {mean_snr:.2f} ± {snr_std:.2f} dB', alpha=0.8)
    
    
    axes[2].set_xlabel('Segment number', fontweight='bold')
    axes[2].set_ylabel('SNR (dB)', fontweight='bold')
    axes[2].set_title('C) Signal to Noise Ratio per segment', fontweight='bold', fontsize=12)
    axes[2].set_xticks(segment_indices)
    axes[2].set_xlim(0.5, len(segments) + 0.5)
    axes[2].legend(loc='lower left', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.close(fig)
    
    return fig