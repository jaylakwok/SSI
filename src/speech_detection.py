import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import config 
import matplotlib.pyplot as plt
from matplotlib import colors
import os

class SpeechActivityDetector:
    def __init__(self, method='spc'):
        self.method = method
        self.fs = config.SAMPLING_RATE
        self.ignore_start_ms = config.IGNORE_START_MS
        self.ignore_end_ms = getattr(config, 'IGNORE_END_MS', 0.0)
        self.threshold_decay = config.THRESHOLD_DECAY
        self.bp_low = config.BP_LOW
        self.bp_high = config.BP_HIGH
        self.bp_order = config.BP_ORDER
        self.notch_freqs = config.NOTCH_FREQS
        self.notch_q = config.NOTCH_Q
        
        # convert ms to samples
        self.window_samples = int(config.WINDOW * self.fs / 1000)
        self.hop_samples = int(config.HOP * self.fs / 1000)


    def _preprocess(self, raw_signal):
        #same as preprocess2 script
        signal = raw_signal - np.mean(raw_signal)                     

        for f0 in self.notch_freqs:                                   
            b, a = iirnotch(w0=f0, Q=self.notch_q, fs=self.fs)
            signal = filtfilt(b, a, signal)

        b, a = butter(self.bp_order,[self.bp_low, self.bp_high],btype='bandpass', fs=self.fs)
        signal = filtfilt(b, a, signal)
        return signal

    def compute_rms_envelope(self, raw_signal):
        #part 2 of plot 2 RAW though
        n_samples_r = len(raw_signal)
        if n_samples_r < self.window_samples:
            return np.array([0.0])
            
        n_windows_r = (n_samples_r - self.window_samples) // self.hop_samples + 1
        rms_env_r = np.zeros(n_windows_r)
        
        for i in range(n_windows_r):
            start = i * self.hop_samples
            end = start + self.window_samples
            rms_env_r[i] = np.sqrt(np.mean(raw_signal[start:end]**2))
            
        return rms_env_r

    def compute_rms_envelope_filtered(self, signal):
        # convert to rms envelope
        n_samples = len(signal)
        if n_samples < self.window_samples:
            return np.array([0.0])
            
        n_windows = (n_samples - self.window_samples) // self.hop_samples + 1
        rms_env = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * self.hop_samples
            end = start + self.window_samples
            rms_env[i] = np.sqrt(np.mean(signal[start:end]**2)) 
        return rms_env

    def smooth_envelope(self, envelope):
        # part 3 of plot 
        if len(envelope) < 2:
            return envelope
            
        frame_rate = 2000.0 / config.HOP          
        nyquist = frame_rate / 2                   

        cutoff = 10.0
        
        b, a = butter(self.bp_order, cutoff / nyquist, btype='low')
        smoothed = filtfilt(b, a, envelope)
        
        return np.maximum(smoothed, 0)          

    def _get_initial_threshold(self, metric):
        if self.method == 'spc':
            if len(metric) == 0:
                return 1e-10
            mu_b = np.percentile(metric, 10)
            baseline = metric[metric <= mu_b]
            sigma_b = np.std(baseline) if len(baseline) > 0 else 0
            k = 5.5 # change this value accordingly to visual plots
            thresh = mu_b + (k * sigma_b)
        else:
            mu_b = np.mean(metric)
            sigma_b = np.std(metric)
            k = 4.5
            thresh = mu_b + (k * sigma_b)
        return max(thresh, 0.036)

    def _get_baseline_value(self, env):
        # 25th percentile like paper
        mu_b = np.percentile(env, 25)
        return mu_b

    def detect_activity(self, envelope):
        if len(envelope) < 2:
            return [], 0.0

        # ignore early noise and end needs adjusting in config file
        env = envelope.copy()
        ignore_windows = int(self.ignore_start_ms / config.HOP)
        if ignore_windows < len(env):
            env[:ignore_windows] = 0.0

        ignore_end_windows = int(self.ignore_end_ms / config.HOP)
        if ignore_end_windows > 0 and ignore_end_windows < len(env):
            env[-ignore_end_windows:] = 0.0

        # get baseline and threshold
        baseline_value = self._get_baseline_value(env)
        current_threshold = self._get_initial_threshold(env)
        decay_floor = current_threshold * config.THRESHOLD_MIN_RATIO

        # loop to test the threshold
        active_mask = env > current_threshold
        for _ in range(100):
            if np.any(active_mask):
                break
            current_threshold *= self.threshold_decay
            if current_threshold < decay_floor:
                break
            active_mask = env > current_threshold

        if not np.any(active_mask):
            return [], current_threshold

        min_peak_distance_ms = getattr(config, 'MIN_PEAK_DISTANCE', 200)
        min_peak_distance_frames = max(1, int(min_peak_distance_ms / config.HOP))
        
        min_prominence = getattr(config, 'MIN_PEAK_PROMINENCE', 2.0)
        
        peaks, properties = find_peaks(env, height=current_threshold,distance=min_peak_distance_frames,prominence=baseline_value * min_prominence,width=3)
        
        if len(peaks) == 0:
            print('No peaks found')
            return [], current_threshold
        
        print(f'Found {len(peaks)} peaks with distance>={min_peak_distance_ms}ms, prominence>={min_prominence}x baseline')
        print(f'Peak locations (frames): {peaks}')
        
        # Remove peaks that are too close in VALUE so doesnt recognise more
        if len(peaks) > 1:
            filtered_peaks = [peaks[0]]
            
            for i in range(1, len(peaks)):
                current_peak = peaks[i]
                prev_peak = filtered_peaks[-1]
                
                # Check the valley between them
                valley_idx = prev_peak + np.argmin(env[prev_peak:current_peak])
                valley_depth = min(env[prev_peak], env[current_peak]) - env[valley_idx]
                
                valley_threshold = baseline_value 
                
                if valley_depth > valley_threshold:
                    filtered_peaks.append(current_peak)
                else:
                    if env[current_peak] > env[prev_peak]:
                        filtered_peaks[-1] = current_peak
            
            peaks = np.array(filtered_peaks)
            print(f"  After valley filtering: {len(peaks)} peaks")
        
        # fixed window approach
        segment_before_peak_ms = getattr(config, 'SEGMENT_BEFORE_PEAK', 1200)  # ms before peak
        segment_after_peak_ms = getattr(config, 'SEGMENT_AFTER_PEAK', 1200)    # ms after peak
        
        before_frames = int(segment_before_peak_ms / config.HOP)
        after_frames = int(segment_after_peak_ms / config.HOP)
        
        print(f'Using fixed window: {segment_before_peak_ms}ms before peak, {segment_after_peak_ms}ms after peak')
        
        segments = []
        
        for i, peak_idx in enumerate(peaks):
            # Fixed window around peak
            onset = max(0, peak_idx - before_frames)
            offset = min(len(env) - 1, peak_idx + after_frames)
            
            segments.append((onset, offset))
            duration_ms = (offset - onset) * config.HOP
            peak_height = env[peak_idx]
            
            print(f'Segment {i+1}: peak at {peak_idx:} (h={peak_height:}) &'
                f'segment [{onset:}, {offset:}], dur={duration_ms:}ms')
        
        # check for overlapping segments and merge if needed
        if len(segments) > 1:
            merged = [segments[0]]
            for onset, offset in segments[1:]:
                prev_onset, prev_offset = merged[-1]
                if onset <= prev_offset:  # Overlapping
                    merged[-1] = (prev_onset, max(offset, prev_offset))
                    print(f'Merged overlapping segments')
                else:
                    merged.append((onset, offset))
            segments = merged
        
        print(f'Final: {len(segments)} segments from {len(peaks)} peaks')
        
        return segments, current_threshold

    def convert_to_sample_space(self, segments):
        if not segments:
            return []
        sample_segments = []
        for onset_env, offset_env in segments:
            t_start = max(0, onset_env * self.hop_samples)
            t_end = offset_env * self.hop_samples 
            sample_segments.append((t_start, t_end))
        return sample_segments
    
    def detect(self, signal, return_metadata=False):
        # preprocess
        clean_signal = self._preprocess(signal)
        print(f"'clean_signal': {np.shape(clean_signal)}")

        # 2c
        rms_env_r = self.compute_rms_envelope(signal)
        print(f'raw rms: {np.shape(rms_env_r)}')

        rms_env = self.compute_rms_envelope_filtered(clean_signal)
        print(f'filtered rms: {np.shape(rms_env)}')
        
        # smooth envelope plot 2d
        smooth_env = self.smooth_envelope(rms_env)
        print(f'smooth env: {np.shape(smooth_env)}')
        
        segments_env, final_threshold = self.detect_activity(smooth_env)
        
        # convert to sample indices
        segments = self.convert_to_sample_space(segments_env)
        print(f'Number of segments: {len(segments)}')
        for i, (onset, offset) in enumerate(segments[:11]):  # print first 5
            print(f'Segment {i+1}: samples [{onset}, {offset}], duration {(offset/self.fs*1000)-(onset/self.fs*1000):}ms')
        
        n_samples = len(signal)
        labels = np.zeros(n_samples, dtype=int)
        for i, (onset, offset) in enumerate(segments):
            labels[onset:min(offset, n_samples)] = i + 1

        results = {
            'segments': segments, 
            'labels': labels,
            'final_threshold': final_threshold,
            'n_segments': len(segments),
            'clean_signal': clean_signal,
            'baseline_value': self._get_baseline_value(smooth_env),
        }
        
        if return_metadata:
            results['clean_signal'] = clean_signal
            results['rms_envelope_r'] = rms_env_r
            results['rms_envelope'] = rms_env
            results['smooth_envelope'] = smooth_env
            results['derivative'] = np.abs(np.diff(smooth_env))
            # store peak locations
            peak_height = final_threshold
            peaks, _ = find_peaks(smooth_env,height=peak_height,distance=max(1, int(getattr(config, 'MIN_PEAK_DISTANCE', 300) / config.HOP)),prominence=self._get_baseline_value(smooth_env) * 0.5)
            results['peaks'] = peaks
        return results

    def apply_segments_to_channel(self, signal, reference_segments):
        """Apply segment timings from reference channel to another channel."""
        n_samples = len(signal)
        labels = np.zeros(n_samples, dtype=int)
        
        for i, (onset, offset) in enumerate(reference_segments):
            onset = max(0, onset)
            offset = min(offset, n_samples)
            labels[onset:offset] = i + 1
            
        return {
            'segments': reference_segments,
            'labels': labels,
            'n_segments': len(reference_segments)
        }

    def visualize_detection(self, signal, results, save_path=None, show=True):
        time = np.arange(len(signal)) / self.fs
        segments = results['segments']

        if 'rms_envelope_r' not in results:
            clean = results.get('clean_signal', self._preprocess(signal))
            rms_r = self.compute_rms_envelope(signal)
            rms = self.compute_rms_envelope_filtered(clean)
            smooth = self.smooth_envelope(rms)
            results['clean_signal'] = clean
            results['rms_envelope_r'] = rms_r
            results['smooth_envelope'] = smooth
            results['derivative'] = np.abs(np.diff(smooth))

        env_time = np.arange(len(results['rms_envelope_r'])) * config.HOP / 1000.0
        deriv_time = np.arange(len(results['derivative'])) * config.HOP / 1000.0
        
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i) for i in range(cmap.N)]

        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        fig.patch.set_facecolor('white')

        # plot 2a: raw emg 
        axes[0].plot(time, signal, color='#333333', lw=0.5)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"A) sEMG ", fontweight='bold')
        for i, (on, off) in enumerate(segments):
            if on < len(time):
                color = palette[i % len(palette)]
                axes[0].axvspan(time[on], time[min(off, len(time)-1)],facecolor=color, alpha=0.4,edgecolor='black', linewidth=1,label=f'S{i+1}' if i < 15 else None)

        # plot 2b:  filtered sEMG
        axes[1].plot(time, results['clean_signal'], color='#444444', lw = 0.5)
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title("B) Filtered sEMG", fontweight = 'bold')
        for i, (on, off) in enumerate(segments):
            color = palette[i % len(palette)]
            axes[1].axvspan(time[on], time[min(off, len(time)-1)],facecolor=color, alpha=0.4,edgecolor='black', linewidth=1)
        

        # plot 2c: rms
        axes[2].plot(env_time, results['rms_envelope_r'], color='#2166ac', lw=1)
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('C) Raw sEMG RMS', fontweight='bold')
        for i, (on, off) in enumerate(segments):
            color = palette[i % len(palette)]
            axes[2].axvspan(time[on], time[min(off, len(time)-1)],facecolor=color, alpha=0.4,edgecolor='black', linewidth=1)

        # plot 2d: smooth envelope with PEAKS marked -> filtered with 10 HP
        axes[3].plot(env_time, results['smooth_envelope'], color='#4dac26', lw=1.5)
        axes[3].set_ylabel('Amplitude')
        axes[3].set_title('D) Smoothed filtered RMS', fontweight='bold')
        
        # detecetd peaks as red circle
        if 'peaks' in results and len(results['peaks']) > 0:
            peak_times = results['peaks'] * config.HOP / 1000.0
            peak_values = results['smooth_envelope'][results['peaks']]
            axes[3].scatter(peak_times, peak_values, c='red', s=10, marker='o', linewidths=2, zorder=5, label=f'{len(results["peaks"])} peaks')
        
        axes[3].axhline(results['final_threshold'], color='red', ls='--', lw=1.5,label='Threshold', alpha=0.7)
        
        if 'baseline_value' in results:
            baseline_val = results['baseline_value']
            axes[3].axhline(baseline_val, color='green', ls=':', lw=2, label=f'Baseline', alpha=0.7)
        
        axes[3].legend(loc='upper left', fontsize=8)
        for i, (on, off) in enumerate(segments):
            color = palette[i % len(palette)]
            axes[3].axvspan(time[on], time[min(off, len(time)-1)],facecolor=color, alpha=0.4,  edgecolor='black', linewidth=1)

        # plot 2e: derivative
        axes[4].plot(deriv_time, results['derivative'], color='#b2182b', lw=1)
        axes[4].set_ylabel('Amplitude')
        axes[4].set_xlabel('Time (s)')
        axes[4].set_title('E) Rectified Derivative of RMS', fontweight='bold')
        for i, (on, off) in enumerate(segments):
            color = palette[i % len(palette)]
            axes[4].axvspan(time[on], time[min(off, len(time)-1)],facecolor=color, alpha=0.4,edgecolor='black', linewidth=1)

        # legend
        if segments and len(segments) <= 15:
            axes[0].legend(loc='upper left', fontsize=7, ncol=min(len(segments), 4))

        for ax in axes:
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig