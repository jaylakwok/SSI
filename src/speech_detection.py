import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import config 
import matplotlib.pyplot as plt
from matplotlib import colors

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
        self.onset_buffer_samples = int(config.ONSET * self.fs / 1000)
        self.offset_buffer_samples = int(config.OFFSET * self.fs / 1000)
        

    def _preprocess(self, raw_signal):
        #same as preprocess2 script
        signal = raw_signal - np.mean(raw_signal)                     

        for f0 in self.notch_freqs:                                   
            b, a = iirnotch(w0=f0, Q=self.notch_q, fs=self.fs)
            signal = filtfilt(b, a, signal)

        b, a = butter(self.bp_order,[self.bp_low, self.bp_high],btype='bandpass', fs=self.fs)
        signal = filtfilt(b, a, signal)
        return signal

    def compute_rms_envelope(self, signal):
        #part 2 of plot 2
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
        # part 3 of plot 2
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
        return max(thresh, 0.021) # change this value accordingly to visual plots -> normally need to make small if huge segment to invalidate

    def detect_activity(self, envelope):
        if len(envelope) < 2:
            return [], 0.0

        # ignore early noise / motion artefacts
        env = envelope.copy()
        ignore_windows = int(self.ignore_start_ms / config.HOP)
        if ignore_windows < len(env):
            env[:ignore_windows] = 0.0

        ignore_end_windows = int(self.ignore_end_ms / config.HOP)
        if ignore_end_windows > 0 and ignore_end_windows < len(env):
            env[-ignore_end_windows:] = 0.0

        # adaptive
        current_threshold = self._get_initial_threshold(env)
        decay_floor = current_threshold * config.THRESHOLD_MIN_RATIO

        # loop to test the threshold -> lower if nothing found until something is found 
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

        # Finite state machine -> walk the envelope frame by frame
        # INACTIVE to ACTIVE  = envelope crosses above threshold
        # ACTIVE to INACTIVE  = envelope stays below threshold for min_duration frames -> in config
        min_duration_frames = int(config.MIN_DURATION / config.HOP)

        segments = []
        state = 'INACTIVE'
        onset = 0
        below_count = 0 # number consecutive frames below threshold

        end_limit = len(env) - ignore_end_windows if ignore_end_windows > 0 else len(env)
        end_limit = max(end_limit, ignore_windows)
        for i in range(ignore_windows, end_limit):
            if state == 'INACTIVE':
                if env[i] > current_threshold:
                    state = 'ACTIVE'
                    onset = i
                    below_count = 0
            else: # ACTIVE mode
                if env[i] <= current_threshold:
                    below_count += 1
                    if below_count >= min_duration_frames:
                        segments.append((onset, i - below_count))
                        state = 'INACTIVE'
                else:
                    below_count = 0  # reset: still active

        if state == 'ACTIVE':
            segments.append((onset, end_limit - 1))

        # join segments that are separated by a small gap -> same activation 
        merge_gap_frames = int(config.MERGE_GAP / config.HOP) # change the numbers based on visual inspection in config
        if len(segments) > 1:
            merged = [segments[0]]
            for onset_next, offset_next in segments[1:]:
                prev_onset, prev_offset = merged[-1]
                gap = onset_next - prev_offset
                if gap <= merge_gap_frames:
                    merged[-1] = (prev_onset, offset_next)   
                else:
                    merged.append((onset_next, offset_next))
            segments = merged

        return segments, current_threshold

    def convert_to_sample_space(self, segments):
        if not segments:
            return []
        sample_segments = []
        for onset_env, offset_env in segments:
            t_start = max(0, onset_env * self.hop_samples - self.onset_buffer_samples)
            t_end = offset_env * self.hop_samples + self.offset_buffer_samples
            sample_segments.append((t_start, t_end))
        return sample_segments
    
    def detect(self, signal, return_metadata=False):
        # preprocess
        clean_signal = self._preprocess(signal)
        print(f"'clean_signal': {np.shape(clean_signal)}")

        # plot rms on filtered (plot 2b)
        rms_env = self.compute_rms_envelope(clean_signal)
        print(f"'rms env': {np.shape(rms_env)}")
        
        # smooth envelope plot 2c
        smooth_env = self.smooth_envelope(rms_env)
        print(f"'smooth env': {np.shape(smooth_env)}")
        
        # derivative plot 2d
        segments_env, final_threshold = self.detect_activity(smooth_env)
        
        # convert to sample indices
        segments = self.convert_to_sample_space(segments_env)
        print(f"'segments': {np.shape(segments)}")
        
        n_samples = len(signal)
        labels = np.zeros(n_samples, dtype=int)
        for onset, offset in segments:
            labels[onset:min(offset, n_samples)] = 1

        results = {
            'segments': segments, 
            'labels': labels,
            'final_threshold': final_threshold,
            'n_segments': len(segments),
            'clean_signal': clean_signal,
        }
        
        if return_metadata:
            results['clean_signal'] = clean_signal
            results['rms_envelope'] = rms_env
            results['smooth_envelope'] = smooth_env
            results['derivative'] = np.abs(np.diff(smooth_env))
            
        return results

    def visualize_detection(self, signal, results, save_path=None, show=True):
        time = np.arange(len(signal)) / self.fs
        segments = results['segments']

        if 'rms_envelope' not in results:
            clean = results.get('clean_signal', self._preprocess(signal))
            rms = self.compute_rms_envelope(clean)
            smooth = self.smooth_envelope(rms)
            results['clean_signal'] = clean
            results['rms_envelope'] = rms
            results['smooth_envelope'] = smooth
            results['derivative'] = np.abs(np.diff(smooth))

        env_time = np.arange(len(results['rms_envelope'])) * config.HOP / 1000.0
        deriv_time = np.arange(len(results['derivative'])) * config.HOP / 1000.0
        
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i) for i in range(cmap.N)]

        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor('white')

        # plot 2a: raw emg 
        axes[0].plot(time, signal, color='#333333', lw=0.5)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("A) sEMG", fontweight='bold')
        for i, (on, off) in enumerate(segments):
            if on < len(time):
                axes[0].axvspan(time[on], time[min(off, len(time)-1)],facecolor=palette[i % len(palette)], alpha=0.35,edgecolor='none',label=f'Seg {i+1}' if i < len(palette) else None)

        # plot 2b: 
        axes[1].plot(time, results['clean_signal'], color='#444444', lw = 0.5)
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title("B) Filtered sEMG ", fontweight = 'bold')
        for i, (on, off) in enumerate(segments):
            axes[1].axvspan(time[on], time[min(off, len(time)-1)],facecolor=palette[i % len(palette)], alpha=0.35,edgecolor='none')
        

        # plot 2c: rms
        axes[2].plot(env_time, results['rms_envelope'], color='#2166ac', lw=1)
        axes[2].set_ylabel("Amplitude")
        axes[2].set_title("C) sEMG RMS", fontweight='bold')
        for i, (on, off) in enumerate(segments):
            axes[2].axvspan(time[on], time[min(off, len(time)-1)],facecolor=palette[i % len(palette)], alpha=0.35,edgecolor='none')

        # plot 2d: smooth envelope
        axes[3].plot(env_time, results['smooth_envelope'], color='#4dac26', lw=1)
        axes[3].set_ylabel("Amplitude")
        axes[3].set_title("D) Filtered RMS", fontweight='bold')
        axes[3].axhline(results['final_threshold'], color='red',ls='--', lw=1.5, label='Threshold')
        axes[3].legend(loc='upper right', fontsize=9)
        for i, (on, off) in enumerate(segments):
            axes[3].axvspan(time[on], time[min(off, len(time)-1)],facecolor=palette[i % len(palette)], alpha=0.35,edgecolor='none')

        # plot 2e: derivative
        axes[4].plot(deriv_time, results['derivative'], color='#b2182b', lw=1)
        axes[4].set_ylabel("Amplitude")
        axes[4].set_xlabel("Time (s)")
        axes[4].set_title("E) Rectified Derivative of RMS", fontweight='bold')
        for i, (on, off) in enumerate(segments):
            axes[4].axvspan(time[on], time[min(off, len(time)-1)],facecolor=palette[i % len(palette)], alpha=0.35,edgecolor='none')

        # legend on 2a
        if segments:
            axes[0].legend(loc='lower left', fontsize=8, ncol=min(len(segments), 5))

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