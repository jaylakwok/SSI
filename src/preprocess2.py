import numpy as np
from scipy import signal as sp
from scipy.signal import filtfilt
import config
import matplotlib.pyplot as plt

class EMGPreprocessor2:
    
    def __init__(self):
        self.fs = config.SAMPLING_RATE
        self.notch_freqs = config.NOTCH_FREQS
        self.Q = config.NOTCH_Q
        self.bp_low = config.BP_LOW
        self.bp_high = config.BP_HIGH
        self.bp_order = config.BP_ORDER

    def remove_dc(self, signal): 
        return signal - np.mean(signal)

    def bandpass(self, signal):
        b, a = sp.butter(self.bp_order, [self.bp_low, self.bp_high], btype='bandpass', fs=self.fs)
        return filtfilt(b, a, signal)
    
    def notch(self, signal):
        x = signal.copy()
        for f0 in self.notch_freqs:
            b, a = sp.iirnotch(w0=f0, Q=self.Q, fs=self.fs)
            x = sp.filtfilt(b, a, x)
        return x

    def process_signal(self, raw_signal):
        raw = np.asarray(raw_signal).squeeze().astype(float)
        print(F"raw array: {np.shape(raw)}")

        raw = np.asarray(raw_signal).squeeze().astype(float)
        
        print(F"raw: {np.shape(raw)}")
        signal = self.remove_dc(raw)
        print(F"signal_dc: {np.shape(signal)}")
        signal = self.notch(signal)
        print(F"signal_notch: {np.shape(signal)}")
        filtered = self.bandpass(signal)
        time = np.arange(len(raw)) / self.fs 
        print(F"Filtered array Shape: {np.shape(filtered)}")
        print (type(filtered))

        return {
            'raw': raw,
            'filtered': filtered,
            'time': time,
     


        }
    
    def visualize_preprocessing(self, raw_signal, save_path=None, show=True):
        processed = self.process_signal(raw_signal)
        raw = processed['raw']
        filtered = processed['filtered']
        time = processed['time']
        # remove shifted problem
        raw_centered = raw - np.mean(raw)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        axes[0].plot(time, raw_centered, 'b-', linewidth=0.5, alpha=0.5, label='Raw (Centered)')
        axes[0].plot(time, filtered, 'r-', linewidth=0.5, alpha=0.7, label='Filtered')
        
        axes[0].set_ylabel('Amplitude (μV)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_title('Time domain: raw vs filtered EMG signal')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        freqs_raw, psd_raw = sp.welch(raw, fs=self.fs, nperseg=2048)
        freqs_filt, psd_filt = sp.welch(filtered, fs=self.fs, nperseg=2048)
        
        axes[1].semilogy(freqs_raw, psd_raw, 'b-', alpha=0.7, label='Raw')
        axes[1].semilogy(freqs_filt, psd_filt, 'r-', alpha=0.8, label='Filtered')
        
        axes[1].axvline(self.bp_low, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label=f'BP Low ({self.bp_low} Hz)')
        axes[1].axvline(self.bp_high, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label=f'BP High ({self.bp_high} Hz)')
        
        for f0 in self.notch_freqs[:1]:  
            axes[1].axvline(f0, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
        
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power spectral density (μV^2/Hz)')
        axes[1].set_title(f'Frequency domain: raw vs filtered (BP: {self.bp_low}-{self.bp_high} Hz)')
        axes[1].set_xlim([0, min(500, self.fs/2)])  
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig