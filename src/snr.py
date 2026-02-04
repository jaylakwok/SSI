''' from lok's paper:
- take a power average of all contraction segments (signal)
- take power average of all rest segments (noise)
- average contraction power/average rest power 
- log it to get snr 
'''

import numpy as np

def calc_snr(signal, detection_results):
    segments = detection_results['segments']
    n_samples = len(signal)

    speech_mask = np.zeros(n_samples, dtype=bool)
    for onset, offset in segments:
        safe_onset = max(0, onset)
        safe_offset = min(n_samples, offset)
        speech_mask[safe_onset:safe_offset] = True
    
    rest_mask = speech_mask == False
    rest_samples = signal[rest_mask]

    global_noise_power = np.mean(np.square(rest_samples))

    segment_snrs = []
    segment_powers = []

    for onset, offset in segments:
        safe_onset = max(0, onset)
        safe_offset = min(n_samples, offset)
            
        word_samples = signal[safe_onset:safe_offset]
        word_power = np.mean(np.square(word_samples))
        snr = 10 * np.log10(word_power / global_noise_power)

        segment_powers.append(word_power)
        segment_snrs.append(snr)
    # set snr to 0 if no segment detected
    average_snr = np.mean(segment_snrs) if len(segment_snrs) > 0 else 0.0

    return {
        'noise_power': global_noise_power,
        'segment_powers': segment_powers,
        'segment_snrs': segment_snrs,
        'average_snr': float(average_snr)
    }