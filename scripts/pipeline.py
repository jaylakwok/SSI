import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt

from preprocess2 import EMGPreprocessor2
from speech_detection import SpeechActivityDetector
from snr import calc_snr

def process_all_files_batch(save_plots=True, save_data=True):
    data_dir = project_root / config.RAW_DATA_DIR
    
    if not data_dir.exists():
        raise FileNotFoundError(f'Not found: {data_dir}')
    
    excel_rows = []
    successful = 0
    failed = 0
    
    # target length needs updating but should be roughly same. now 2400 ms per segment so N = fx x T means N = 2000 x 2.4 = 4800 ms
    target_len = 4800  

    REFERENCE_SEGMENTS = None
    
    # process 1st file in config list as the reference to follow for all segments 
    first_filepath = data_dir / config.FILE_PATTERNS[0]
    if not first_filepath.exists():
        raise FileNotFoundError(f'First reference file not found')

    first_raw = np.load(first_filepath).astype(float).squeeze()
    print(f"\nReference file: {config.FILE_PATTERNS[0]}")
    # print(f"Shape: {first_raw.shape}")
    
    detector = SpeechActivityDetector(method='spc')
    
    # detect segment
    first_results = detector.detect(first_raw, return_metadata=True)
    REFERENCE_SEGMENTS = first_results['segments']
    
    if save_plots:
        reports_dir = project_root / config.REPORTS_DIR
        reports_dir.mkdir(exist_ok=True, parents=True)
        filename = Path(first_filepath).stem
        
        preprocessor = EMGPreprocessor2()
        preprocessor.visualize_preprocessing(first_raw, save_path=reports_dir / f"{filename}_REFERENCE_preprocessing.png",show=False)
        
        detector.visualize_detection(first_raw, first_results,save_path=reports_dir / f"{filename}_REFERENCE_detection.png",show=False)

    for i, pattern in enumerate(config.FILE_PATTERNS):
        filepath = data_dir / pattern
        print(f"\n[{i+1}/{len(config.FILE_PATTERNS)}] {pattern}")
        
        if not filepath.exists():
            print(f'File not found')
            failed += 1
            continue
        
        try:
            # Load single channel
            raw = np.load(filepath).astype(float).squeeze()
            print(f"shape: {raw.shape}")
            
            duration_sec = len(raw) / config.SAMPLING_RATE
            filename = Path(filepath).stem
            
            print(f"Duration: {duration_sec:}s")
            
            # preprocess
            detector = SpeechActivityDetector(method='spc')
            clean_signal = detector._preprocess(raw)
            
            applied_results = detector.apply_segments_to_channel(clean_signal, REFERENCE_SEGMENTS)
            applied_results['clean_signal'] = clean_signal
            rms_env = detector.compute_rms_envelope_filtered(clean_signal)
            smooth_env = detector.smooth_envelope(rms_env)
            applied_results['rms_envelope'] = rms_env
            applied_results['smooth_envelope'] = smooth_env
            applied_results['derivative'] = np.abs(np.diff(smooth_env))
            applied_results['final_threshold'] = 0
            applied_results['baseline_value'] = detector._get_baseline_value(smooth_env)
            applied_results['peaks'] = []
            
            if save_plots:
                reports_dir = project_root / config.REPORTS_DIR
                reports_dir.mkdir(exist_ok=True, parents=True)
                
                preprocessor = EMGPreprocessor2()
                preprocessor.visualize_preprocessing(raw,save_path=reports_dir / f'{filename}_preprocessing.png',show=False)
                
                detector.visualize_detection(raw, applied_results,save_path=reports_dir / f'{filename}_detection.png',show=False)
            
            # SNR 
            snr_results = calc_snr(clean_signal, applied_results,visualize=save_plots, save_path=reports_dir / f'{filename}_snr.png' if save_plots else None,sampling_rate=config.SAMPLING_RATE)
            
            speech_ratio = np.sum(applied_results['labels']) / len(applied_results['labels'])
            
            print(f"SNR: {snr_results['average_snr']:.2f}dB ± {snr_results['snr_std']:.2f}dB")
            
            if save_plots:
                print(f'SUCCESS!!! plots (preprocessing, detection, SNR)')
            
            # save data
            if save_data:
                processed_dir = project_root / config.PROCESSED_DIR
                processed_dir.mkdir(exist_ok=True, parents=True)
                
                X_list = []
                Y_list = []

                parts = filename.split('_')
                label = parts[1] if len(parts) > 1 else parts[0]

                
                for onset, offset in REFERENCE_SEGMENTS:
                    onset = int(max(0, onset))
                    offset = int(min(len(clean_signal), offset))
                    if offset <= onset:
                        continue
                    
                    snippet = clean_signal[onset:offset]
                    
                    # Pad to 4800 ms
                    if len(snippet) >= target_len:
                        snippet = snippet[:target_len]
                    else:
                        snippet = np.pad(snippet, (0, target_len - len(snippet)),  mode='constant')
                    
                    # Z normalize
                    snippet = (snippet - np.mean(snippet)) / (np.std(snippet) + 1e-8)
                    
                    X_list.append(snippet.astype(np.float32))
                    Y_list.append(label)
                
                if len(X_list) > 0:
                    X_channel = np.stack(X_list, axis=0)
                    Y_channel = np.array(Y_list, dtype = str)
                    
                    np.save(processed_dir / f'{filename}_X.npy', X_channel) # (n_segments, target_len)
                    np.save(processed_dir / f'{filename}_Y.npy', Y_channel) # label names
                    
                    print(f'SUCCESS EXPORT! {filename}_X.npy shape={X_channel.shape}' f'{filename}_Y.npy shape={Y_channel.shape}')
            
            # Excel summary
            excel_rows.append({
                'Filename': Path(filepath).name,
                'Duration (s)': duration_sec,
                'Baseline Noise Power': snr_results['noise_power'],
                'Baseline Noise Std': snr_results['noise_std'],
                'Average SNR (dB)': snr_results['average_snr'],
                'SNR Std (dB)': snr_results['snr_std'],
                'Num Segments': len(REFERENCE_SEGMENTS),
                'Speech Ratio': speech_ratio,
                'All Segment SNRs': ','.join([f'{x:.2f}' for x in snr_results['segment_snrs']])
            })
            
            successful += 1
            
        except Exception as e:
            print(f'error: {str(e)}')
            import traceback
            traceback.print_exc()
            failed += 1
    
    
    if excel_rows:
        df = pd.DataFrame(excel_rows)
        reports_dir = project_root / config.REPORTS_DIR
        excel_path = reports_dir / f'{filename}_SNR summary.xlsx'
        df.to_excel(excel_path, index=False)
    
    return excel_rows

if __name__ == '__main__':    
    process_all_files_batch(
        save_plots=True, 
        save_data=True
    )