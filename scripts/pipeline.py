import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt

# Import your classes
from preprocess2 import EMGPreprocessor2
from speech_detection import SpeechActivityDetector
from snr import calc_snr

def test_single_file_with_plots(filepath):
    print(f"\nProcessing Single File: {filepath}")
    raw = np.load(filepath).squeeze().astype(float)
    print(F"raw from file path:{raw}")
    print(F"raw from file path shape:{raw.shape})")
    
    # plot 1 aka raw vs filtered
    preprocessor = EMGPreprocessor2()
    reports_dir = project_root / config.REPORTS_DIR
    reports_dir.mkdir(exist_ok=True, parents=True)
    filename = Path(filepath).stem
    
    preprocess_save_path = reports_dir / f"{filename}_preprocessing.png"
    preprocessor.visualize_preprocessing(raw, save_path=preprocess_save_path)
    
    # detects raw
    detector = SpeechActivityDetector(method='spc')
    detection_results = detector.detect(raw, return_metadata=True)
    
    detection_save_path = reports_dir / f"{filename}_detection.png"
    detector.visualize_detection(
        raw, 
        detection_results, 
        save_path=detection_save_path
    )
    
    return {
        'filepath': str(filepath),
        'detection_results': detection_results
    }


def process_all_files_batch(save_plots=True, save_data=True):
    data_dir = project_root / config.RAW_DATA_DIR
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    excel_rows = []
    successful = 0
    failed = 0
    target_len = 2000
    
    for i, pattern in enumerate(config.FILE_PATTERNS):
        filepath = data_dir / pattern
        print(f"\n[{i+1}/{len(config.FILE_PATTERNS)}] {pattern}")
        
        if not filepath.exists():
            print(f'File not found')
            failed += 1
            continue
        
        try:
            raw = np.load(filepath).squeeze().astype(float)
            print(f" raw batch: {np.shape (raw)}")
            duration_sec = len(raw) / config.SAMPLING_RATE
            print(f"  Duration: {duration_sec:.2f}s")
            

            preprocessor = EMGPreprocessor2()

            print('Method 1: Paper SPC (P10 + 3std)')
            det_paper = SpeechActivityDetector(method='spc')

            res_paper = det_paper.detect(raw, return_metadata=True)

            # SNR
            clean_signal = res_paper['clean_signal']
            snr_results = calc_snr(clean_signal, res_paper)

            print(f"  Global noise : {snr_results['noise_power']:.4e}")
            print(f"  Average Word SNR:   {snr_results['average_snr']:.4f} dB")

            speech_ratio = np.sum(res_paper['labels']) / len(res_paper['labels'])

            excel_rows.append({
                'Filename': Path(filepath).name,
                'Duration (s)': duration_sec,
                'Global Noise Power': snr_results['noise_power'],
                'Average SNR (dB)': snr_results['average_snr'],
                'Num Segments': len(snr_results['segment_snrs']),
                'Speech Ratio': speech_ratio,
                'All Segment SNRs': ", ".join([f"{x:.2f}" for x in snr_results['segment_snrs']])
            })

            reports_dir = project_root / config.REPORTS_DIR
            reports_dir.mkdir(exist_ok=True, parents=True)
            filename = Path(filepath).stem

            if save_plots:
                preprocessor.visualize_preprocessing(
                    raw, 
                    save_path=reports_dir / f"{filename}_preprocessing.png",
                    show=False 
                )
                
                det_paper.visualize_detection(
                    raw, res_paper, 
                    save_path=reports_dir / f"{filename}_detect.png",
                    show=False
                )

            if save_data:
                processed_dir = project_root / config.PROCESSED_DIR
                processed_dir.mkdir(exist_ok=True, parents=True)

                clean = clean_signal
                segments = res_paper['segments']
                target_len = 2000

                parts = filename.split("_")
                label = parts[1] if len(parts) > 1 else parts[0]

                X_list = []
                y_list = []

                for onset, offset in segments:
                    onset = int(max(0, onset))
                    offset = int(min(len(clean), offset))
                    if offset <= onset:
                        continue

                    snippet = clean[onset:offset]

                    # padding
                    if len(snippet) >= target_len:
                        snippet = snippet[:target_len]
                    else:
                        snippet = np.pad(snippet, (0, target_len - len(snippet)), mode="constant")

                    # z normalisation
                    snippet = (snippet - np.mean(snippet)) / (np.std(snippet) + 1e-8)

                    X_list.append(snippet.astype(np.float32))
                    y_list.append(label)

                if len(X_list) > 0:
                    X_file = np.stack(X_list, axis=0)   # (n_segments, 2000)
                    y_file = np.array(y_list, dtype=object)

                    np.save(processed_dir / f"{filename}_X.npy", X_file)
                    np.save(processed_dir / f"{filename}_y.npy", y_file)

                    print(f"  Saved: {filename}_X.npy {X_file.shape}, {filename}_y.npy {y_file.shape}")
                


        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    
    if excel_rows:
        df = pd.DataFrame(excel_rows)
        reports_dir = project_root / config.REPORTS_DIR
        excel_path = reports_dir / "batch_processing_summary.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"\nBatch summary saved to: {excel_path}")
        
    return excel_rows     


if __name__ == "__main__":
    print(f"\nProject root: {project_root}")
    print(f"Data directory: {project_root / config.RAW_DATA_DIR}")
    print(f"Reports directory: {project_root / config.REPORTS_DIR}")
    
    # change this if just want 1 file -> for debugging really
    MODE = "batch" 
    
    if MODE == "single":
        test_file = project_root / config.RAW_DATA_DIR / config.FILE_PATTERNS[0]
        if test_file.exists():
            test_single_file_with_plots(test_file)
        else:
            print(f"\n✗ File not found: {test_file}")
    
    elif MODE == "batch":
        process_all_files_batch(save_plots=True, save_data=True)