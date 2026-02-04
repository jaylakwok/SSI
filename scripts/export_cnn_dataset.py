import os
import numpy as np
from speech_detection import SpeechActivityDetector
import config

TARGET_LEN = 2000  
DATA_DIR = "session_05_12_25/pedot20"     # change to pedot25 for the other run
SAVE_PREFIX = "pedot20"       # pedot25 for the other run
# ======================


def pad_or_crop(x, target_len):
    if len(x) > target_len:
        return x[:target_len]
    else:
        return np.pad(x, (0, target_len - len(x)))


def z_norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def load_signal(path):
    # Adjust if you use csv or txt — assumes .npy
    return np.load(path)


def get_label_from_filename(fname):
    # Example: "hello_trial3.npy" -> "hello"
    return fname.split("_")[0]


def process_file(detector, file_path):
    raw = load_signal(file_path)
    results = detector.detect(raw)

    clean = results['clean_signal']
    segments = results['segments']

    snippets = []
    for on, off in segments:
        snippet = clean[on:off]
        snippet = pad_or_crop(snippet, TARGET_LEN)
        snippet = z_norm(snippet)
        snippets.append(snippet)

    return snippets


def main():
    detector = SpeechActivityDetector()

    X = []
    y = []

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".npy"):
            continue

        path = os.path.join(DATA_DIR, fname)
        label = get_label_from_filename(fname)

        snippets = process_file(detector, path)

        for snip in snippets:
            X.append(snip)
            y.append(label)

        print(f"Processed {fname} → {len(snippets)} words")

    X = np.array(X)
    y = np.array(y)

    print("Final dataset shape:", X.shape)

    np.save(f"{SAVE_PREFIX}_X.npy", X)
    np.save(f"{SAVE_PREFIX}_y.npy", y)


if __name__ == "__main__":
    main()
