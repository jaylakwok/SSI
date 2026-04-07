# File paths
RAW_DATA_DIR = "raw/001_sophia_mok/session_18_03_26/wait/s10"
PROCESSED_DIR = "processed/18_03_26/wait/s10/TEST"
REPORTS_DIR = "reports/18_03_26/wait/TEST"

FILE_PATTERNS = [
    'wait_CNC_s10_ch4.npy',

    'wait_CNC_s10_ch1.npy',
    'wait_CNC_s10_ch3.npy',
    'wait_CNC_s10_ch2.npy',
    'wait_CNC_s10_ch5.npy',
    'wait_CNC_s10_ch6.npy',
    'wait_CNC_s10_ch7.npy',
    'wait_CNC_s10_ch8.npy',
]
        
MIN_PEAK_DISTANCE = 2000  # words are 20 bpm apart -> once every 3 seconds 
MIN_PEAK_PROMINENCE = 2.0  # just played around w this based on visual inspection
SAMPLING_RATE = 2000  # Hz

# filtering parameters
NOTCH_Q = 30.0
NOTCH_FREQS = [50.0]  # Mains + even  harmonic
BP_LOW = 20.0                 # Bandpass low cutoff (Hz)
BP_HIGH = 450.0               # Bandpass high cutoff (Hz)
BP_ORDER = 4                  # Butterworth filter order
WINDOW = 200.0 
HOP = 50.0
THRESHOLD_DECAY = 0.99
THRESHOLD_MIN_RATIO = 0.4      # decay floor = this × the adaptive initial threshold
IGNORE_START_MS = 8500.0        # skip this many ms at the start (motion artifact on electrode pickup)
IGNORE_END_MS = 500.0
ONSET = 0.0
OFFSET = 0.0
