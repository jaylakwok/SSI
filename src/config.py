# File paths
RAW_DATA_DIR = "raw/001_sophia_mok"
PROCESSED_DIR = "processed"
REPORTS_DIR = "reports"

FILE_PATTERNS = [
    'cnc_electrodes/bye/session_3/bye_CNC_ch8_s3.npy', 
]


'''
    "session_05_12_25/pedot20/yes_ABD20.npy",
    "session_05_12_25/pedot20/yes_RIS20.npy", 
    "session_05_12_25/pedot25/yes_RIS25.npy",
    "session_05_12_25/pedot20/yes_STRN20.npy",
    "session_05_12_25/pedot25/yes_STRN25.npy",
    "session_05_12_25/pedot20/yes_STY20.npy",
    'cnc_electrodes/no/session_1/no_CNC_ch1_s1.npy',
    'cnc_electrodes/no/session_2/no_CNC_ch1_s2.npy',
    'cnc_electrodes/no/session_3/no_CNC_ch1_s3.npy',
    'cnc_electrodes/help/session_1/help_CNC_ch1_s1.npy',
    'cnc_electrodes/help/session_2/help_CNC_ch1_s2.npy',
    'cnc_electrodes/help/session_3/help_CNC_ch1_s3.npy',
    'cnc_electrodes/bye/session_1/bye_CNC_ch1_s1.npy',
    'cnc_electrodes/bye/session_2/bye_CNC_ch1_s2.npy',
    'cnc_electrodes/bye/session_3/bye_CNC_ch1_s3.npy',
    'cnc_electrodes/yes/session_2/yes_CNC_ch1_s2.npy',
    'cnc_electrodes/yes/session_3/yes_CNC_ch1_s3.npy',
    'cnc_electrodes/yes/session_1/yes_CNC_ch1_s1.npy',
    'cnc_electrodes/bye/session_3/bye_CNC_ch1_s3.npy',
    'cnc_electrodes/wait/session_1/wait_CNC_ch1_s1.npy',
    'cnc_electrodes/wait/session_2/wait_CNC_ch1_s2.npy',
    'cnc_electrodes/wait/session_3/wait_CNC_ch1_s3.npy',
    'cnc_electrodes/pain/session_1/pain_CNC_ch1_s1.npy',
    'cnc_electrodes/pain/session_2/pain_CNC_ch1_s2.npy',
    'cnc_electrodes/pain/session_3/pain_CNC_ch1_s3.npy',
        'cnc_electrodes/yes/session_1/yes_CNC_ch2_s1.npy',
    'cnc_electrodes/yes/session_2/yes_CNC_ch2_s2.npy',
    'cnc_electrodes/yes/session_3/yes_CNC_ch2_s3.npy',

    'cnc_electrodes/wait/session_1/wait_CNC_ch2_s1.npy',
    'cnc_electrodes/wait/session_2/wait_CNC_ch2_s2.npy',
    'cnc_electrodes/wait/session_3/wait_CNC_ch2_s3.npy',

    'cnc_electrodes/pain/session_1/pain_CNC_ch2_s1.npy',
    'cnc_electrodes/pain/session_2/pain_CNC_ch2_s2.npy',
    'cnc_electrodes/pain/session_3/pain_CNC_ch2_s3.npy',
    'cnc_electrodes/bye/session_2/bye_CNC_ch2_s2.npy',
    'cnc_electrodes/bye/session_3/bye_CNC_ch2_s3.npy',
    'cnc_electrodes/yes/session_1/yes_CNC_ch3_s1.npy',
'cnc_electrodes/yes/session_2/yes_CNC_ch3_s2.npy',
'cnc_electrodes/yes/session_3/yes_CNC_ch3_s3.npy',
'cnc_electrodes/wait/session_1/wait_CNC_ch3_s1.npy',
'cnc_electrodes/wait/session_3/wait_CNC_ch3_s3.npy',
 'cnc_electrodes/pain/session_1/pain_CNC_ch3_s1.npy',
  'cnc_electrodes/pain/session_3/pain_CNC_ch3_s3.npy',
'cnc_electrodes/no/session_1/no_CNC_ch3_s1.npy',
'cnc_electrodes/no/session_2/no_CNC_ch3_s2.npy',
'cnc_electrodes/no/session_3/no_CNC_ch3_s3.npy',
'cnc_electrodes/wait/session_2/wait_CNC_ch3_s2.npy',
  'cnc_electrodes/pain/session_2/pain_CNC_ch3_s2.npy',
  'cnc_electrodes/help/session_3/help_CNC_ch3_s3.npy',
  'cnc_electrodes/help/session_2/help_CNC_ch3_s2.npy',

'cnc_electrodes/help/session_1/help_CNC_ch3_s1.npy',
'cnc_electrodes/bye/session_1/bye_CNC_ch3_s1.npy',
'cnc_electrodes/bye/session_2/bye_CNC_ch3_s2.npy',
'cnc_electrodes/bye/session_3/bye_CNC_ch3_s3.npy',
'cnc_electrodes/yes/session_1/yes_CNC_ch4_s1.npy',
    'cnc_electrodes/yes/session_2/yes_CNC_ch4_s2.npy',
     'cnc_electrodes/yes/session_3/yes_CNC_ch4_s3.npy',
     'cnc_electrodes/wait/session_2/wait_CNC_ch4_s2.npy',
      'cnc_electrodes/pain/session_1/pain_CNC_ch4_s1.npy',
    'cnc_electrodes/pain/session_2/pain_CNC_ch4_s2.npy',
    'cnc_electrodes/pain/session_3/pain_CNC_ch4_s3.npy',
        'cnc_electrodes/no/session_1/no_CNC_ch4_s1.npy',
         'cnc_electrodes/no/session_2/no_CNC_ch4_s2.npy',
    'cnc_electrodes/no/session_3/no_CNC_ch4_s3.npy',
    'cnc_electrodes/help/session_1/help_CNC_ch4_s1.npy',
    'cnc_electrodes/help/session_2/help_CNC_ch4_s2.npy',
    'cnc_electrodes/help/session_3/help_CNC_ch4_s3.npy',
       'cnc_electrodes/bye/session_1/bye_CNC_ch4_s1.npy',

 
    'cnc_electrodes/bye/session_2/bye_CNC_ch4_s2.npy',
     'cnc_electrodes/bye/session_3/bye_CNC_ch4_s3.npy',
         'cnc_electrodes/yes/session_1/yes_CNC_ch5_s1.npy',
    'cnc_electrodes/yes/session_2/yes_CNC_ch5_s2.npy',
    'cnc_electrodes/yes/session_3/yes_CNC_ch5_s3.npy',
    'cnc_electrodes/wait/session_1/wait_CNC_ch5_s1.npy',
    'cnc_electrodes/wait/session_2/wait_CNC_ch5_s2.npy',
    'cnc_electrodes/wait/session_3/wait_CNC_ch5_s3.npy',
        'cnc_electrodes/no/session_1/no_CNC_ch5_s1.npy',
    'cnc_electrodes/no/session_2/no_CNC_ch5_s2.npy',
    'cnc_electrodes/no/session_3/no_CNC_ch5_s3.npy',
    'cnc_electrodes/help/session_1/help_CNC_ch5_s1.npy',
    'cnc_electrodes/help/session_2/help_CNC_ch5_s2.npy',
    'cnc_electrodes/help/session_3/help_CNC_ch5_s3.npy',
     'cnc_electrodes/pain/session_1/pain_CNC_ch5_s1.npy',
    'cnc_electrodes/pain/session_2/pain_CNC_ch5_s2.npy',
    'cnc_electrodes/pain/session_3/pain_CNC_ch5_s3.npy',
    'cnc_electrodes/bye/session_1/bye_CNC_ch5_s1.npy',
    'cnc_electrodes/bye/session_2/bye_CNC_ch5_s2.npy',
        'cnc_electrodes/bye/session_3/bye_CNC_ch5_s3.npy',
            'cnc_electrodes/wait/session_1/wait_CNC_ch6_s1.npy',
    'cnc_electrodes/wait/session_2/wait_CNC_ch6_s2.npy',
    'cnc_electrodes/wait/session_3/wait_CNC_ch6_s3.npy',
    'cnc_electrodes/pain/session_1/pain_CNC_ch6_s1.npy',
    'cnc_electrodes/pain/session_2/pain_CNC_ch6_s2.npy',
    'cnc_electrodes/pain/session_3/pain_CNC_ch6_s3.npy',
    'cnc_electrodes/no/session_1/no_CNC_ch6_s1.npy',
    'cnc_electrodes/no/session_2/no_CNC_ch6_s2.npy',
    'cnc_electrodes/no/session_3/no_CNC_ch6_s3.npy',
    'cnc_electrodes/help/session_1/help_CNC_ch6_s1.npy',
    'cnc_electrodes/help/session_2/help_CNC_ch6_s2.npy',
    'cnc_electrodes/help/session_3/help_CNC_ch6_s3.npy',
      'cnc_electrodes/bye/session_1/bye_CNC_ch6_s1.npy',
    'cnc_electrodes/bye/session_2/bye_CNC_ch6_s2.npy',
    'cnc_electrodes/bye/session_3/bye_CNC_ch6_s3.npy',
    'cnc_electrodes/yes/session_1/yes_CNC_ch7_s1.npy',
     'cnc_electrodes/yes/session_2/yes_CNC_ch7_s2.npy',
     'cnc_electrodes/yes/session_3/yes_CNC_ch7_s3.npy',
'cnc_electrodes/pain/session_1/pain_CNC_ch7_s1.npy',
     'cnc_electrodes/pain/session_2/pain_CNC_ch7_s2.npy',
     'cnc_electrodes/pain/session_3/pain_CNC_ch7_s3.npy',
      'cnc_electrodes/wait/session_1/wait_CNC_ch7_s1.npy',
       'cnc_electrodes/wait/session_2/wait_CNC_ch7_s2.npy',
        'cnc_electrodes/no/session_2/no_CNC_ch7_s2.npy',
     'cnc_electrodes/no/session_3/no_CNC_ch7_s3.npy',
      'cnc_electrodes/no/session_1/no_CNC_ch7_s1.npy',
      'cnc_electrodes/help/session_1/help_CNC_ch7_s1.npy',
     'cnc_electrodes/help/session_2/help_CNC_ch7_s2.npy',
     'cnc_electrodes/help/session_3/help_CNC_ch7_s3.npy',
          'cnc_electrodes/bye/session_1/bye_CNC_ch7_s1.npy',
     'cnc_electrodes/bye/session_2/bye_CNC_ch7_s2.npy',
      'cnc_electrodes/bye/session_3/bye_CNC_ch7_s3.npy',
      'cnc_electrodes/yes/session_1/yes_CNC_ch8_s1.npy',
    'cnc_electrodes/yes/session_2/yes_CNC_ch8_s2.npy', 
    'cnc_electrodes/yes/session_3/yes_CNC_ch8_s3.npy',
    'cnc_electrodes/wait/session_1/wait_CNC_ch8_s1.npy',
    'cnc_electrodes/wait/session_2/wait_CNC_ch8_s2.npy',
    'cnc_electrodes/wait/session_3/wait_CNC_ch8_s3.npy',
    'cnc_electrodes/pain/session_1/pain_CNC_ch8_s1.npy',
    'cnc_electrodes/pain/session_2/pain_CNC_ch8_s2.npy',
    'cnc_electrodes/pain/session_3/pain_CNC_ch8_s3.npy',
        'cnc_electrodes/bye/session_1/bye_CNC_ch8_s1.npy',
    'cnc_electrodes/bye/session_2/bye_CNC_ch8_s2.npy',
        'cnc_electrodes/help/session_1/help_CNC_ch8_s1.npy',
    'cnc_electrodes/help/session_2/help_CNC_ch8_s2.npy',
    'cnc_electrodes/no/session_1/no_CNC_ch8_s1.npy',
    'cnc_electrodes/no/session_2/no_CNC_ch8_s2.npy',
    'cnc_electrodes/no/session_3/no_CNC_ch8_s3.npy',
    'cnc_electrodes/help/session_3/help_CNC_ch8_s3.npy',
]

'''
# signal parameters
SAMPLING_RATE = 2000  # Hz

# filtering parameters
NOTCH_Q = 30.0
NOTCH_FREQS = [50.0]  # Mains + even  harmonic
BP_LOW = 20.0                 # Bandpass low cutoff (Hz)
BP_HIGH = 450.0               # Bandpass high cutoff (Hz)
BP_ORDER = 4                  # Butterworth filter order

# speech detection — all in ms
WINDOW = 40.0 
HOP = 20.0
THRESHOLD_DECAY = 0.99
THRESHOLD_MIN_RATIO = 0.4      # decay floor = this × the adaptive initial threshold
MIN_ACTIVE_CH = 2
IGNORE_START_MS = 8500.0        # skip this many ms at the start (motion artifact on electrode pickup)
IGNORE_END_MS = 40000.0
MIN_DURATION = 950.0
MERGE_GAP = 1080.0              # merge segments separated by less than this (ms)
ONSET = 20.0
OFFSET = 120.0