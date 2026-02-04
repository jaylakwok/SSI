
1. PREPROCESSING                                
- Bandpass 20-450 Hz (4th order Butterworth) 
- Notch 50, 100, 150, 200 Hz                 
- Motion artifact removal by visual inspection i.e cut front and back ends 

                      ↓

2. SPEECH ACTIVITY DETECTION                    
- Two stage (local and global)                 
- 40ms window, 20ms overlap                  
- Multi-channel fusion in paper, but we have single channel so neglect
- uses adaptive thresholding mu + k*std where k is ~ 3 but can be tuned. online forums say visual inspection asw

               ↓                        ↓

3A. STATISTICAL ANALYSIS PATH     3B. CNN PIPELINE    
- SNR CALCULATION                 - Z-score normalise
- Per channel: 10×log(P_s/P_n)      
- mean ± std EACH channel         - Generate inputs:   
                                    Option 1: 1D signal
- Extract features:                 Option 2: Mel-spec
  - RMS, MAV, VAR, WL, SD
                                 - Feed to maks for training 
                                 - evaluate:
                                   - decoding accuracy 
                                   - optimum position 
                                 - text to speech

