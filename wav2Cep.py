import librosa
import  numpy as np
import random
from scipy.fftpack import dct


def compute_2D_mfcc(file, frameSize, CepCoeficients, random_segment_duration=None, max_frames=None):

    """
    samples random audio according to segment_duration from the original wav
    and outputs random (max_len_frames * cepCoeficients)
    
    """

    y_ini, sr = librosa.load(file, sr=None)       #load signal

   ###############################(Random Sampling)##########################################
    # Convert segment duration (in seconds) to samples 
    if random_segment_duration == None:
        print("Using the complete audio")
        y = y_ini

    else:
        segment_len = int(random_segment_duration * sr)
        # Choose a random start point ensuring it doesn't exceed bounds
        if len(y_ini) > segment_len:
            start = random.randint(0, len(y_ini) - segment_len)
            y = y_ini[start:start + segment_len]
        else:
            y = y_ini  # Use whole audio if it's shorter than segment_len

    #######################################
    pre_emphasis = 0.97 #pre emphasising higher freq
    y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    #######################################
    frame_size = frameSize  # in ms breaking signal into smaller frames
    frame_stride = 0.20  # 20 ms are you sure about this? in the original this was 0.025 for ms
    frame_length, frame_step = frame_size * sr, frame_stride * sr  # Convert from seconds to samples
    signal_length = len(y_preemphasized)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    ###########################################################################
    # Pad signal to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y_preemphasized, z)
    ###############################################################
    # Slice the signal into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    #############################################################################
    frames *= np.hamming(frame_length) #applly hamming window to minimize leakages at the frame edges
    ################################################################################
    NFFT = 512 #FFT stuff
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    #################################################################################
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sr)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1)))) #from here bin/filter computation for each parallel process/filter 
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
    filter_banks = 20 * np.log10(filter_banks)  # dB transformation
    ######################################################################################
    num_ceps = CepCoeficients #discrete cosine transform (DCT) to convert mel spectrum into mel cespstral coefficients
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]


    
    if max_frames == None:
        print("No Max frame selected. Returning the complete frames from the audio segment")
        return mfcc

    # Pad or truncate MFCC matrix to max_len frames
    if mfcc.shape[0] < max_frames:
        pad_width = max_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')  # pad rows
    else:
        mfcc = mfcc[:max_frames, :]  # truncate rows

    return mfcc


def compute_mfcc(file, frameSize, CepCoeficients, random_segment_duration=None):
    
    """
    takes the full audio and outputs random (m_frames * cepCoeficients)
    
    """
    y_ini, sr = librosa.load(file, sr=None)       #load signal

    if random_segment_duration == None:
        print("Using the complete audio")
        y = y_ini
    else:
        segment_len = int(random_segment_duration * sr)
        # Choose a random start point ensuring it doesn't exceed bounds
        if len(y_ini) > segment_len:
            start = random.randint(0, len(y_ini) - segment_len)
            y = y_ini[start:start + segment_len]
        else:
            print("Using full audio")
            y = y_ini  # Use whole audio if it's shorter than segment_len

    
    #######################################
    pre_emphasis = 0.97 #pre emphasising higher freq
    y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    #######################################
    frame_size = frameSize  # in ms breaking signal into smaller frames
    frame_stride = 0.20  # 20 ms are you sure about this? in the original this was 0.025 for ms
    frame_length, frame_step = frame_size * sr, frame_stride * sr  # Convert from seconds to samples
    signal_length = len(y_preemphasized)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    ###########################################################################
    # Pad signal to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y_preemphasized, z)
    ###############################################################
    # Slice the signal into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    #############################################################################
    frames *= np.hamming(frame_length) #applly hamming window to minimize leakages at the frame edges
    ################################################################################
    NFFT = 512 #FFT stuff
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    #################################################################################
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sr)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1)))) #from here bin/filter computation for each parallel process/filter 
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
    filter_banks = 20 * np.log10(filter_banks)  # dB transformation
    ######################################################################################
    num_ceps = CepCoeficients #discrete cosine transform (DCT) to convert mel spectrum into mel cespstral coefficients
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    return mfcc