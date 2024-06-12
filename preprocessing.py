import wfdb
from pathlib import Path
from config import database_name
import numpy as np
import os
import scipy.signal as sp
import matplotlib.pyplot as plt

output_dir = r'C:\Users\20190896\Downloads\Thesis\Preprocess Output'  # Update this path as needed

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def load_records(file_path):
    records = []
    with open(file_path, 'r') as file:
        for line in file:
            records.append(line.strip())
    return records


def hampel_filter(signal, window_size=7, n_sigmas=3):
    n = len(signal)
    new_signal = signal.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    for i in range(window_size, n - window_size):
        window = signal[i - window_size:i + window_size + 1]
        median = np.median(window)
        diff = np.abs(window - median)
        std_dev = k * np.median(diff)
        if np.abs(signal[i] - median) > n_sigmas * std_dev:
            new_signal[i] = median
    return new_signal


def detect_flat_lines(signal, min_length=3):
    if len(signal) == 0:
        return []
    flat_lines = np.where(np.diff(signal, prepend=signal[0]) == 0)[0]
    segments = []
    start = None
    for i in range(len(flat_lines) - 1):
        if flat_lines[i + 1] == flat_lines[i] + 1:
            if start is None:
                start = flat_lines[i]
        else:
            if start is not None:
                if flat_lines[i] - start + 1 >= min_length:
                    segments.append((start, flat_lines[i] + 1))
                start = None
    return segments


def segment_cycles(signal, fs):
    peaks, _ = sp.find_peaks(signal, distance=fs / 2)  # Assuming average heart rate of 60 bpm
    valleys, _ = sp.find_peaks(-signal, distance=fs / 2)
    return peaks, valleys


def preprocess_signals(segment_name, segment_dir, output_dir):
    try:
        # Load the signal
        segment_data = wfdb.rdrecord(record_name=segment_name, pn_dir=segment_dir)
        segment_metadata = wfdb.rdheader(record_name=segment_name, pn_dir=segment_dir)
        fs = round(segment_metadata.fs)
        total_seconds = len(segment_data.p_signal) / fs

        abp_col = None
        ppg_col = None
        for sig_no in range(len(segment_data.sig_name)):
            if "ABP" in segment_data.sig_name[sig_no]:
                abp_col = sig_no
            if "Pleth" in segment_data.sig_name[sig_no]:
                ppg_col = sig_no
            if abp_col is not None and ppg_col is not None:
                break

        # Extract ABP and PPG signals
        abp = segment_data.p_signal[:, abp_col]
        ppg = segment_data.p_signal[:, ppg_col]

        # Normalize PPG signal to zero mean and unit variance
        ppg_normalized = (ppg - np.mean(ppg)) / np.std(ppg)

        lowcut = 0.5  # Hz
        highcut = 8.0  # Hz
        sos = sp.butter(4, [lowcut, highcut], btype='bandpass', output='sos', fs=fs)

        ppg_filtered = sp.sosfiltfilt(sos, ppg_normalized)

        ppg_filtered_hampel = hampel_filter(ppg_filtered, window_size=7, n_sigmas=3)
        abp_filtered_hampel = hampel_filter(abp, window_size=100, n_sigmas=3)

        ppg_flat_lines = detect_flat_lines(ppg_filtered_hampel)
        abp_flat_lines = detect_flat_lines(abp_filtered_hampel)

        if sum([end - start for start, end in ppg_flat_lines]) / len(ppg_filtered_hampel) > 0.10:
            return False, "PPG signal has excessive flat lines"
        if sum([end - start for start, end in abp_flat_lines]) / len(abp_filtered_hampel) > 0.10:
            return False, "ABP signal has excessive flat lines"

        ppg_peaks, ppg_valleys = segment_cycles(ppg_filtered_hampel, fs)
        abp_peaks, abp_valleys = segment_cycles(abp_filtered_hampel, fs)

        ppg_flat_peaks = detect_flat_lines(ppg_filtered_hampel[ppg_peaks])
        abp_flat_peaks = detect_flat_lines(abp_filtered_hampel[abp_peaks]) if len(abp_peaks) > 0 else []

        if len(ppg_flat_peaks) / len(ppg_peaks) > 0.05:
            return False, "PPG signal has excessive flat peaks"
        if len(abp_peaks) == 0 or len(abp_flat_peaks) / len(abp_peaks) > 0.05:
            return False, "ABP signal has excessive flat peaks"

        for start, end in ppg_flat_lines + ppg_flat_peaks:
            ppg_filtered_hampel[start:end] = np.nan
        for start, end in abp_flat_lines + abp_flat_peaks:
            abp_filtered_hampel[start:end] = np.nan

        ppg_filtered_hampel = ppg_filtered_hampel[~np.isnan(ppg_filtered_hampel)]
        abp_filtered_hampel = abp_filtered_hampel[~np.isnan(abp_filtered_hampel)]

        if len(ppg_filtered_hampel) == 0 or len(abp_filtered_hampel) == 0:
            return False, "Filtered signal is empty after removing NaN values"

        t_ppg = np.arange(0, len(ppg_filtered_hampel) / fs, 1.0 / fs)
        t_abp = np.arange(0, len(abp_filtered_hampel) / fs, 1.0 / fs)

        if len(t_ppg) > len(ppg_filtered_hampel):
            t_ppg = t_ppg[:len(ppg_filtered_hampel)]
        if len(t_abp) > len(abp_filtered_hampel):
            t_abp = t_abp[:len(abp_filtered_hampel)]

        np.save(os.path.join(output_dir, f'ppg_filtered_hampel_{segment_name}.npy'), ppg_filtered_hampel)
        np.save(os.path.join(output_dir, f'abp_filtered_hampel_{segment_name}.npy'), abp_filtered_hampel)

        plt.figure()
        plt.plot(np.arange(0, len(ppg_normalized) / fs, 1.0 / fs), ppg_normalized, color='black', label='Original PPG')
        plt.title(f"Original PPG Signal from Segment {segment_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'original_ppg_{segment_name}.png'))
        plt.close()

        plt.figure()
        plt.plot(t_ppg, ppg_filtered_hampel, color='green', label='Filtered + Hampel PPG')
        plt.title(f"Filtered + Hampel PPG Signal from Segment {segment_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'filtered_hampel_ppg_{segment_name}.png'))
        plt.close()

        plt.figure()
        plt.plot(np.arange(0, len(abp) / fs, 1.0 / fs), abp, color='blue', label='Original ABP')
        plt.title(f"Original ABP Signal from Segment {segment_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'original_abp_{segment_name}.png'))
        plt.close()

        plt.figure()
        plt.plot(t_abp, abp_filtered_hampel, color='red', label='Hampel Filtered ABP')
        plt.title(f"ABP Signal After Hampel Processing from Segment {segment_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'filtered_hampel_abp_{segment_name}.png'))
        plt.close()

        return True, None

    except Exception as e:
        return False, str(e)