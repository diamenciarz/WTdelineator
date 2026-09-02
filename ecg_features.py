import numpy as np
import neurokit2 as nk
from typing import Any, Dict, Callable, List
import pandas as pd

# ====================== METRIC FUNCTIONS ======================


def metric_heart_rate(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    """Mean heart rate (bpm)"""
    rpeaks = delineation.get("R_peaks")
    if rpeaks is None or len(rpeaks) < 2:
        return np.nan
    rr = np.diff(rpeaks) / fs
    return float(60 / np.nanmean(rr))


def metric_hr_min(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    rpeaks = delineation.get("R_peaks")
    if rpeaks is None or len(rpeaks) < 2:
        return np.nan
    rr = np.diff(rpeaks) / fs
    return float(60 / np.nanmax(rr))


def metric_hr_max(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    rpeaks = delineation.get("R_peaks")
    if rpeaks is None or len(rpeaks) < 2:
        return np.nan
    rr = np.diff(rpeaks) / fs
    return float(60 / np.nanmin(rr))


def metric_HRV_SDNN(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    """Standard deviation of NN intervals"""
    rpeaks = delineation.get("R_peaks")
    if rpeaks is None or len(rpeaks) < 3:
        return np.nan
    return nk.hrv_time(rpeaks, sampling_rate=fs)["HRV_SDNN"].iloc[0]


def metric_HRV_RMSSD(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    rpeaks = delineation.get("R_peaks")
    if rpeaks is None or len(rpeaks) < 3:
        return np.nan
    return nk.hrv_time(rpeaks, sampling_rate=fs)["HRV_RMSSD"].iloc[0]


def metric_PR_interval(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    return _compute_interval(ecg, fs, delineation, "PR")


def metric_QRS_duration(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    return _compute_interval(ecg, fs, delineation, "QRS")


def metric_QT_interval(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    return _compute_interval(ecg, fs, delineation, "QT")


def metric_JT_interval(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    """JT Interval = QT - QRS"""
    qt = metric_QT_interval(ecg, fs, delineation)
    qrs = metric_QRS_duration(ecg, fs, delineation)
    if np.isnan(qt) or np.isnan(qrs):
        return np.nan
    return qt - qrs


def metric_QTc_Bazett(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    """Bazett formula: QTc = QT / √RR"""
    qt = metric_QT_interval(ecg, fs, delineation)
    rr = _get_mean_rr(delineation, fs)
    if np.isnan(qt) or np.isnan(rr):
        return np.nan
    return float(qt / np.sqrt(rr))


def metric_QTc_Fridericia(
    ecg: np.ndarray, fs: int, delineation: Dict, **kwargs
) -> float:
    """Fridericia formula: QTc = QT / ∛RR"""
    qt = metric_QT_interval(ecg, fs, delineation)
    rr = _get_mean_rr(delineation, fs)
    if np.isnan(qt) or np.isnan(rr):
        return np.nan
    return float(qt / (rr ** (1 / 3)))


def metric_P_amplitude(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    peaks = delineation.get("P_peaks")
    if peaks is None or len(peaks) == 0:
        return np.nan
    amps = [ecg[int(p)] for p in peaks if not np.isnan(p) and 0 <= int(p) < len(ecg)]
    return float(np.nanmean(amps))


def metric_R_amplitude(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    peaks = delineation.get("R_peaks")
    if peaks is None or len(peaks) == 0:
        return np.nan
    amps = [ecg[int(p)] for p in peaks if not np.isnan(p) and 0 <= int(p) < len(ecg)]
    return float(np.nanmean(amps))


def metric_S_amplitude(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    peaks = delineation.get("S_peaks")
    if peaks is None or len(peaks) == 0:
        return np.nan
    amps = [ecg[int(p)] for p in peaks if not np.isnan(p) and 0 <= int(p) < len(ecg)]
    return float(np.nanmean(amps))


def metric_T_amplitude(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    peaks = delineation.get("T_peaks")
    if peaks is None or len(peaks) == 0:
        return np.nan
    amps = [ecg[int(p)] for p in peaks if not np.isnan(p) and 0 <= int(p) < len(ecg)]
    return float(np.nanmean(amps))


def metric_STE_Jpoint(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    return _st_level(ecg, fs, delineation, offset_ms=0)


def metric_STE60(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    return _st_level(ecg, fs, delineation, offset_ms=60)


def metric_STE80(ecg: np.ndarray, fs: int, delineation: Dict, **kwargs) -> float:
    return _st_level(ecg, fs, delineation, offset_ms=80)


def metric_smith_formula_3var(
    ecg: np.ndarray, fs: int, delineation: Dict, **kwargs
) -> float:
    """3-variable Smith formula (STE60 in V3 + QTc - RA in V4)"""
    ste60 = metric_STE60(ecg, fs, delineation)
    qtc = metric_QTc_Bazett(ecg, fs, delineation)
    ra = metric_R_amplitude(ecg, fs, delineation)  # assume lead V4 or similar
    if np.isnan(ste60) or np.isnan(qtc) or np.isnan(ra):
        return np.nan
    return (1.196 * ste60) + (0.059 * qtc) - (0.326 * ra)


# ====================== HELPERS (unchanged from previous) ======================


def _get_mean_rr(delineation: Dict, fs: int) -> float:
    rpeaks = delineation.get("R_peaks")
    if rpeaks is None or len(rpeaks) < 2:
        return np.nan
    return float(np.nanmean(np.diff(rpeaks) / fs))


def _compute_interval(
    ecg: np.ndarray, fs: int, delineation: Dict, interval_type: str
) -> float:
    """Compute ECG intervals (PR, QRS, QT) in milliseconds"""
    try:
        intervals = []

        if interval_type == "PR":
            # PR interval: from P onset to QRS onset (R onset)
            p_onsets = delineation.get("P_onsets")
            r_onsets = delineation.get("R_onsets")
            if p_onsets is None or r_onsets is None:
                return np.nan
            for p_on, r_on in zip(p_onsets, r_onsets):
                if not np.isnan(p_on) and not np.isnan(r_on):
                    intervals.append((r_on - p_on) / fs * 1000)  # Convert to ms

        elif interval_type == "QRS":
            # QRS duration: from QRS onset to QRS offset (R onset to R offset)
            r_onsets = delineation.get("R_onsets")
            r_offsets = delineation.get("R_offsets")
            if r_onsets is None or r_offsets is None:
                return np.nan
            for r_on, r_off in zip(r_onsets, r_offsets):
                if not np.isnan(r_on) and not np.isnan(r_off):
                    intervals.append((r_off - r_on) / fs * 1000)  # Convert to ms

        elif interval_type == "QT":
            # QT interval: from QRS onset to T offset
            r_onsets = delineation.get("R_onsets")
            t_offsets = delineation.get("T_offsets")
            if r_onsets is None or t_offsets is None:
                return np.nan
            for r_on, t_off in zip(r_onsets, t_offsets):
                if not np.isnan(r_on) and not np.isnan(t_off):
                    intervals.append((t_off - r_on) / fs * 1000)  # Convert to ms

        else:
            return np.nan

        if len(intervals) == 0:
            return np.nan
        return float(np.nanmean(intervals))

    except:
        return np.nan


def _st_level(
    ecg: np.ndarray, fs: int, delineation: Dict, offset_ms: int = 60
) -> float:
    j_points = delineation.get("R_offsets")
    if j_points is None or len(j_points) == 0:
        return np.nan
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    levels = []
    offset_samples = int(offset_ms * fs / 1000)
    for jp in j_points:
        if np.isnan(jp):
            continue
        st_idx = int(jp) + offset_samples
        if st_idx >= len(cleaned):
            continue
        baseline_start = max(0, int(jp) - 120)
        baseline: np.floating[Any] = np.nanmean(cleaned[baseline_start : int(jp) - 20])
        levels.append(cleaned[st_idx] - baseline)
    return float(np.nanmean(levels))


# ====================== REGISTRY ======================

METRIC_REGISTRY: Dict[str, Callable] = {
    # 1. Temporal / HR / HRV
    "heart_rate": metric_heart_rate,
    "hr_min": metric_hr_min,
    "hr_max": metric_hr_max,
    "pr_interval": metric_PR_interval,
    "qrs_duration": metric_QRS_duration,
    "qt_interval": metric_QT_interval,
    "jt_interval": metric_JT_interval,
    "qtc_bazett": metric_QTc_Bazett,
    "qtc_fridericia": metric_QTc_Fridericia,
    "hrv_sdnn": metric_HRV_SDNN,
    "hrv_rmssd": metric_HRV_RMSSD,
    # 2. Amplitudes
    "p_amplitude": metric_P_amplitude,
    "r_amplitude": metric_R_amplitude,
    "s_amplitude": metric_S_amplitude,
    "t_amplitude": metric_T_amplitude,
    # 3. ST / J-point
    "ste_j_point": metric_STE_Jpoint,
    "ste60": metric_STE60,
    "ste80": metric_STE80,
    # 6. Risk scores
    "smith_3var": metric_smith_formula_3var,
}


def compute_ecg_metrics(
    ecg_signal: np.ndarray | Dict[str, np.ndarray],
    fs: int,
    delineation: Dict[str, Dict[str, np.ndarray]],
    requested_metrics: List[str] = list(),
    lead_name: str = "Unknown",
) -> pd.DataFrame:
    """
    Compute only the requested metrics using the registry.
    If requested_metrics is empty → compute all.

    Args:
        ecg_signal: ECG signal(s) - either 1D array or dict of signals keyed by lead name
        fs: Sampling frequency in Hz
        delineation: Dict of delineation dicts, keyed by lead name
        requested_metrics: List of metric names to compute, or empty list for all
        lead_name: Which lead/channel to compute metrics for
    """
    if len(requested_metrics) == 0:
        requested_metrics = list(METRIC_REGISTRY.keys())

    # Extract the signal and delineation for the specified lead
    if isinstance(ecg_signal, dict):
        signal = ecg_signal[lead_name]
    else:
        signal = ecg_signal

    channel_delineation = delineation[lead_name]

    results: dict[str, float | str] = {"lead": lead_name}

    for metric_name in requested_metrics:
        if metric_name.lower() in METRIC_REGISTRY:
            try:
                func = METRIC_REGISTRY[metric_name.lower()]
                results[metric_name] = func(signal, fs, channel_delineation)
            except Exception as e:
                results[metric_name] = np.nan
                print(f"Warning: Failed to compute {metric_name}: {e}")

    return pd.DataFrame([results])


# Can add more (ST slope, axes, morphology, etc.)
