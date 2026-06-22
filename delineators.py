import neurokit2 as nk

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

import WTdelineator as wav


class DelineatorInterface:
    """Abstract interface for any delineator"""

    def __init__(self, channel_names: List[str]):
        self.channel_names = channel_names

    def delineate(
        self, ecg: np.ndarray, fs: int, **kwargs
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        raise NotImplementedError


class WTdelineatorWrapper(DelineatorInterface):
    """WTdelineator wrapper - uses signalDelineation function"""

    def _clean_signal(self, signal, sampling_rate):
        return np.stack(
            [nk.ecg_clean(channel, sampling_rate=sampling_rate) for channel in signal],  # type: ignore
            axis=0,
        )  # type: ignore

    def get_ecg_parameters(
        self, ecg_cleaned: list | np.ndarray | pd.Series, sampling_rate: int
    ) -> Dict[str, np.ndarray]:
        Pwav, QRS, Twav = wav.signalDelineation(ecg_cleaned, sampling_rate)

        # P wave: [Pon, P1, P2, Pend]
        # QRS: [QRSon, Q, R, S, QRSend]
        # T wave: [Ton, T1, T2, Tend]
        return {
            "P_peaks": Pwav[:, 1],  # P1 is column 1
            "P2_peaks": Pwav[:, 2],  # P2 is column 2
            "Q_peaks": QRS[:, 1],  # Q is column 1
            "R_peaks": QRS[:, 2],  # R is column 2
            "S_peaks": QRS[:, 3],  # S is column 3
            "T_peaks": Twav[:, 1],  # T1 is column 1
            "T2_peaks": Twav[:, 2],  # T2 is column 2
            "P_onsets": Pwav[:, 0],  # Pon is column 0
            "P_offsets": Pwav[:, 3],  # Pend is column 3
            "R_onsets": QRS[:, 0],  # QRSon is column 0
            "R_offsets": QRS[:, 4],  # QRSend is column 4
            "T_onsets": Twav[:, 0],  # Ton is column 0
            "T_offsets": Twav[:, 3],  # Tend is column 3
        }

    def fill_zeroes_with_nans(self, ecg_parameters: List[Dict[str, np.ndarray]]):
        """WTdelineator returns 0 for undetected peaks, convert to NaN for consistency"""
        for channel_params in ecg_parameters:
            for key, values in channel_params.items():
                if isinstance(values, np.ndarray):
                    channel_params[key] = np.where(values == 0, np.nan, values)
                elif isinstance(values, (int, float)):
                    channel_params[key] = np.nan if values == 0 else values  # type: ignore

    def delineate(
        self, ecg: Dict[str, np.ndarray] | np.ndarray, fs: int, **kwargs
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        # Convert dict to array in channel order
        if isinstance(ecg, dict):
            ecg_array = np.array([ecg[ch_name] for ch_name in self.channel_names])
        else:
            ecg_array = ecg

        cleaned_signal = self._clean_signal(ecg_array, fs)

        # Generate parameter dicts for each channel
        ecg_parameters_dict = {}
        cleaned_signal_dict = {}

        for ch_idx, channel_name in enumerate(self.channel_names):
            ecg_parameters_dict[channel_name] = self.get_ecg_parameters(
                cleaned_signal[ch_idx], fs
            )
            cleaned_signal_dict[channel_name] = cleaned_signal[ch_idx]

        self.fill_zeroes_with_nans(list(ecg_parameters_dict.values()))

        return cleaned_signal_dict, ecg_parameters_dict


class NeuroKitDelineator(DelineatorInterface):
    """NeuroKit2 as alternative delineator"""

    def _clean_signal(self, signal, sampling_rate):
        return np.stack(
            [nk.ecg_clean(channel, sampling_rate=sampling_rate) for channel in signal],  # type: ignore
            axis=0,
        )  # type: ignore

    def get_r_peaks(
        self, ecg_cleaned: list | np.ndarray | pd.Series, sampling_rate: int
    ) -> np.ndarray:
        _, peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        return np.array(peaks["ECG_R_Peaks"])

    def get_pqst_peaks(
        self,
        ecg_cleaned: list | np.ndarray | pd.Series,
        sampling_rate: int,
        r_peaks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, peaks = nk.ecg_delineate(
            list(ecg_cleaned),
            sampling_rate=sampling_rate,
            rpeaks=r_peaks,
            method="peaks",
            show=False,
        )

        return (
            np.array(peaks["ECG_P_Peaks"]),
            np.array(peaks["ECG_Q_Peaks"]),
            np.array(peaks["ECG_S_Peaks"]),
            np.array(peaks["ECG_T_Peaks"]),
        )

    def get_p_offsets_s_offsets_q_onsets_t_offsets(
        self, ecg_cleaned, sampling_rate, r_peaks
    ):
        _, waves_dwt = nk.ecg_delineate(
            list(ecg_cleaned),
            sampling_rate=sampling_rate,
            rpeaks=r_peaks,
            method="dwt",
            show=False,
        )

        return (
            np.array(waves_dwt["ECG_P_Onsets"]),
            np.array(waves_dwt["ECG_P_Offsets"]),
            np.array(waves_dwt["ECG_R_Onsets"]),
            np.array(waves_dwt["ECG_R_Offsets"]),
            np.array(waves_dwt["ECG_T_Onsets"]),
            np.array(waves_dwt["ECG_T_Offsets"]),
        )

    def get_mean_rr(self, r_peaks):
        rr_lengths = []

        for i in range(len(r_peaks) - 1):
            rr_lengths.append(r_peaks[i + 1] - r_peaks[i])

        return round(np.nanmean(np.array(rr_lengths)))

    def get_ecg_parameters(
        self, ecg_cleaned: list | np.ndarray | pd.Series, sampling_rate: int
    ) -> Dict[str, np.ndarray]:
        ecg_parameters: dict[str, np.ndarray] = {}

        ecg_parameters["R_peaks"] = self.get_r_peaks(ecg_cleaned, sampling_rate)
        (
            ecg_parameters["P_peaks"],
            ecg_parameters["Q_peaks"],
            ecg_parameters["S_peaks"],
            ecg_parameters["T_peaks"],
        ) = self.get_pqst_peaks(ecg_cleaned, sampling_rate, ecg_parameters["R_peaks"])

        (
            ecg_parameters["P_onsets"],
            ecg_parameters["P_offsets"],
            ecg_parameters["R_onsets"],
            ecg_parameters["R_offsets"],
            ecg_parameters["T_onsets"],
            ecg_parameters["T_offsets"],
        ) = self.get_p_offsets_s_offsets_q_onsets_t_offsets(
            ecg_cleaned, sampling_rate, ecg_parameters["R_peaks"]
        )

        ecg_parameters["RR"] = self.get_mean_rr(ecg_parameters["R_peaks"])  # type: ignore

        return ecg_parameters

    def delineate(
        self, ecg: Dict[str, np.ndarray] | np.ndarray, fs: int, **kwargs
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        # Convert dict to array in channel order
        if isinstance(ecg, dict):
            ecg_array = np.array([ecg[ch_name] for ch_name in self.channel_names])
        else:
            ecg_array = ecg

        cleaned_signal = self._clean_signal(ecg_array, fs)

        # Generate parameter dicts for each channel
        ecg_parameters_dict = {}
        cleaned_signal_dict = {}

        for ch_idx, channel_name in enumerate(self.channel_names):
            ecg_parameters_dict[channel_name] = self.get_ecg_parameters(
                cleaned_signal[ch_idx], fs
            )
            cleaned_signal_dict[channel_name] = cleaned_signal[ch_idx]

        return cleaned_signal_dict, ecg_parameters_dict


class CombinedDelineator(DelineatorInterface):
    """Combines results from multiple delineators for robust detection

    Strategy:
    - If both delineators detect a point: average the two indices
    - If only one delineator detects a point: use that value
    - If neither detects a point: mark as NaN
    """

    def __init__(self, channel_names: List[str]):
        self.delineator1 = WTdelineatorWrapper(channel_names)
        self.delineator2 = NeuroKitDelineator(channel_names)
        self.channel_names = channel_names

    def merge_arrays(self, arr1, arr2):
        """Merge two arrays by stripping NaNs and matching values within RR tolerance.

        Strategy:
        1. Strip all NaN values from both arrays
        2. Use two pointers to iterate through cleaned arrays
        3. If values are within 20% of RR interval: average them, advance both
        4. Otherwise: keep the lower value, advance pointer in that array
        """
        if arr1 is None and arr2 is None:
            return np.array([])
        if arr1 is None:
            return np.asarray(arr2, dtype=float)
        if arr2 is None:
            return np.asarray(arr1, dtype=float)

        # Convert to numpy arrays
        arr1 = np.asarray(arr1, dtype=float)
        arr2 = np.asarray(arr2, dtype=float)

        # Replace 0s with NaN for consistency
        arr1 = np.where(arr1 == 0, np.nan, arr1)
        arr2 = np.where(arr2 == 0, np.nan, arr2)

        # Strip all NaN values
        valid1 = arr1[~np.isnan(arr1)]
        valid2 = arr2[~np.isnan(arr2)]

        if len(valid1) == 0 and len(valid2) == 0:
            return np.array([])
        if len(valid1) == 0:
            return valid2
        if len(valid2) == 0:
            return valid1

        # Estimate RR interval (mean difference between consecutive peaks)
        rr1 = np.mean(np.diff(valid1)) if len(valid1) > 1 else 1
        rr2 = np.mean(np.diff(valid2)) if len(valid2) > 1 else 1
        mean_rr = (rr1 + rr2) / 2
        threshold = 0.2 * mean_rr  # 20% of RR interval

        # Two-pointer merge
        merged = []
        i1, i2 = 0, 0

        while i1 < len(valid1) and i2 < len(valid2):
            val1 = valid1[i1]
            val2 = valid2[i2]
            diff = abs(val1 - val2)

            if diff <= threshold:
                # Values are close enough: average and advance both
                merged.append((val1 + val2) / 2)
                i1 += 1
                i2 += 1
            elif val1 < val2:
                # val1 is lower: keep it and advance i1
                merged.append(val1)
                i1 += 1
            else:
                # val2 is lower: keep it and advance i2
                merged.append(val2)
                i2 += 1

        # Append remaining values
        merged.extend(valid1[i1:])
        merged.extend(valid2[i2:])

        return np.array(merged, dtype=float)

    def merge_channel_results(self, params1, params2):
        """Merge delineation results from two channels using nanmean"""
        merged = {}

        # Get all keys from both parameter dicts
        all_keys = set(params1.keys()) | set(params2.keys())

        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)

            # Handle special case for RR (scalar value, not array)
            if key == "RR":
                if val1 is not None and val2 is not None:
                    merged[key] = np.nanmean([val1, val2])
                elif val1 is not None:
                    merged[key] = val1
                elif val2 is not None:
                    merged[key] = val2
                else:
                    merged[key] = np.nan
            else:
                # For array values
                merged[key] = self.merge_arrays(val1, val2)

        return merged

    def delineate(
        self, ecg: Dict[str, np.ndarray] | np.ndarray, fs: int, **kwargs
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Run both delineators and merge results"""
        # Get results from both delineators
        cleaned1, params1 = self.delineator1.delineate(ecg, fs, **kwargs)
        cleaned2, params2 = self.delineator2.delineate(ecg, fs, **kwargs)

        # Use cleaned signal from first delineator
        cleaned_signal_dict = cleaned1

        # Merge results for each channel
        merged_params = {}
        for channel_name in self.channel_names:
            merged = self.merge_channel_results(
                params1[channel_name], params2[channel_name]
            )
            merged_params[channel_name] = merged

        return cleaned_signal_dict, merged_params
