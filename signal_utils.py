import numpy as np
import pandas as pd
import wfdb
from typing import Dict, List, Tuple


def expand_to_12_leads(signal_9ch: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Expand 9-channel ECG (V1-V6, I, II, III) to 12-lead ECG by deriving augmented leads.

    Args:
        signal_9ch: (9, n_samples) array with channels in order [V1, V2, V3, V4, V5, V6, I, II, III]

    Returns:
        signal_12ch: (12, n_samples) array with leads [V1, V2, V3, V4, V5, V6, I, II, III, aVR, aVL, aVF]
        new_channel_names: List of 12 channel names
    """
    if signal_9ch.shape[0] != 9:
        raise ValueError(f"Expected 9 channels, got {signal_9ch.shape[0]}")

    # Extract limb leads (I, II, III are at indices 6, 7, 8)
    I = signal_9ch[6, :]
    II = signal_9ch[7, :]
    III = signal_9ch[8, :]

    # Derive augmented leads using Einthoven triangle relationships
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2

    # Combine into 12-lead signal
    signal_12ch = np.vstack(
        [signal_9ch, np.array([aVR, aVL, aVF])]  # V1-V6, I, II, III  # aVR, aVL, aVF
    )

    # Updated channel names
    new_channel_names = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
    ]

    return signal_12ch, new_channel_names
