from typing import Dict, List

import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np


class DelineationVisualizer:
    """Interactive visualization of ECG signals with delineation points using Plotly"""
    
    # Define visual properties for each peak type
    # Organized by wave (P, Q, R, S, T) with different symbols for onsets, peaks, offsets
    PEAK_STYLES = {
        # P wave - Blue family
        'P_onsets': {'color': 'blue', 'symbol': 'triangle-up', 'size': 9},
        'P_peaks': {'color': 'blue', 'symbol': 'circle', 'size': 9},
        'P2_peaks': {'color': 'blue', 'symbol': 'square', 'size': 9},
        'P_offsets': {'color': 'blue', 'symbol': 'triangle-down', 'size': 9},
        
        # Q wave - Green (only peak)
        'Q_peaks': {'color': 'green', 'symbol': 'circle', 'size': 9},
        
        # R wave - Red family
        'R_onsets': {'color': 'red', 'symbol': 'triangle-up', 'size': 9},
        'R_peaks': {'color': 'red', 'symbol': 'circle', 'size': 9},
        'R_offsets': {'color': 'red', 'symbol': 'triangle-down', 'size': 9},
        
        # S wave - Purple (only peak)
        'S_peaks': {'color': 'mediumpurple', 'symbol': 'circle', 'size': 9},
        
        # T wave - Orange family
        'T_onsets': {'color': 'orange', 'symbol': 'triangle-up', 'size': 9},
        'T_peaks': {'color': 'orange', 'symbol': 'circle', 'size': 9},
        'T2_peaks': {'color': 'orange', 'symbol': 'square', 'size': 9},
        'T_offsets': {'color': 'orange', 'symbol': 'triangle-down', 'size': 9},
    }
    
    def __init__(self, signal, qrs_features, fs: int, channel_names: List[str]):
        """
        Initialize the visualizer
        
        Args:
            signal: Either (n_channels, n_samples) array or dict {lead_name: 1D array}
            qrs_features: Delineation output (list-like by channel index or dict keyed by lead)
            fs: Sampling rate
            channel_names: Optional list of channel names
        """
        self.fs = fs
        self.qrs_features = qrs_features

        if isinstance(signal, dict):
            # Keep only channels that exist in the signal dict, preserving order
            channel_names = [name for name in channel_names if name in signal]

            if len(channel_names) == 0:
                raise ValueError("No valid channels found in signal dictionary")

            stacked_signal = []
            n_samples = None

            for name in channel_names:
                channel = np.asarray(signal[name]).squeeze()
                if channel.ndim != 1:
                    raise ValueError(f"Signal for channel '{name}' must be 1D")

                if n_samples is None:
                    n_samples = channel.shape[0]
                elif channel.shape[0] != n_samples:
                    raise ValueError("All signal channels must have the same number of samples")

                stacked_signal.append(channel)

            self.signal = np.vstack(stacked_signal)
            self.channel_names = channel_names
        else:
            signal_array = np.asarray(signal)
            if signal_array.ndim != 2:
                raise ValueError("Signal array must have shape (n_channels, n_samples)")

            self.signal = signal_array
            if channel_names is None:
                self.channel_names = [f"Channel {i}" for i in range(self.signal.shape[0])]
            else:
                self.channel_names = channel_names

        self.n_channels = self.signal.shape[0]
        self.n_samples = self.signal.shape[1]
        self.time = np.arange(self.n_samples) / fs  # Convert to seconds
    
    def _sanitize_indices(self, indices) -> np.ndarray:
        """Normalize index arrays, drop NaNs, and keep only valid sample indices."""
        indices_arr = np.asarray(indices, dtype=float).ravel()
        indices_arr = indices_arr[~np.isnan(indices_arr)].astype(int)
        return indices_arr[(indices_arr >= 0) & (indices_arr < self.n_samples)]
    
    def _convert_indices_to_time(self, indices) -> np.ndarray:
        """Convert sample indices to time in seconds"""
        indices_clean = self._sanitize_indices(indices)
        return self.time[indices_clean]
    
    def _get_signal_values(self, indices, channel_idx: int) -> np.ndarray:
        """Get signal values at given indices"""
        indices_clean = self._sanitize_indices(indices)
        return self.signal[channel_idx, indices_clean]

    def _get_channel_features(self, channel_idx: int) -> Dict[str, list]:
        """Return delineation features for a channel, supporting list and dict formats."""
        if isinstance(self.qrs_features, dict):
            lead_name = self.channel_names[channel_idx]
            if lead_name in self.qrs_features and isinstance(self.qrs_features[lead_name], dict):
                return self.qrs_features[lead_name]

            if channel_idx in self.qrs_features and isinstance(self.qrs_features[channel_idx], dict):
                return self.qrs_features[channel_idx]

            idx_key = str(channel_idx)
            if idx_key in self.qrs_features and isinstance(self.qrs_features[idx_key], dict):
                return self.qrs_features[idx_key]

            return {}

        if isinstance(self.qrs_features, (list, tuple)) and channel_idx < len(self.qrs_features):
            channel_features = self.qrs_features[channel_idx]
            if isinstance(channel_features, dict):
                return channel_features

        return {}
    
    def plot(self, channel_indices: List[int]|None = None, height_per_channel: int = 300) -> go.Figure:
        """
        Create interactive plot
        
        Args:
            channel_indices: List of channel indices to plot (default: all)
            height_per_channel: Height in pixels for each subplot
        
        Returns:
            Plotly figure object
        """
        if channel_indices is None:
            channel_indices = list(range(self.n_channels))
        
        # Validate channel indices
        channel_indices = [i for i in channel_indices if 0 <= i < self.n_channels]
        
        n_plots = len(channel_indices)
        height = height_per_channel * n_plots + 100
        
        # Create subplots
        fig = sp.make_subplots(
            rows=n_plots, 
            cols=1,
            subplot_titles=tuple(self.channel_names[i] for i in channel_indices),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Add traces for each channel
        for plot_idx, ch_idx in enumerate(channel_indices):
            row = plot_idx + 1
            
            # Add the signal trace
            fig.add_trace(
                go.Scatter(
                    x=self.time,
                    y=self.signal[ch_idx, :],
                    mode='lines',
                    name=f'{self.channel_names[ch_idx]} (signal)',
                    line=dict(color='steelblue', width=1),
                    hovertemplate='<b>Signal</b><br>Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>',
                ),
                row=row, col=1
            )
            
            # Add peaks and offsets for this channel
            features = self._get_channel_features(ch_idx)
            
            for peak_key, peak_indices in features.items():
                # Skip RR which is a single value, not indices
                if peak_key == 'RR':
                    continue
                
                if peak_indices is None or (isinstance(peak_indices, float) and np.isnan(peak_indices)):
                    continue
                
                # Handle case where peak_indices might be a scalar
                if isinstance(peak_indices, (int, float, np.integer)):
                    peak_indices = np.array([peak_indices])
                else:
                    peak_indices = np.array(peak_indices)
                
                if len(peak_indices) == 0:
                    continue
                
                # Get style for this peak type
                style = self.PEAK_STYLES.get(peak_key, {
                    'color': 'gray', 'symbol': 'circle', 'size': 8
                })
                
                # Convert indices to time and get values
                peak_times = self._convert_indices_to_time(peak_indices)
                peak_values = self._get_signal_values(peak_indices, ch_idx)
                
                if len(peak_times) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=peak_times,
                            y=peak_values,
                            mode='markers',
                            name=peak_key,
                            marker=dict(
                                color=style['color'],
                                symbol=style['symbol'],
                                size=style['size'],
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate=f'<b>{peak_key}</b><br>Time: %{{x:.3f}}s<br>Amplitude: %{{y:.3f}}<extra></extra>',
                        ),
                        row=row, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title_text="ECG Signal with Delineation Points",
            height=height,
            hovermode='x unified',
            font=dict(size=11),
        )
        
        # Update x-axis label
        fig.update_xaxes(title_text="Time (s)", row=n_plots, col=1)
        
        # Update y-axis labels
        for i, _ in enumerate(channel_indices):
            fig.update_yaxes(title_text="Amplitude (mV)", row=i+1, col=1)
        
        return fig
    
    def show(self, channel_indices: List[int]|None = None):
        """Display the plot"""
        fig = self.plot(channel_indices)
        fig.show()
