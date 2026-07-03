#%%
import os
import struct

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


BASE_PATH = os.path.dirname(__file__)
BIN_FILE = "O4scope_4.bin"

CHANNEL_MAP = {
    "CH1": (BIN_FILE, 1),
}

TRACE_COLORS = {
    "CH1": "#78206E",
}

SPIKE_CONFIG = {
    "CH1": {"height": 0.2, "distance_sec": 0.005},
}

PLOT_CONFIG = {
    "x_label": "Time (s)",
    "show_grid": True,
    "grid_style": "--",
    "grid_alpha": 0.35,
    "x_step": 10,
}

RATE_CONFIG = {
    "bin_width_sec": 0.01,
    "sigma_sec": 0.1,
    "y_label": "Firing Rate (Hz)",
}


def load_trace(base_path, bin_file, channel_num, t_start=None, t_end=None):
    """Load a single channel from an Agilent/Keysight oscilloscope binary (.bin) file."""
    path = os.path.join(base_path, bin_file)

    with open(path, "rb") as f:
        cookie = f.read(2)
        if cookie != b"AG":
            raise ValueError("Not a valid Agilent/Keysight bin file.")

        f.read(2)
        _, num_waveforms = struct.unpack("2i", f.read(8))

        target_idx = channel_num - 1
        if target_idx >= num_waveforms:
            raise ValueError(f"Channel {channel_num} not found in the bin file.")

        time_arr = None
        voltage_arr = None

        for i in range(num_waveforms):
            header_bytes = f.read(140)
            points = struct.unpack_from("i", header_bytes, 12)[0]
            num_buffers = struct.unpack_from("i", header_bytes, 8)[0]
            x_inc = struct.unpack_from("d", header_bytes, 32)[0]
            x_org = struct.unpack_from("d", header_bytes, 40)[0]

            for _ in range(num_buffers):
                buf_header = f.read(12)
                _, buf_type, _, buf_size = struct.unpack("ihhi", buf_header)
                data_bytes = f.read(buf_size)

                if i == target_idx and buf_type == 1:
                    time_arr = x_org + np.arange(points) * x_inc
                    voltage_arr = np.frombuffer(data_bytes, dtype=np.float32)

        if time_arr is None or voltage_arr is None:
            raise ValueError(f"Failed to load data for channel {channel_num}.")

    if t_start is not None or t_end is not None:
        mask = np.ones(len(time_arr), dtype=bool)
        if t_start is not None:
            mask &= time_arr >= t_start
        if t_end is not None:
            mask &= time_arr <= t_end
        time_arr = time_arr[mask]
        voltage_arr = voltage_arr[mask]

    return time_arr, voltage_arr


def detect_spikes(time_arr, voltage, height, distance_sec):
    """Return spike times (seconds) from a voltage trace."""
    if len(time_arr) < 2:
        return np.array([])

    dt = float(time_arr[1] - time_arr[0])
    distance_samples = max(1, int(distance_sec / dt))
    peaks, _ = find_peaks(voltage, height=height, distance=distance_samples)
    return time_arr[peaks]


def calculate_instantaneous_firing_rate(spike_times, time_arr, bin_width_sec, sigma_sec):
    """Return a smoothed instantaneous firing-rate trace."""
    if len(time_arr) < 2:
        return np.array([]), np.array([])

    bins = np.arange(time_arr[0], time_arr[-1] + bin_width_sec, bin_width_sec)
    if len(bins) < 2:
        return np.array([]), np.array([])

    counts, edges = np.histogram(spike_times, bins=bins)
    rate = counts / bin_width_sec
    sigma_bins = sigma_sec / bin_width_sec
    smoothed_rate = gaussian_filter1d(rate.astype(float), sigma=sigma_bins)
    centers = edges[:-1] + bin_width_sec / 2
    return centers, smoothed_rate


def plot_four_channels(time_window=None):
    """Plot the four oscilloscope channels as stacked traces."""
    if time_window is None:
        t_start = None
        t_end = None
    else:
        t_start, t_end = time_window

    fig, axs = plt.subplots(len(CHANNEL_MAP), 1, figsize=(14, 4 * len(CHANNEL_MAP)), sharex=True)
    axs = np.atleast_1d(axs)
    fig.suptitle("Oscilloscope Traces", fontsize=14, fontweight="bold")

    for ax, name in zip(axs, CHANNEL_MAP):
        csv_file, ch = CHANNEL_MAP[name]
        time, voltage = load_trace(BASE_PATH, csv_file, ch, t_start, t_end)

        ax.plot(time, voltage, color=TRACE_COLORS[name], linewidth=0.8)
        ax.set_ylabel(name, rotation=0, labelpad=35, va="center", ha="right")
        ax.grid(PLOT_CONFIG["show_grid"], linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
        ax.tick_params(labelsize=9)

    axs[-1].set_xlabel(PLOT_CONFIG["x_label"])
    axs[-1].xaxis.set_major_locator(MultipleLocator(float(PLOT_CONFIG["x_step"])))
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, "oscilloscope_four_channels.png"), dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_instantaneous_firing_rates(time_window=None):
    """Detect spikes on each channel and plot the instantaneous firing rates."""
    if time_window is None:
        t_start = None
        t_end = None
    else:
        t_start, t_end = time_window

    fig, axs = plt.subplots(len(CHANNEL_MAP), 1, figsize=(14, 4 * len(CHANNEL_MAP)), sharex=True)
    axs = np.atleast_1d(axs)
    fig.suptitle("Instantaneous Firing Rates", fontsize=14, fontweight="bold")

    for ax, name in zip(axs, CHANNEL_MAP):
        csv_file, ch = CHANNEL_MAP[name]
        time, voltage = load_trace(BASE_PATH, csv_file, ch, t_start, t_end)
        cfg = SPIKE_CONFIG[name]
        spike_times = detect_spikes(time, voltage, cfg["height"], cfg["distance_sec"])
        rate_time, rate = calculate_instantaneous_firing_rate(
            spike_times,
            time,
            RATE_CONFIG["bin_width_sec"],
            RATE_CONFIG["sigma_sec"],
        )

        ax.plot(rate_time, rate, color=TRACE_COLORS[name], linewidth=1.4)
        ax.set_ylabel(f"{name}\n{RATE_CONFIG['y_label']}", rotation=0, labelpad=42, va="center", ha="right")
        ax.grid(True, linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])
        ax.tick_params(labelsize=9)

    axs[-1].set_xlabel(PLOT_CONFIG["x_label"])
    axs[-1].xaxis.set_major_locator(MultipleLocator(float(PLOT_CONFIG["x_step"])))
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, "oscilloscope_instantaneous_firing_rates.png"), dpi=300, bbox_inches="tight")
    plt.show()
    return fig


#%%
plot_four_channels(time_window=(-1, 2))

#%%
plot_instantaneous_firing_rates(time_window=(-1, 2))