# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: spectral_connectivity
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Usage Examples
#
# Here are examples for how to use the API to compute the connectivity measures of interest.
#
# Every example builds its data with `spectral_connectivity.simulate`, so the structure in each plot is planted and known: `simulate_shared_oscillation` puts one sinusoid, with chosen amplitudes and phase offsets, into several signals, and `simulate_lagged_broadband` delays copies of one broadband source by a known number of samples. Each section ends with an `assert` cell that checks what its plot shows, so running the notebook verifies it. Every simulation passes `random_state`, so the output is reproducible.

# %%
import matplotlib.pyplot as plt
import numpy as np

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.simulate import (
    simulate_lagged_broadband,
    simulate_shared_oscillation,
)

# %% [markdown]
# ### Power Spectrum
# #### 200 Hz signal

# %%
# One 200 Hz sinusoid (one trial, one signal) in white noise of standard
# deviation 4. Both calls share `random_state`, so `signal` is exactly `data`
# without its noise.
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 50)
noise_level = 4
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.arange(n_time_samples) / sampling_frequency
simulation = dict(
    frequency=frequency_of_interest,
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=1,
    amplitudes=[1.0],
    random_phase_per_trial=False,
    random_state=0,
)
signal = simulate_shared_oscillation(**simulation)  # (n_time_samples, 1, 1)
data = simulate_shared_oscillation(**simulation, noise_levels=noise_level)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(time[:100], signal[:100, 0, 0], label="Signal", zorder=3)
axes[0].plot(time[:100], data[:100, 0, 0], label="Signal + Noise")
axes[0].legend()
axes[0].set_title("Time Domain")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axes[1].plot(connectivity.frequencies, connectivity.power().squeeze())
axes[1].set_title("Frequency Domain")

# %%
# The spectrum peaks at 200 Hz (within one frequency-resolution bin) and,
# away from the peak, sits at the white-noise floor 2 * noise_level**2 / fs.
power = connectivity.power()[0, :, 0]
frequencies = connectivity.frequencies
peak_frequency = frequencies[np.argmax(power)]
assert abs(peak_frequency - frequency_of_interest) <= multitaper.frequency_resolution
off_peak = np.abs(frequencies - frequency_of_interest) > 5
np.testing.assert_allclose(
    power[off_peak].mean(), 2 * noise_level**2 / sampling_frequency, rtol=0.05
)

# %% [markdown]
# #### 30 Hz signal

# %%
# One 30 Hz sinusoid (one trial, one signal) in white noise of standard
# deviation 4. Both calls share `random_state`, so `signal` is exactly `data`
# without its noise.
frequency_of_interest = 30
sampling_frequency = 1500
time_extent = (0, 50)
noise_level = 4
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.arange(n_time_samples) / sampling_frequency
simulation = dict(
    frequency=frequency_of_interest,
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=1,
    amplitudes=[1.0],
    random_phase_per_trial=False,
    random_state=0,
)
signal = simulate_shared_oscillation(**simulation)  # (n_time_samples, 1, 1)
data = simulate_shared_oscillation(**simulation, noise_levels=noise_level)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(time[:500], signal[:500, 0, 0], label="Signal", zorder=3)
axes[0].plot(time[:500], data[:500, 0, 0], label="Signal + Noise")
axes[0].legend()
axes[0].set_title("Time Domain")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axes[1].plot(connectivity.frequencies, connectivity.power().squeeze())
axes[1].set_title("Frequency Domain")
axes[1].set_xlim((0, 100))

# %%
# The spectrum peaks at 30 Hz (within one frequency-resolution bin) and,
# away from the peak, sits at the white-noise floor 2 * noise_level**2 / fs.
power = connectivity.power()[0, :, 0]
frequencies = connectivity.frequencies
peak_frequency = frequencies[np.argmax(power)]
assert abs(peak_frequency - frequency_of_interest) <= multitaper.frequency_resolution
off_peak = np.abs(frequencies - frequency_of_interest) > 5
np.testing.assert_allclose(
    power[off_peak].mean(), 2 * noise_level**2 / sampling_frequency, rtol=0.05
)

# %% [markdown]
# ### Spectrogram
#
# #### No trials, 200 Hz signal with 50 Hz signal starting at 25 seconds

# %%
# Simulate signal: one trial, (n_time_samples, 1, 1)
sampling_frequency = 1500
time_extent = (0, 50)
n_trials = 1
noise_level = 4
onset = 25  # the 50 Hz oscillation starts here
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
# 200 Hz throughout plus 50 Hz from `onset` on; the noise rides on the 200 Hz
# call, and the shared `random_state` makes `signal` exactly `data` without it.
simulation = dict(
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=n_trials,
    amplitudes=[1.0],
    random_phase_per_trial=False,
    random_state=0,
)
after_onset = (time >= onset)[:, np.newaxis, np.newaxis]
fifty_hz = after_onset * simulate_shared_oscillation(50, **simulation)
signal = simulate_shared_oscillation(200, **simulation) + fifty_hz
data = simulate_shared_oscillation(200, **simulation, noise_levels=noise_level) + fifty_hz

# Plot
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signal[:, 0, 0])
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].set_xlim((24.90, 25.10))
axes[0, 0].set_ylim((-10, 10))

axes[0, 1].plot(time, data[:, 0, 0])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].set_xlim((24.90, 25.10))
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(connectivity.frequencies, connectivity.power().squeeze())
axes[1, 0].set_xlabel("Frequency")
axes[1, 0].set_ylabel("Power")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(connectivity.frequencies, connectivity.power().squeeze())
axes[1, 1].set_xlabel("Frequency")
axes[1, 1].set_ylabel("Power")


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    time_window_duration=0.600,
    time_window_step=0.300,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
mesh = axes[2, 0].pcolormesh(
    connectivity.time,
    connectivity.frequencies,
    connectivity.power().squeeze().T,
    vmin=0.0,
    vmax=0.03,
    cmap="viridis",
    shading="auto",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].axvline(onset, color="black")
axes[2, 0].set_ylabel("Frequency")
axes[2, 0].set_xlabel("Time")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    time_window_duration=0.600,
    time_window_step=0.300,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
mesh = axes[2, 1].pcolormesh(
    connectivity.time,
    connectivity.frequencies,
    connectivity.power().squeeze().T,
    vmin=0.0,
    vmax=0.03,
    cmap="viridis",
    shading="auto",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].axvline(onset, color="black")
axes[2, 1].set_ylabel("Frequency")
axes[2, 1].set_xlabel("Time")

plt.tight_layout()
cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Power",
)
cb.outline.set_linewidth(0)

# %%
# Windows wholly before the onset have only noise at 50 Hz (the white-noise
# floor 2 * noise_level**2 / fs); windows wholly after it have 50 Hz power well
# above that floor. 200 Hz is present throughout and is the strongest peak.
power = connectivity.power()[..., 0]  # (n_windows, n_frequencies)
frequencies = connectivity.frequencies
half_window = multitaper.time_window_duration / 2
before = connectivity.time + half_window < onset
after = connectivity.time - half_window > onset
at_50_hz = power[:, np.argmin(np.abs(frequencies - 50))]
at_200_hz = power[:, np.argmin(np.abs(frequencies - 200))]
noise_floor = 2 * noise_level**2 / sampling_frequency
peak_frequency = frequencies[np.argmax(power.mean(axis=0))]
assert abs(peak_frequency - 200) <= multitaper.frequency_resolution
np.testing.assert_allclose(at_50_hz[before].mean(), noise_floor, rtol=0.25)
assert at_50_hz[after].mean() > 3 * noise_floor
assert at_200_hz[before].mean() > 3 * noise_floor
assert at_200_hz[after].mean() > 3 * noise_floor

# %% [markdown]
# #### With trial structure (time x trials)

# %%
time_halfbandwidth_product = 1

sampling_frequency = 1500
time_extent = (0, 0.600)
n_trials = 100
noise_level = 2
onset = 0.300  # the 50 Hz oscillation starts here
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
# 200 Hz throughout plus 50 Hz from `onset` on; the noise rides on the 200 Hz
# call, and the shared `random_state` makes `signal` exactly `data` without it.
simulation = dict(
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=n_trials,
    amplitudes=[1.0],
    random_phase_per_trial=False,
    random_state=0,
)
after_onset = (time >= onset)[:, np.newaxis, np.newaxis]
fifty_hz = after_onset * simulate_shared_oscillation(50, **simulation)
signal = simulate_shared_oscillation(200, **simulation) + fifty_hz
data = simulate_shared_oscillation(200, **simulation, noise_levels=noise_level) + fifty_hz

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signal[:, 0, 0])
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].set_ylim((-10, 10))

axes[0, 1].plot(time, data[:, 0, 0])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(connectivity.frequencies, connectivity.power().squeeze())

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(connectivity.frequencies, connectivity.power().squeeze())


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.060,
    time_window_step=0.060,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
mesh = axes[2, 0].pcolormesh(
    connectivity.time,
    connectivity.frequencies,
    connectivity.power().squeeze().T,
    vmin=0.0,
    vmax=0.03,
    cmap="viridis",
    shading="auto",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].axvline(onset, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.060,
    time_window_step=0.060,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
mesh = axes[2, 1].pcolormesh(
    connectivity.time,
    connectivity.frequencies,
    connectivity.power().squeeze().T,
    vmin=0.0,
    vmax=0.03,
    cmap="viridis",
    shading="auto",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].axvline(onset, color="black")

plt.tight_layout()
cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Power",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# Windows wholly before the onset have only noise at 50 Hz (the white-noise
# floor 2 * noise_level**2 / fs); windows wholly after it have 50 Hz power well
# above that floor. 200 Hz is present throughout and is the strongest peak.
power = connectivity.power()[..., 0]  # (n_windows, n_frequencies)
frequencies = connectivity.frequencies
half_window = multitaper.time_window_duration / 2
before = connectivity.time + half_window < onset
after = connectivity.time - half_window > onset
at_50_hz = power[:, np.argmin(np.abs(frequencies - 50))]
at_200_hz = power[:, np.argmin(np.abs(frequencies - 200))]
noise_floor = 2 * noise_level**2 / sampling_frequency
peak_frequency = frequencies[np.argmax(power.mean(axis=0))]
assert abs(peak_frequency - 200) <= multitaper.frequency_resolution
np.testing.assert_allclose(at_50_hz[before].mean(), noise_floor, rtol=0.25)
assert at_50_hz[after].mean() > 3 * noise_floor
assert at_200_hz[before].mean() > 3 * noise_floor
assert at_200_hz[after].mean() > 3 * noise_floor

# %% [markdown]
# #### Decrease frequency resolution by decreasing time_halfbandwidth

# %%
time_halfbandwidth_product = 3

sampling_frequency = 1500
time_extent = (0, 0.600)
n_trials = 100
noise_level = 2
onset = 0.300  # the 50 Hz oscillation starts here
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
# 200 Hz throughout plus 50 Hz from `onset` on; the noise rides on the 200 Hz
# call, and the shared `random_state` makes `signal` exactly `data` without it.
simulation = dict(
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=n_trials,
    amplitudes=[1.0],
    random_phase_per_trial=False,
    random_state=0,
)
after_onset = (time >= onset)[:, np.newaxis, np.newaxis]
fifty_hz = after_onset * simulate_shared_oscillation(50, **simulation)
signal = simulate_shared_oscillation(200, **simulation) + fifty_hz
data = simulate_shared_oscillation(200, **simulation, noise_levels=noise_level) + fifty_hz

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signal[:, 0, 0])
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].set_ylim((-10, 10))

axes[0, 1].plot(time, data[:, 0, 0])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(connectivity.frequencies, connectivity.power().squeeze())

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(connectivity.frequencies, connectivity.power().squeeze())


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.060,
    time_window_step=0.060,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
mesh = axes[2, 0].pcolormesh(
    connectivity.time,
    connectivity.frequencies,
    connectivity.power().squeeze().T,
    vmin=0.0,
    vmax=0.03,
    cmap="viridis",
    shading="auto",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].axvline(onset, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.060,
    time_window_step=0.060,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
mesh = axes[2, 1].pcolormesh(
    connectivity.time,
    connectivity.frequencies,
    connectivity.power().squeeze().T,
    vmin=0.0,
    vmax=0.03,
    cmap="viridis",
    shading="auto",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].axvline(onset, color="black")

plt.tight_layout()
cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Power",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# Windows wholly before the onset have only noise at 50 Hz (the white-noise
# floor 2 * noise_level**2 / fs); windows wholly after it have 50 Hz power well
# above that floor. 200 Hz is present throughout and is the strongest peak.
# With time_halfbandwidth_product = 3 the 60 ms windows have 2W = 100 Hz,
# which spreads each sinusoid's power over a wider band and lowers its peak.
power = connectivity.power()[..., 0]  # (n_windows, n_frequencies)
frequencies = connectivity.frequencies
half_window = multitaper.time_window_duration / 2
before = connectivity.time + half_window < onset
after = connectivity.time - half_window > onset
at_50_hz = power[:, np.argmin(np.abs(frequencies - 50))]
at_200_hz = power[:, np.argmin(np.abs(frequencies - 200))]
noise_floor = 2 * noise_level**2 / sampling_frequency
peak_frequency = frequencies[np.argmax(power.mean(axis=0))]
assert abs(peak_frequency - 200) <= multitaper.frequency_resolution
np.testing.assert_allclose(at_50_hz[before].mean(), noise_floor, rtol=0.25)
assert at_50_hz[after].mean() > 1.5 * noise_floor
assert at_200_hz[before].mean() > 1.5 * noise_floor
assert at_200_hz[after].mean() > 1.5 * noise_floor

# %% [markdown]
# ### Coherence
#
# #### No trials, 200 Hz, $\pi / 2$ phase offset

# %%
time_halfbandwidth_product = 5
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 50)
noise_level = 4
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.arange(n_time_samples) / sampling_frequency
# Two signals share one 200 Hz sinusoid; signal 2 leads signal 1 by pi / 2.
# Shape (n_time_samples, 1, 2): a single trial.
simulation = dict(
    frequency=frequency_of_interest,
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=1,
    amplitudes=[1.0, 1.0],
    phase_offsets=[0.0, np.pi / 2],
    random_phase_per_trial=False,
    random_state=0,
)
signal = simulate_shared_oscillation(**simulation)
data = simulate_shared_oscillation(**simulation, noise_levels=noise_level)

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.title("Signal", fontweight="bold")
plt.plot(time, signal[:, 0, 0], label="Signal1")
plt.plot(time, signal[:, 0, 1], label="Signal2")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Signal + Noise", fontweight="bold")
plt.plot(time, data[:, 0, :])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))
plt.legend()

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
plt.subplot(2, 2, 3)
plt.plot(connectivity.frequencies, connectivity.coherence_magnitude()[0, :, 0, 1])


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
plt.subplot(2, 2, 4)
plt.plot(connectivity.frequencies, connectivity.coherence_magnitude()[0, :, 0, 1])

# %%
# Coherence is near 1 at 200 Hz and at the noise level away from it. The
# cross-spectrum [0, 1] is E[X_0 conj(X_1)], so its phase is phase_0 - phase_1:
# signal 2 leads signal 1 by pi / 2, giving -pi / 2.
coherence = connectivity.coherence_magnitude()[0, :, 0, 1]
frequencies = connectivity.frequencies
peak = np.argmin(np.abs(frequencies - frequency_of_interest))
off_peak = np.abs(frequencies - frequency_of_interest) > multitaper.frequency_resolution
assert coherence[peak] > 0.95
assert np.median(coherence[off_peak]) < 0.1
np.testing.assert_allclose(
    connectivity.coherence_phase()[0, peak, 0, 1], -np.pi / 2, atol=0.1
)

# %% [markdown]
# #### With trial structure (time x trials), 200 Hz, $\pi / 2$ phase offset

# %%
time_halfbandwidth_product = 5
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 0.600)
n_trials = 100
noise_level = 0.5  # 0.6 s trials hold less signal than the 50 s recording above
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
# Each trial starts the shared sinusoid at a random phase, but signal 2 always
# leads signal 1 by pi / 2. Shape (n_time_samples, n_trials, 2).
simulation = dict(
    frequency=frequency_of_interest,
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=n_trials,
    amplitudes=[1.0, 1.0],
    phase_offsets=[0.0, np.pi / 2],
    random_state=0,
)
signal = simulate_shared_oscillation(**simulation)
data = simulate_shared_oscillation(**simulation, noise_levels=noise_level)

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.title("Signal", fontweight="bold")
plt.plot(time, signal[:, 0, 0], label="Signal1")
plt.plot(time, signal[:, 0, 1], label="Signal2")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim(time_extent)
plt.ylim((-2, 2))
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Signal + Noise", fontweight="bold")
plt.plot(time, data[:, 0, :])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim(time_extent)
plt.ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
plt.subplot(2, 2, 3)
plt.plot(connectivity.frequencies, connectivity.coherence_magnitude()[0, :, 0, 1])


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
plt.subplot(2, 2, 4)
plt.plot(connectivity.frequencies, connectivity.coherence_magnitude()[0, :, 0, 1])

# %%
# Coherence is near 1 at 200 Hz and at the noise level away from it. The
# cross-spectrum [0, 1] is E[X_0 conj(X_1)], so its phase is phase_0 - phase_1:
# signal 2 leads signal 1 by pi / 2, giving -pi / 2.
coherence = connectivity.coherence_magnitude()[0, :, 0, 1]
frequencies = connectivity.frequencies
peak = np.argmin(np.abs(frequencies - frequency_of_interest))
off_peak = np.abs(frequencies - frequency_of_interest) > multitaper.frequency_resolution
assert coherence[peak] > 0.95
assert np.median(coherence[off_peak]) < 0.1
np.testing.assert_allclose(
    connectivity.coherence_phase()[0, peak, 0, 1], -np.pi / 2, atol=0.1
)

# %% [markdown]
# ### Cohereograms
#
# This and the following sections use the same pair of 200 Hz signals. Before 1.5 s, signal 2's phase is drawn independently of signal 1's on every trial, so the pair is uncoupled; from 1.5 s on, both share each trial's phase and signal 2 leads by $\pi / 2$. `simulate_coupling_onset` splices two `simulate_shared_oscillation` calls to build it, and `split_at_onset` picks out a measure at 200 Hz in the windows wholly before and wholly after the onset for the assert cells.
#
# A single taper (`time_halfbandwidth_product = 1`) is used: the odd-order Slepian tapers have zero gain at the center frequency of their band, so with several tapers a pure sinusoid contributes only noise to those tapers' phases, which caps the phase-based measures (PLV, PLI, PPC) well below 1.

# %%
def simulate_coupling_onset(
    sampling_frequency,
    n_time_samples,
    n_trials,
    noise_level,
    frequency=200,
    onset=1.5,
    random_state=0,
):
    """Two signals at ``frequency`` that phase-lock (signal 2 leading by pi / 2) at ``onset``.

    Returns shape (n_time_samples, n_trials, 2). The same ``random_state`` with
    ``noise_level=0`` gives the noise-free version.
    """
    rng = np.random.default_rng(random_state)
    simulation = dict(
        frequency=frequency,
        sampling_frequency=sampling_frequency,
        n_time_samples=n_time_samples,
        n_trials=n_trials,
        amplitudes=[1.0, 1.0],
        noise_levels=noise_level,
        random_state=rng,
    )
    coupled = simulate_shared_oscillation(**simulation, phase_offsets=[0.0, np.pi / 2])
    uncoupled = simulate_shared_oscillation(**simulation)  # independent trial phases
    time = np.arange(n_time_samples) / sampling_frequency
    before_onset = (time < onset)[:, np.newaxis, np.newaxis] & (np.arange(2) == 1)
    return np.where(before_onset, uncoupled, coupled)


def split_at_onset(values, connectivity, multitaper, frequency=200, onset=1.5):
    """Split a (n_windows, n_frequencies) measure around the coupling onset.

    Returns the values at ``frequency`` in windows wholly before and wholly after
    ``onset``, and the values away from ``frequency`` in the windows after it.
    """
    frequencies = connectivity.frequencies
    peak = np.argmin(np.abs(frequencies - frequency))
    off_peak = np.abs(frequencies - frequency) > multitaper.frequency_resolution
    half_window = multitaper.time_window_duration / 2
    before = connectivity.time + half_window < onset
    after = connectivity.time - half_window > onset
    return values[before, peak], values[after, peak], values[after][:, off_peak]


# %%
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.coherence_magnitude()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.coherence_magnitude()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Coherence",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# After the onset, the pair is coherent at 200 Hz; before it, and away from
# 200 Hz, coherence stays near the chance level.
before, after, off_peak_after = split_at_onset(
    connectivity.coherence_magnitude()[..., 0, 1], connectivity, multitaper
)
assert np.all(after > 0.95)
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1 or below
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Imaginary Coherence

# %%
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies, connectivity.imaginary_coherence()[..., 0, 1].squeeze()
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies, connectivity.imaginary_coherence()[..., 0, 1].squeeze()
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.imaginary_coherence()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.imaginary_coherence()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Imaginary Coherence",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# A pi / 2 lead is purely imaginary, so after the onset the imaginary
# coherence at 200 Hz is near 1; before it, and away from 200 Hz, it is near 0.
before, after, off_peak_after = split_at_onset(
    connectivity.imaginary_coherence()[..., 0, 1], connectivity, multitaper
)
assert np.all(after > 0.95)
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1 or below
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Phase Locking Value

# %%
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies, connectivity.phase_locking_value()[..., 0, 1].squeeze()
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies, connectivity.phase_locking_value()[..., 0, 1].squeeze()
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.phase_locking_value()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.phase_locking_value()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Phase Locking Value",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# After the onset the phase difference is the same on every trial, so the
# phase locking value at 200 Hz is near 1; before it, and away from 200 Hz,
# it stays near the chance level.
before, after, off_peak_after = split_at_onset(
    connectivity.phase_locking_value()[..., 0, 1], connectivity, multitaper
)
assert np.all(after > 0.95)
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1 or below
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Phase Lag Index

# %%
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies, connectivity.phase_lag_index()[..., 0, 1].squeeze()
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies, connectivity.phase_lag_index()[..., 0, 1].squeeze()
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Phase Lag Index",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# Signal 2 leads, so Im(E[X_0 conj(X_1)]) < 0 and the signed phase lag index
# [0, 1] is -1 after the onset ([1, 0] is +1); before it, and away from
# 200 Hz, it is near 0.
values = connectivity.phase_lag_index()
before, after, off_peak_after = split_at_onset(values[..., 0, 1], connectivity, multitaper)
assert np.all(after < -0.95)
np.testing.assert_allclose(values[..., 1, 0], -values[..., 0, 1])
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Weighted Phase Lag Index

# %% pycharm={"is_executing": true}
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies,
    connectivity.weighted_phase_lag_index()[..., 0, 1].squeeze(),
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies,
    connectivity.weighted_phase_lag_index()[..., 0, 1].squeeze(),
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.weighted_phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.weighted_phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Weighted Phase Lag Index",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# Same sign convention as the phase lag index: [0, 1] is -1 after the onset
# because signal 2 leads; before it, and away from 200 Hz, it is near 0.
values = connectivity.weighted_phase_lag_index()
before, after, off_peak_after = split_at_onset(values[..., 0, 1], connectivity, multitaper)
assert np.all(after < -0.95)
np.testing.assert_allclose(values[..., 1, 0], -values[..., 0, 1])
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Debiased Squared Phase Lag Index

# %% pycharm={"is_executing": true}
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies,
    connectivity.debiased_squared_phase_lag_index()[..., 0, 1].squeeze(),
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies,
    connectivity.debiased_squared_phase_lag_index()[..., 0, 1].squeeze(),
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.debiased_squared_phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.debiased_squared_phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Debiased Squared Phase Lag Index",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# The debiased squared phase lag index is unsigned: near 1 at 200 Hz after
# the onset, near 0 before it and away from 200 Hz.
before, after, off_peak_after = split_at_onset(
    connectivity.debiased_squared_phase_lag_index()[..., 0, 1], connectivity, multitaper
)
assert np.all(after > 0.95)
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1 or below
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Debiased Squared Weighted Phase Lag Index

# %% pycharm={"is_executing": true}
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies,
    connectivity.debiased_squared_weighted_phase_lag_index()[..., 0, 1].squeeze(),
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies,
    connectivity.debiased_squared_weighted_phase_lag_index()[..., 0, 1].squeeze(),
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.debiased_squared_weighted_phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.debiased_squared_weighted_phase_lag_index()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Debiased Weighted Squared Phase Lag Index",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# Near 1 at 200 Hz after the onset, near 0 before it and away from 200 Hz.
before, after, off_peak_after = split_at_onset(
    connectivity.debiased_squared_weighted_phase_lag_index()[..., 0, 1], connectivity, multitaper
)
assert np.all(after > 0.95)
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1 or below
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Pairwise Phase Consistency

# %% pycharm={"is_executing": true}
time_halfbandwidth_product = 1
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
signal = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, 0)
data = simulate_coupling_onset(sampling_frequency, n_time_samples, n_trials, noise_level)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9), constrained_layout=True)
axes[0, 0].set_title("Signal", fontweight="bold")
axes[0, 0].plot(time, signal[:, 0, 0], label="Signal1")
axes[0, 0].plot(time, signal[:, 0, 1], label="Signal2")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_ylim((-2, 2))

axes[0, 1].set_title("Signal + Noise", fontweight="bold")
axes[0, 1].plot(time, data[:, 0, :])
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 0].plot(
    connectivity.frequencies,
    connectivity.pairwise_phase_consistency()[..., 0, 1].squeeze(),
)
axes[1, 0].set_xlim((0, multitaper.nyquist_frequency))

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
axes[1, 1].plot(
    connectivity.frequencies,
    connectivity.pairwise_phase_consistency()[..., 0, 1].squeeze(),
)
axes[1, 1].set_xlim((0, multitaper.nyquist_frequency))


multitaper = Multitaper(
    signal,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)
time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 0].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.pairwise_phase_consistency()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 0].set_ylim((0, 300))
axes[2, 0].set_xlim(time_extent)
axes[2, 0].axvline(1.5, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

mesh = axes[2, 1].pcolormesh(
    time_grid,
    freq_grid,
    connectivity.pairwise_phase_consistency()[..., 0, 1].squeeze().T,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
)
axes[2, 1].set_ylim((0, 300))
axes[2, 1].set_xlim(time_extent)
axes[2, 1].axvline(1.5, color="black")

cb = fig.colorbar(
    mesh,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Pairwise Phase Consistency",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")

# %%
# Pairwise phase consistency is near 1 at 200 Hz after the onset, near 0
# before it and away from 200 Hz.
before, after, off_peak_after = split_at_onset(
    connectivity.pairwise_phase_consistency()[..., 0, 1], connectivity, multitaper
)
assert np.all(after > 0.95)
assert np.all(np.abs(before) < 0.3)  # chance level for 100 trials is ~0.1 or below
assert np.abs(np.median(off_peak_after)) < 0.1

# %% [markdown]
# ### Group Delay
#
# Group delay is the slope of the coherence phase across frequency, so it needs signals that are coherent across a band: a sinusoid delayed by a whole number of cycles is indistinguishable from the original. Each example here delays copies of one white-noise source by 10 ms with `simulate_lagged_broadband`. A positive `delay[..., i, j]` means signal `i` leads signal `j`.
#
# #### Signal \#1 leads Signal \#2

# %% pycharm={"is_executing": true}
sampling_frequency = 1000
time_extent = (0, 1)
n_trials = 100
time_halfbandwidth_product = 1
time_lag = 0.010  # signal 1 leads signal 2 by 10 ms
lag_samples = round(time_lag * sampling_frequency)
lags = (0, lag_samples)  # samples behind the shared source; the smaller lag leads
noise_levels = [1.0, 0.5]

n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

# Both calls draw the same source (it is drawn before the noise), so `signals`
# is exactly `data` without its noise. Shape (n_time_samples, n_trials, 2).
signals = simulate_lagged_broadband(lags, 0.0, n_time_samples, n_trials, random_state=0)
data = simulate_lagged_broadband(lags, noise_levels, n_time_samples, n_trials, random_state=0)

fig, axis_handles = plt.subplots(5, 2, figsize=(12, 9), constrained_layout=True)
axis_handles[0, 0].plot(time, signals[:, 0, 0], color="blue")
axis_handles[0, 0].plot(time, signals[:, 0, 1], color="green")
axis_handles[0, 0].set_xlim((0, 0.1))
axis_handles[0, 0].set_xlabel("Time")

axis_handles[0, 1].plot(time, data[:, 0, 0], color="blue")
axis_handles[0, 1].plot(time, data[:, 0, 1], color="green")
axis_handles[0, 1].set_xlim((0, 0.1))
axis_handles[0, 1].set_xlabel("Time")

multitaper = Multitaper(
    signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 0].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 0].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

delay, slope, r_value = connectivity.group_delay()
axis_handles[4, 0].bar(
    [1, 2], [delay[..., 0, 1].squeeze(), delay[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 0].set_xlim((0.5, 2.5))
axis_handles[4, 0].axhline(0, color="black")
axis_handles[4, 0].set_xticks([1])
axis_handles[4, 0].set_xticklabels(["x1 → x2"])


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 1].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 1].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

delay, slope, r_value = connectivity.group_delay()
axis_handles[4, 1].bar(
    [1, 2], [delay[..., 0, 1].squeeze(), delay[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 1].set_xlim((0.5, 2.5))
axis_handles[4, 1].axhline(0, color="black")
axis_handles[4, 1].set_xticks([1])
axis_handles[4, 1].set_xticklabels(["x1 → x2"])

# %%
# Signal 1 leads by 10 ms: delay [0, 1] is +10 ms and [1, 0] is -10 ms (to 1 ms).
np.testing.assert_allclose(delay[..., 0, 1], time_lag, atol=1e-3)
np.testing.assert_allclose(delay[..., 1, 0], -time_lag, atol=1e-3)

# %% [markdown]
# #### Signal \#2 leads Signal \#1

# %% pycharm={"is_executing": true}
sampling_frequency = 1000
time_extent = (0, 1)
n_trials = 100
time_halfbandwidth_product = 1
time_lag = 0.010  # signal 2 leads signal 1 by 10 ms
lag_samples = round(time_lag * sampling_frequency)
lags = (lag_samples, 0)  # samples behind the shared source; the smaller lag leads
noise_levels = [1.0, 0.5]

n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

# Both calls draw the same source (it is drawn before the noise), so `signals`
# is exactly `data` without its noise. Shape (n_time_samples, n_trials, 2).
signals = simulate_lagged_broadband(lags, 0.0, n_time_samples, n_trials, random_state=0)
data = simulate_lagged_broadband(lags, noise_levels, n_time_samples, n_trials, random_state=0)

fig, axis_handles = plt.subplots(5, 2, figsize=(12, 9), constrained_layout=True)
axis_handles[0, 0].plot(time, signals[:, 0, 0], color="blue")
axis_handles[0, 0].plot(time, signals[:, 0, 1], color="green")
axis_handles[0, 0].set_xlim((0, 0.1))
axis_handles[0, 0].set_xlabel("Time")

axis_handles[0, 1].plot(time, data[:, 0, 0], color="blue")
axis_handles[0, 1].plot(time, data[:, 0, 1], color="green")
axis_handles[0, 1].set_xlim((0, 0.1))
axis_handles[0, 1].set_xlabel("Time")

multitaper = Multitaper(
    signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 0].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 0].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

delay, slope, r_value = connectivity.group_delay()
axis_handles[4, 0].bar(
    [1, 2], [delay[..., 0, 1].squeeze(), delay[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 0].set_xlim((0.5, 2.5))
axis_handles[4, 0].axhline(0, color="black")
axis_handles[4, 0].set_xticks([1])
axis_handles[4, 0].set_xticklabels(["x1 → x2"])


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 1].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 1].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

delay, slope, r_value = connectivity.group_delay()
axis_handles[4, 1].bar(
    [1, 2], [delay[..., 0, 1].squeeze(), delay[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 1].set_xlim((0.5, 2.5))
axis_handles[4, 1].axhline(0, color="black")
axis_handles[4, 1].set_xticks([1])
axis_handles[4, 1].set_xticklabels(["x1 → x2"])

# %%
# Signal 2 leads by 10 ms: delay [0, 1] is -10 ms and [1, 0] is +10 ms (to 1 ms).
np.testing.assert_allclose(delay[..., 0, 1], -time_lag, atol=1e-3)
np.testing.assert_allclose(delay[..., 1, 0], time_lag, atol=1e-3)

# %% [markdown]
# #### Signal \#2 leads Signal \#1 over time

# %% pycharm={"is_executing": true}
sampling_frequency = 1000
time_extent = (0, 2)
n_trials = 100
time_halfbandwidth_product = 1
time_lag = 0.010  # signal 2 leads signal 1 by 10 ms
lag_samples = round(time_lag * sampling_frequency)
lags = (lag_samples, 0)  # samples behind the shared source; the smaller lag leads
noise_levels = [1.0, 0.5]

n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

# Both calls draw the same source (it is drawn before the noise), so `signals`
# is exactly `data` without its noise. Shape (n_time_samples, n_trials, 2).
signals = simulate_lagged_broadband(lags, 0.0, n_time_samples, n_trials, random_state=0)
data = simulate_lagged_broadband(lags, noise_levels, n_time_samples, n_trials, random_state=0)

fig, axis_handles = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axis_handles[0, 0].plot(time, signals[:, 0, 0], color="blue")
axis_handles[0, 0].plot(time, signals[:, 0, 1], color="green")
axis_handles[0, 0].set_xlabel("Time")
axis_handles[0, 0].set_xlim(time_extent)

axis_handles[0, 1].plot(time, data[:, 0, 0], color="blue")
axis_handles[0, 1].plot(time, data[:, 0, 1], color="green")
axis_handles[0, 1].set_xlabel("Time")
axis_handles[0, 1].set_xlim(time_extent)

multitaper = Multitaper(
    signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.500,
    time_window_step=0.100,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

delay, slope, r_value = connectivity.group_delay()
axis_handles[1, 0].plot(
    connectivity.time + multitaper.time_window_duration / 2, delay[..., 0, 1]
)
axis_handles[1, 0].set_xlim(time_extent)

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.500,
    time_window_step=0.100,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

delay, slope, r_value = connectivity.group_delay()
axis_handles[1, 1].plot(
    connectivity.time + multitaper.time_window_duration / 2, delay[..., 0, 1]
)
axis_handles[1, 1].set_xlim(time_extent)

# %%
# In every window, signal 2 leads by 10 ms: delay [0, 1] is -10 ms (to 1 ms).
np.testing.assert_allclose(delay[..., 0, 1], -time_lag, atol=1e-3)

# %% [markdown]
# ## Phase Slope Index
#
# #### Signal \#1 leads Signal \#2

# %% pycharm={"is_executing": true}
sampling_frequency = 1000
time_extent = (0, 1)
n_trials = 100
time_halfbandwidth_product = 1
time_lag = 0.010  # signal 1 leads signal 2 by 10 ms
lag_samples = round(time_lag * sampling_frequency)
lags = (0, lag_samples)  # samples behind the shared source; the smaller lag leads
noise_levels = [1.0, 0.5]

n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

# Both calls draw the same source (it is drawn before the noise), so `signals`
# is exactly `data` without its noise. Shape (n_time_samples, n_trials, 2).
signals = simulate_lagged_broadband(lags, 0.0, n_time_samples, n_trials, random_state=0)
data = simulate_lagged_broadband(lags, noise_levels, n_time_samples, n_trials, random_state=0)

fig, axis_handles = plt.subplots(5, 2, figsize=(12, 9), constrained_layout=True)
axis_handles[0, 0].plot(time, signals[:, 0, 0], color="blue")
axis_handles[0, 0].plot(time, signals[:, 0, 1], color="green")
axis_handles[0, 0].set_xlim((0, 0.1))
axis_handles[0, 0].set_xlabel("Time")

axis_handles[0, 1].plot(time, data[:, 0, 0], color="blue")
axis_handles[0, 1].plot(time, data[:, 0, 1], color="green")
axis_handles[0, 1].set_xlim((0, 0.1))
axis_handles[0, 1].set_xlabel("Time")

multitaper = Multitaper(
    signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 0].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 0].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

psi = connectivity.phase_slope_index()
axis_handles[4, 0].bar(
    [1, 2], [psi[..., 0, 1].squeeze(), psi[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 0].set_xlim((0.5, 2.5))
axis_handles[4, 0].axhline(0, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 1].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 1].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

psi = connectivity.phase_slope_index()
axis_handles[4, 1].bar(
    [1, 2], [psi[..., 0, 1].squeeze(), psi[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 1].set_xlim((0.5, 2.5))
axis_handles[4, 1].axhline(0, color="black")

# %%
# Signal 1 leads: the phase slope index [0, 1] is positive and [1, 0] is its negative.
assert np.all(psi[..., 0, 1] > 0)
np.testing.assert_allclose(psi[..., 1, 0], -psi[..., 0, 1])

# %% [markdown]
# #### Signal \#2 leads Signal \#1

# %% pycharm={"is_executing": true}
sampling_frequency = 1000
time_extent = (0, 1)
n_trials = 100
time_halfbandwidth_product = 1
time_lag = 0.010  # signal 2 leads signal 1 by 10 ms
lag_samples = round(time_lag * sampling_frequency)
lags = (lag_samples, 0)  # samples behind the shared source; the smaller lag leads
noise_levels = [1.0, 0.5]

n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

# Both calls draw the same source (it is drawn before the noise), so `signals`
# is exactly `data` without its noise. Shape (n_time_samples, n_trials, 2).
signals = simulate_lagged_broadband(lags, 0.0, n_time_samples, n_trials, random_state=0)
data = simulate_lagged_broadband(lags, noise_levels, n_time_samples, n_trials, random_state=0)

fig, axis_handles = plt.subplots(5, 2, figsize=(12, 9), constrained_layout=True)
axis_handles[0, 0].plot(time, signals[:, 0, 0], color="blue")
axis_handles[0, 0].plot(time, signals[:, 0, 1], color="green")
axis_handles[0, 0].set_xlim((0, 0.1))
axis_handles[0, 0].set_xlabel("Time")

axis_handles[0, 1].plot(time, data[:, 0, 0], color="blue")
axis_handles[0, 1].plot(time, data[:, 0, 1], color="green")
axis_handles[0, 1].set_xlim((0, 0.1))
axis_handles[0, 1].set_xlabel("Time")

multitaper = Multitaper(
    signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 0].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 0].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 0].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

psi = connectivity.phase_slope_index()
axis_handles[4, 0].bar(
    [1, 2], [psi[..., 0, 1].squeeze(), psi[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 0].set_xlim((0.5, 2.5))
axis_handles[4, 0].axhline(0, color="black")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 0].squeeze()
)
axis_handles[1, 1].plot(
    connectivity.frequencies, connectivity.power()[..., 1].squeeze()
)
axis_handles[2, 1].plot(
    connectivity.frequencies, connectivity.coherence_magnitude()[..., 0, 1].squeeze()
)
axis_handles[3, 1].plot(
    connectivity.frequencies,
    connectivity.coherence_phase()[..., 0, 1].squeeze(),
    linestyle="None",
    marker="8",
)

psi = connectivity.phase_slope_index()
axis_handles[4, 1].bar(
    [1, 2], [psi[..., 0, 1].squeeze(), psi[..., 1, 0].squeeze()], color=["b", "g"]
)
axis_handles[4, 1].set_xlim((0.5, 2.5))
axis_handles[4, 1].axhline(0, color="black")

# %%
# Signal 2 leads: the phase slope index [0, 1] is negative and [1, 0] is its negative.
assert np.all(psi[..., 0, 1] < 0)
np.testing.assert_allclose(psi[..., 1, 0], -psi[..., 0, 1])

# %% [markdown]
# #### Signal \#2 leads Signal \#1 over time

# %% pycharm={"is_executing": true}
sampling_frequency = 1000
time_extent = (0, 2)
n_trials = 100
time_halfbandwidth_product = 1
time_lag = 0.010  # signal 2 leads signal 1 by 10 ms
lag_samples = round(time_lag * sampling_frequency)
lags = (lag_samples, 0)  # samples behind the shared source; the smaller lag leads
noise_levels = [1.0, 0.5]

n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

# Both calls draw the same source (it is drawn before the noise), so `signals`
# is exactly `data` without its noise. Shape (n_time_samples, n_trials, 2).
signals = simulate_lagged_broadband(lags, 0.0, n_time_samples, n_trials, random_state=0)
data = simulate_lagged_broadband(lags, noise_levels, n_time_samples, n_trials, random_state=0)

fig, axis_handles = plt.subplots(
    2, 2, figsize=(12, 9), constrained_layout=True, sharex=True
)
axis_handles[0, 0].plot(time, signals[:, 0, 0], color="blue")
axis_handles[0, 0].plot(time, signals[:, 0, 1], color="green")
axis_handles[0, 0].set_title("Signals")
axis_handles[0, 0].set_xlim(time_extent)

axis_handles[0, 1].plot(time, data[:, 0, 0], color="blue")
axis_handles[0, 1].plot(time, data[:, 0, 1], color="green")
axis_handles[0, 1].set_title("Signals")
axis_handles[0, 1].set_xlim(time_extent)

multitaper = Multitaper(
    signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.500,
    time_window_step=0.100,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

psi = connectivity.phase_slope_index()
axis_handles[1, 0].plot(
    connectivity.time + multitaper.time_window_duration / 2,
    psi[..., 0, 1],
    connectivity.time + multitaper.time_window_duration / 2,
    psi[..., 1, 0],
)
axis_handles[1, 0].set_xlim(time_extent)
axis_handles[1, 0].set_xlabel("Time [s]")
axis_handles[1, 0].set_ylabel("Phase Slope Index")

multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.500,
    time_window_step=0.100,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

psi = connectivity.phase_slope_index()
axis_handles[1, 1].plot(
    connectivity.time + multitaper.time_window_duration / 2,
    psi[..., 0, 1],
    connectivity.time + multitaper.time_window_duration / 2,
    psi[..., 1, 0],
)
axis_handles[1, 1].set_xlim(time_extent)
axis_handles[1, 1].set_xlabel("Time [s]")
axis_handles[1, 1].set_ylabel("Phase Slope Index")

# %%
# In every window signal 2 leads: the phase slope index [0, 1] is negative.
assert np.all(psi[..., 0, 1] < 0)
np.testing.assert_allclose(psi[..., 1, 0], -psi[..., 0, 1])

# %% [markdown]
# ## Canonical Coherence
#
# The advantage of canonical coherence is that it can be more statistically powerful than coherence because it is combining coherence from groups.
#
# Every signal carries a shared 20 Hz rhythm, and only group `b` also carries a private 40 Hz rhythm. The groups are therefore coherent with each other at 20 Hz, while at 40 Hz only the members of group `b` are coherent with one another: canonical coherence between the groups is high at 20 Hz and low at 40 Hz. The private rhythm is a second `simulate_shared_oscillation` call with zero amplitude in group `a`.

# %% pycharm={"is_executing": true}
from itertools import product

time_halfbandwidth_product = 2
sampling_frequency = 500
time_extent = (0, 2.400)
n_trials = 100
n_signals = 4
noise_level = 0.5
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
group_labels = ["a"] * 2 + ["b"] * 2
in_group_b = np.array(group_labels) == "b"

rng = np.random.default_rng(0)
simulation = dict(
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=n_trials,
    random_state=rng,
)
shared = simulate_shared_oscillation(
    20, **simulation, amplitudes=[1.0] * n_signals, noise_levels=noise_level
)
private = simulate_shared_oscillation(40, **simulation, amplitudes=in_group_b * 1.0)
data = shared + private  # (n_time_samples, n_trials, n_signals)


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.400,
    time_window_step=0.400,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

fig, axes = plt.subplots(nrows=n_signals, ncols=n_signals, figsize=(15, 9))
meshes = []
for ind1, ind2 in product(range(n_signals), range(n_signals)):
    if ind1 == ind2:
        vmin, vmax = connectivity.power().min(), connectivity.power().max()
    else:
        vmin, vmax = 0, 0.5
    mesh = axes[ind1, ind2].pcolormesh(
        time_grid,
        freq_grid,
        connectivity.coherence_magnitude()[..., ind1, ind2].squeeze().T,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    meshes.append(mesh)
    axes[ind1, ind2].set_ylim((0, 100))
    axes[ind1, ind2].set_xlim(time_extent)

plt.suptitle("Coherence", y=1.02, fontsize=30)
plt.tight_layout()
cb = fig.colorbar(
    meshes[-2],
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Coherence",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")


canonical_coherence, pair_labels = connectivity.canonical_coherence(group_labels)
fig = plt.figure()
mesh = plt.pcolormesh(
    time_grid,
    freq_grid,
    canonical_coherence[..., 0, 1].squeeze().T,
    vmin=0,
    vmax=0.5,
    cmap="viridis",
)
plt.ylim((0, 100))
plt.suptitle("Canonical Coherence", y=1.02, fontsize=30)
cb = fig.colorbar(
    mesh,
    ax=plt.gca(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Coherence",
)
cb.outline.set_linewidth(0)

# %%
# In every window, canonical coherence between the groups is high at the shared
# 20 Hz and low at group b's private 40 Hz, where group b is nonetheless
# internally coherent (so the low value is not an absence of signal).
frequencies = connectivity.frequencies
at_20_hz = np.argmin(np.abs(frequencies - 20))
at_40_hz = np.argmin(np.abs(frequencies - 40))
assert np.all(canonical_coherence[:, at_20_hz, 0, 1] > 0.95)
assert np.all(canonical_coherence[:, at_40_hz, 0, 1] < 0.5)
b_members = np.flatnonzero(in_group_b)
coherence_in_b = connectivity.coherence_magnitude()[:, at_40_hz, b_members[0], b_members[1]]
assert np.all(coherence_in_b > 0.8)

# %% [markdown]
# #### More signals, higher noise

# %% pycharm={"is_executing": true}
from itertools import product

time_halfbandwidth_product = 2
sampling_frequency = 500
time_extent = (0, 2.400)
n_trials = 100
n_signals = 6
noise_level = 0.8
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency
group_labels = ["a"] * 3 + ["b"] * 3
in_group_b = np.array(group_labels) == "b"

rng = np.random.default_rng(0)
simulation = dict(
    sampling_frequency=sampling_frequency,
    n_time_samples=n_time_samples,
    n_trials=n_trials,
    random_state=rng,
)
shared = simulate_shared_oscillation(
    20, **simulation, amplitudes=[1.0] * n_signals, noise_levels=noise_level
)
private = simulate_shared_oscillation(40, **simulation, amplitudes=in_group_b * 1.0)
data = shared + private  # (n_time_samples, n_trials, n_signals)


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.400,
    time_window_step=0.400,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(connectivity.frequencies, multitaper.nyquist_frequency),
)

fig, axes = plt.subplots(nrows=n_signals, ncols=n_signals, figsize=(15, 9))
meshes = []
for ind1, ind2 in product(range(n_signals), range(n_signals)):
    if ind1 == ind2:
        vmin, vmax = connectivity.power().min(), connectivity.power().max()
    else:
        vmin, vmax = 0, 0.5
    mesh = axes[ind1, ind2].pcolormesh(
        time_grid,
        freq_grid,
        connectivity.coherence_magnitude()[..., ind1, ind2].squeeze().T,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    meshes.append(mesh)
    axes[ind1, ind2].set_ylim((0, 100))
    axes[ind1, ind2].set_xlim(time_extent)

plt.suptitle("Coherence", y=1.02, fontsize=30)
plt.tight_layout()
cb = fig.colorbar(
    meshes[-2],
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Coherence",
)
cb.outline.set_linewidth(0)
print(f"frequency resolution: {multitaper.frequency_resolution}")


canonical_coherence, pair_labels = connectivity.canonical_coherence(group_labels)
fig = plt.figure()
mesh = plt.pcolormesh(
    time_grid,
    freq_grid,
    canonical_coherence[..., 0, 1].squeeze().T,
    vmin=0,
    vmax=0.5,
    cmap="viridis",
)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim((0, 100))
plt.suptitle("Canonical Coherence", y=1.02, fontsize=30)
cb = fig.colorbar(
    mesh,
    ax=plt.gca(),
    orientation="horizontal",
    shrink=0.5,
    aspect=15,
    pad=0.1,
    label="Coherence",
)
cb.outline.set_linewidth(0)

# %%
# In every window, canonical coherence between the groups is high at the shared
# 20 Hz and low at group b's private 40 Hz, where group b is nonetheless
# internally coherent (so the low value is not an absence of signal).
frequencies = connectivity.frequencies
at_20_hz = np.argmin(np.abs(frequencies - 20))
at_40_hz = np.argmin(np.abs(frequencies - 40))
assert np.all(canonical_coherence[:, at_20_hz, 0, 1] > 0.95)
assert np.all(canonical_coherence[:, at_40_hz, 0, 1] < 0.5)
b_members = np.flatnonzero(in_group_b)
coherence_in_b = connectivity.coherence_magnitude()[:, at_40_hz, b_members[0], b_members[1]]
assert np.all(coherence_in_b > 0.8)

# %% [markdown]
# ## Global Coherence
#
# Global coherence finds the linear combinations of signals that maximizes the power at a given frequency.
#
# Six signals share one 200 Hz oscillation, each with its own amplitude and phase offset, plus independent noise, so a single component captures nearly all of the power at 200 Hz.

# %%
time_halfbandwidth_product = 2
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 2.400)
n_trials = 100
n_signals = 6
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.arange(n_time_samples) / sampling_frequency

data = simulate_shared_oscillation(
    frequency_of_interest,
    sampling_frequency,
    n_time_samples,
    n_trials,
    amplitudes=0.5 + 0.2 * np.arange(n_signals),
    phase_offsets=np.linspace(0, np.pi, n_signals),
    noise_levels=0.5,
    random_state=0,
)  # (n_time_samples, n_trials, n_signals)


multitaper = Multitaper(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

global_coherence, unnormalized_global_coherence = connectivity.global_coherence()
print(global_coherence.shape)  # n_time, n_frequencies, n_components

# Extract non-negative frequencies (first N//2+1 frequencies)
n_nonneg_freqs = len(connectivity.frequencies)
global_coherence_nonneg = global_coherence[:, :n_nonneg_freqs, 0].T  # (freqs, time)

time_grid, freq_grid = np.meshgrid(
    np.append(connectivity.time, time_extent[-1]),
    np.append(
        connectivity.frequencies, connectivity.frequencies[-1]
    ),  # Add edge for pcolormesh
)
plt.figure()
plt.pcolormesh(
    time_grid,
    freq_grid,
    global_coherence_nonneg,
    shading="flat",
)
plt.title("Global Coherence (1st component)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")

# %%
# The leading component carries nearly all of the power at 200 Hz in every
# window; away from it, noise alone gives it only about 1 / n_signals.
frequencies = connectivity.frequencies
peak = np.argmin(np.abs(frequencies - frequency_of_interest))
off_peak = np.abs(frequencies - frequency_of_interest) > multitaper.frequency_resolution
assert np.all(global_coherence[:, peak, 0] > 0.95)
# global_coherence spans all FFT frequencies; the first frequencies.size are >= 0.
assert np.median(global_coherence[:, : frequencies.size][:, off_peak, 0]) < 2 / n_signals

# %% [markdown] pycharm={"is_executing": true, "name": "#%% md\n"}
# ## Xarray interface
#
# The xarray interface provides three things:
# 1. a nicely labeled output for the connectivity dimensions
# 2. a unified way of estimating the spectral power and connectivity together.
# 3. easy and quick plotting

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
coherence_magnitude = multitaper_connectivity(
    data,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.080,
    time_window_step=0.080,
    method="coherence_magnitude",
)

coherence_magnitude

# %%
coherence_magnitude.plot(col="source", row="target", x="time")

# %%
# The labeled result holds the same numbers as `Connectivity.coherence_magnitude`
# for the same data and parameters (the Global Coherence section above).
np.testing.assert_allclose(
    coherence_magnitude.values, connectivity.coherence_magnitude(), equal_nan=True
)
