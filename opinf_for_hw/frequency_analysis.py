import scipy
import xarray as xr
from ot import wasserstein_1d


def wasserstein_fourier_metric(ts0, ts1, frequency=1 / 0.025, nperseg=4000):
    freq0, ps0 = scipy.signal.welch(
        ts0, fs=frequency, nperseg=nperseg, detrend="constant", average="mean"
    )

    freq1, ps1 = scipy.signal.welch(
        ts1, fs=frequency, nperseg=nperseg, detrend="constant", average="mean"
    )

    return wasserstein_1d(freq0, freq1, ps0, ps1, p=2)


def power_spectrum(data, frequency=1 / 0.025, nperseg=4000):
    freq, ps = scipy.signal.welch(
        data, fs=frequency, nperseg=nperseg, detrend="constant", average="mean"
    )
    return xr.DataArray(ps, dims="freq_time", coords={"freq_time": freq})
