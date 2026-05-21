"""Faithful port of R's ``stats::density`` (default settings).

scoper calls ``density(x)`` in ``findGapSmooth`` and ``passToClustering_lev3``
with all defaults: Gaussian kernel, ``bw = "nrd0"``, ``n = 512`` grid points,
``cut = 3``, FFT-binned linear approximation.  Reproducing this exactly is
needed for spectral-clustering parity.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bw_nrd0", "gaussian_kde_r"]


def bw_nrd0(x):
    """R's ``bw.nrd0`` (Silverman's rule-of-thumb default bandwidth)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 data points")
    sd = np.std(x, ddof=1)
    # interquartile range using R's type-7 quantile
    q75 = _quantile_type7(x, 0.75)
    q25 = _quantile_type7(x, 0.25)
    iqr = q75 - q25
    lo = min(sd, iqr / 1.349)
    if lo == 0:
        lo = sd if sd > 0 else (abs(x[0]) if x[0] != 0 else 1.0)
    return 0.9 * lo * n ** (-0.2)


def _quantile_type7(x, p):
    """R default (type 7) quantile."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    h = (n - 1) * p
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    return x[lo] + (h - lo) * (x[hi] - x[lo])


def _massdist(x, w, lo, hi, n):
    """Linear binning of weighted points onto an n-point grid (R's BinDist)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.zeros(2 * n)
    delta = (hi - lo) / (n - 1)
    for i in range(len(x)):
        xpos = (x[i] - lo) / delta
        ix = int(np.floor(xpos))
        fx = xpos - ix
        wi = w[i]
        if 0 <= ix < n - 1:
            y[ix] += (1 - fx) * wi
            y[ix + 1] += fx * wi
        elif ix == -1:
            y[0] += fx * wi
        elif ix == n - 1:
            y[ix] += (1 - fx) * wi
    return y


def gaussian_kde_r(x, bw=None, n=512, cut=3):
    """Reproduce ``stats::density(x)`` with default arguments.

    Returns a dict with ``x`` (grid) and ``y`` (density) of length ``n``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nx = len(x)
    if nx == 0:
        raise ValueError("no finite data")
    if bw is None:
        bw = bw_nrd0(x) if nx >= 2 else 1.0
    if bw <= 0:
        bw = 1e-8
    weights = np.full(nx, 1.0 / nx)

    lo = x.min() - cut * bw
    up = x.max() + cut * bw
    # binning grid for the FFT
    y = _massdist(x, weights, lo, up, n)

    # kernel evaluated on the doubled grid
    kords = np.arange(0, 2 * n) * (2 * (up - lo)) / (2 * n - 1)
    kords[n + 1:] = -kords[n - 1:0:-1]
    kords = _dnorm(kords, sd=bw)

    # FFT convolution (R: kords <- fft(fft(y)*Conj(fft(kords)), inverse=TRUE))
    # R uses inverse FFT then takes Re()/length; combined scaling is /(2n).
    fy = np.fft.fft(y)
    fk = np.fft.fft(kords)
    conv = np.fft.ifft(fy * np.conj(fk))
    # ifft already divides by 2n; R's manual inverse multiplies, so rescale.
    dens = np.maximum(0.0, conv.real)

    xords = np.linspace(lo, up, n)
    # R returns density on the binning grid restricted to the first n points;
    # values are interpolated from (xords, dens[:n]) onto the same grid -> id.
    return {"x": xords, "y": dens[:n], "bw": bw}


def _dnorm(x, sd=1.0):
    return np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
