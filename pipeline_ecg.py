"""
pipeline_ecg.py
Extrait de Testfixee.ipynb — fonctions de filtrage, détection R, QC et HRV.
Copier ce fichier dans C:\ecg\ à côté de dashboard.py
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import linregress, entropy as sp_entropy
import mne
import neurokit2 as nk

# ── Paramètres (identiques au notebook) ──
BP_L   = 0.5
BP_H   = 40.0
NOTCH  = 50.0
WIN_S  = 120
OVL_S  = 60
MIN_RP = 60
MIN_RR = 40
HR_LO  = 40
HR_HI  = 150

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ── Filtrage ECG ──
def ecg_filt(ecg, fs):
    fs = int(fs)
    nyq = fs / 2
    b, a = sp_signal.butter(4, [BP_L / nyq, BP_H / nyq], btype='bandpass')
    ef = sp_signal.filtfilt(b, a, ecg)
    if NOTCH < nyq:
        bn, an = sp_signal.iirnotch(NOTCH, Q=30, fs=fs)
        ef = sp_signal.filtfilt(bn, an, ef)
    try:
        return nk.ecg_clean(ef, sampling_rate=fs)
    except:
        return ef


# ── Détection pics R ──
def get_rp(ec, fs):
    _, i = nk.ecg_peaks(ec, sampling_rate=int(fs))
    return np.array(i.get("ECG_R_Peaks", []))


# ── Intervalles RR (ms), filtrés 300–2000 ms ──
def get_rr(rp, fs):
    if len(rp) < 2:
        return np.array([])
    rr = np.diff(rp) / fs * 1000
    return rr[(rr > 300) & (rr < 2000)]


# ── Fenêtres glissantes ──
def make_wins(ecg, fs):
    ws = int(WIN_S * fs)
    st = int((WIN_S - OVL_S) * fs)
    return [ecg[s:s + ws] for s in range(0, len(ecg) - ws + 1, st)]


# ── Contrôle qualité ──
def qc(rp, rr):
    if len(rp) < MIN_RP or len(rr) < MIN_RR:
        return False
    hr = 60000 / np.mean(rr) if np.mean(rr) > 0 else 0
    if not (HR_LO <= hr <= HR_HI):
        return False
    if len(rr) > 2 and np.sum(np.abs(np.diff(rr)) > 200) / len(rr) > 0.15:
        return False
    return True


# ── 41 features HRV ──
def hrv(rr):
    rr = np.asarray(rr, dtype=float)
    if rr.size < 12:
        return None
    dr = np.diff(rr)
    hr = 60000 / rr
    n50 = int(np.sum(np.abs(dr) > 50))
    n20 = int(np.sum(np.abs(dr) > 20))
    n30 = int(np.sum(np.abs(dr) > 30))
    n10 = int(np.sum(np.abs(dr) > 10))

    f = {
        "Mean_RR":   float(np.mean(rr)),
        "Median_RR": float(np.median(rr)),
        "SDNN":      float(np.std(rr, ddof=1)),
        "RMSSD":     float(np.sqrt(np.mean(dr**2))),
        "pNN50":     n50 / len(dr) * 100,
        "pNN30":     n30 / len(dr) * 100,
        "pNN20":     n20 / len(dr) * 100,
        "pNN10":     n10 / len(dr) * 100,
        "NN50":      float(n50),
        "NN20":      float(n20),
        "CV_RR":     float(np.std(rr, ddof=1) / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
        "Range_RR":  float(np.ptp(rr)),
        "IQR_RR":    float(np.percentile(rr, 75) - np.percentile(rr, 25)),
        "Skew_RR":   float(pd.Series(rr).skew()),
        "Kurt_RR":   float(pd.Series(rr).kurt()),
        "MAD_RR":    float(np.median(np.abs(rr - np.median(rr)))),
        "P20_RR":    float(np.percentile(rr, 20)),
        "P80_RR":    float(np.percentile(rr, 80)),
        "Mean_HR":   float(np.mean(hr)),
        "Std_HR":    float(np.std(hr, ddof=1)),
        "CV_HR":     float(np.std(hr, ddof=1) / np.mean(hr)) if np.mean(hr) > 0 else np.nan,
        "Min_HR":    float(np.min(hr)),
        "Max_HR":    float(np.max(hr)),
        "Range_HR":  float(np.ptp(hr)),
    }

    # Poincaré
    if len(rr) >= 4:
        x, y = rr[:-1], rr[1:]
        s1 = float(np.std(y - x, ddof=1) / np.sqrt(2))
        s2 = float(np.std(y + x, ddof=1) / np.sqrt(2))
        f.update({
            "SD1":      s1,
            "SD2":      s2,
            "SD1_SD2":  s1 / s2 if s2 > 0 else np.nan,
            "S_ellipse": float(np.pi * s1 * s2),
        })
    else:
        f.update({"SD1": np.nan, "SD2": np.nan, "SD1_SD2": np.nan, "S_ellipse": np.nan})

    # Tendance linéaire
    if len(rr) >= 4:
        sl, _, rv, _, _ = linregress(np.arange(len(rr)), rr)
        f.update({"RR_Slope": float(sl), "RR_R2": float(rv**2)})
    else:
        f.update({"RR_Slope": np.nan, "RR_R2": np.nan})

    # DFA alpha
    sc = [4, 8, 16, 32]
    y2 = np.cumsum(rr - np.mean(rr))
    fn, vs = [], []
    for n in sc:
        if len(y2) < 2 * n:
            continue
        ns = len(y2) // n
        if ns < 2:
            continue
        sg = y2[:ns * n].reshape(ns, n)
        xi = np.arange(n)
        fl = [np.sqrt(np.mean((s - np.polyval(np.polyfit(xi, s, 1), xi))**2)) for s in sg]
        fn.append(np.mean(fl))
        vs.append(n)
    f["DFA_alpha"] = float(linregress(np.log(vs), np.log(fn))[0]) if len(vs) >= 2 else np.nan

    # Spectral HRV
    rs = rr / 1000
    tr = np.cumsum(rs) - rs[0]
    if tr[-1] > 0:
        ti = np.arange(0, tr[-1], 0.25)
        if len(ti) >= 8:
            ri = np.interp(ti, tr, rs) - np.mean(rs)
            ff, px = sp_signal.welch(ri, fs=4.0, nperseg=min(256, len(ri)))
            def bp(lo, hi):
                m = (ff >= lo) & (ff < hi)
                return float(_trapz(px[m], ff[m])) if m.sum() >= 2 else np.nan
            vl, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
            tp = float(np.nansum([vl, lf, hf]))
            dn = (lf or 0) + (hf or 0)
            f.update({
                "VLF":  vl, "LF": lf, "HF": hf,
                "LF_HF": lf / hf if hf and hf > 0 else np.nan,
                "TP":   tp,
                "LFn":  lf / dn * 100 if dn > 0 else np.nan,
                "HFn":  hf / dn * 100 if dn > 0 else np.nan,
            })
        else:
            f.update({k: np.nan for k in ["VLF", "LF", "HF", "LF_HF", "TP", "LFn", "HFn"]})
    else:
        f.update({k: np.nan for k in ["VLF", "LF", "HF", "LF_HF", "TP", "LFn", "HFn"]})

    # Entropies
    try:
        f["ShEn"] = float(sp_entropy(np.histogram(rr, bins=10)[0] + 1e-10))
    except:
        f["ShEn"] = np.nan
    try:
        import antropy as ant
        f["ApEn"]   = float(ant.app_entropy(rr))
        f["SampEn"] = float(ant.sample_entropy(rr))
        f["PermEn"] = float(ant.perm_entropy(rr, normalize=True))
    except:
        f["ApEn"] = f["SampEn"] = f["PermEn"] = np.nan

    return f
