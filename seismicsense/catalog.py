"""
catalog.py - loads the NASA Apollo Passive Seismic Experiment catalog and turns
every event into numbers ("features") that a machine-learning model can learn from.

Data source: NASA PDS Geosciences Node - Apollo Seismic Event Catalog
(levent.1008weber = Y. Nakamura's long-period event catalog, edited by R. C. Weber).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
STATIONS = [12, 14, 15, 16]

# ---------------------------------------------------------------- class info
# Event-type codes in the post-2004 catalog column (T2), checked against the
# official counts in Nunn et al. (2020), Space Sci. Rev. 216:89, Table 8.
TYPE_CODES = {
    "A": "deep", "M": "deep",            # deep moonquakes (assigned / not assigned to a nest)
    "C": "meteoroid",                    # meteoroid impacts (1,743)
    "H": "shallow",                      # shallow moonquakes / HFT events (28)
    "S": "artificial", "L": "artificial" # Saturn S-IVB rocket stages / Lunar Module impacts (9)
}
# T, X, Z and blank = tentative / thermal / short-period / unclassified -> "unknown"

CLASS_INFO = {
    "deep": dict(
        name="Deep Moonquake", emoji="🌑", color="#8B6CFF",
        short="The Moon creaking ~900 km underground",
        story=("Earth's gravity squeezes and stretches the Moon a little every month - like "
               "squeezing a stress ball. Deep inside the Moon (about 700-1,200 km down) rock "
               "slips under that stress and makes a small, repeating quake."),
    ),
    "meteoroid": dict(
        name="Meteoroid Impact", emoji="☄️", color="#FF8A3D",
        short="A space rock smashed into the surface",
        story=("The Moon has no air to burn up space rocks, so even small meteoroids hit the "
               "ground at tens of km per second and ring the Moon like a bell."),
    ),
    "shallow": dict(
        name="Shallow Moonquake", emoji="⚡", color="#FF4D79",
        short="A rare, powerful quake near the surface",
        story=("The closest thing the Moon has to an earthquake: a sudden slip of the crust "
               "near the surface. Only 28 were recorded in 8 years, but they were the strongest."),
    ),
    "artificial": dict(
        name="Artificial Impact", emoji="🚀", color="#2EC4B6",
        short="An Apollo rocket stage crashed on purpose",
        story=("NASA deliberately crashed used Saturn V upper stages and lunar module ascent "
               "stages into the Moon. Their exact time and place were known, so scientists used "
               "them to calibrate the seismometers."),
    ),
    "unknown": dict(
        name="Unclassified", emoji="❔", color="#9AA0A6",
        short="NASA scientists never decided what this was",
        story="The original scientists could not confidently decide what caused this signal.",
    ),
}
NATURAL = ["deep", "meteoroid", "shallow"]  # classes the ML model learns

# Apollo landing sites (well-known coordinates) and seismometer operating periods
APOLLO_SITES = pd.DataFrame([
    dict(site="Apollo 11", lat=0.67, lon=23.47, seismometer=False, start="1969-07-21", end="1969-08-27"),
    dict(site="Apollo 12", lat=-3.01, lon=-23.42, seismometer=True, start="1969-11-19", end="1977-09-30"),
    dict(site="Apollo 14", lat=-3.65, lon=-17.47, seismometer=True, start="1971-02-05", end="1977-09-30"),
    dict(site="Apollo 15", lat=26.13, lon=3.63, seismometer=True, start="1971-07-31", end="1977-09-30"),
    dict(site="Apollo 16", lat=-8.97, lon=15.50, seismometer=True, start="1972-04-21", end="1977-09-30"),
    dict(site="Apollo 17", lat=20.19, lon=30.77, seismometer=False, start="1972-12-11", end="1977-09-30"),
])
STATION_START = {12: "1969-11-19", 14: "1971-02-05", 15: "1971-07-31", 16: "1972-04-21"}


# ---------------------------------------------------------------- loading
def _parse_time(df: pd.DataFrame) -> pd.Series:
    yy = df["Y"].astype(int)
    year = 1900 + yy
    hhmm = df["S"].fillna("0000").str.zfill(4)
    hh = pd.to_numeric(hhmm.str[:2], errors="coerce").fillna(0).astype(int)
    mm = pd.to_numeric(hhmm.str[2:], errors="coerce").fillna(0).astype(int)
    base = pd.to_datetime(year.astype(str) + "-01-01")
    return base + pd.to_timedelta(df["JD"].astype(int) - 1, unit="D") + pd.to_timedelta(hh * 60 + mm, unit="m")


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "levent.1008weber.csv"
    raw = pd.read_csv(path, dtype=str)
    df = pd.DataFrame(index=raw.index)
    df["event_id"] = [f"EV-{i:05d}" for i in range(1, len(raw) + 1)]
    df["time"] = _parse_time(raw)
    df["nasa_label"] = raw["T2"].map(TYPE_CODES).fillna("unknown")
    df["nasa_code"] = raw["T2"].fillna("")
    df["nest"] = pd.to_numeric(raw["N2"], errors="coerce")
    df["comment"] = raw["COMMENTS"].fillna("")
    return pd.concat([df, raw], axis=1)


# ---------------------------------------------------------------- physics helpers
def moon_geometry(jd: np.ndarray):
    """Low-precision lunar ephemeris (Meeus, Astronomical Algorithms ch.47).
    Returns Earth-Moon distance (km) and the mean elongation / anomaly / latitude angles."""
    T = (jd - 2451545.0) / 36525.0
    D = np.radians((297.8501921 + 445267.1114034 * T) % 360)
    M = np.radians((357.5291092 + 35999.0502909 * T) % 360)
    Mp = np.radians((134.9633964 + 477198.8675055 * T) % 360)
    F = np.radians((93.2720950 + 483202.0175233 * T) % 360)
    r = (385000.56 - 20905.355 * np.cos(Mp) - 3699.111 * np.cos(2 * D - Mp) - 2955.968 * np.cos(2 * D)
         - 569.925 * np.cos(2 * Mp) + 48.888 * np.cos(M) - 3.149 * np.cos(2 * F)
         + 246.158 * np.cos(2 * D - 2 * Mp) - 152.138 * np.cos(2 * D - M - Mp)
         - 170.733 * np.cos(2 * D + Mp) - 204.586 * np.cos(2 * D - M) - 129.620 * np.cos(M - Mp)
         + 108.743 * np.cos(D) + 104.755 * np.cos(M + Mp))
    return r, D, Mp, F


def _days(t: pd.Series) -> np.ndarray:
    return ((t - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")).to_numpy(dtype=float)


# ---------------------------------------------------------------- features
def build_features(cat: pd.DataFrame) -> pd.DataFrame:
    """Turn catalog rows into ML features. No label columns are used."""
    X = pd.DataFrame(index=cat.index)
    t = cat["time"]
    days = _days(t)

    # duration of the visible signal
    s = cat["S"].fillna("0000").str.zfill(4)
    e = cat["E"].fillna("9999").str.zfill(4)
    smin = pd.to_numeric(s.str[:2], errors="coerce") * 60 + pd.to_numeric(s.str[2:], errors="coerce")
    emin = pd.to_numeric(e.str[:2], errors="coerce") * 60 + pd.to_numeric(e.str[2:], errors="coerce")
    dur = (emin - smin).to_numpy(dtype=float)
    dur = np.where(dur < 0, dur + 1440, dur)
    dur = np.where(e.to_numpy() == "9999", np.nan, dur)
    X["duration_min"] = dur
    X["log_duration"] = np.log10(np.nan_to_num(dur, nan=np.nanmedian(dur)) + 1)

    # shaking strength at each station
    for i, st in enumerate(STATIONS, start=1):
        a = pd.to_numeric(cat[f"A{i}"], errors="coerce")
        a = a.where(a >= 0)
        X[f"amp_{st}"] = a
        X[f"logamp_{st}"] = np.log10(a + 0.1)
        X[f"seen_{st}"] = cat[f"A{i}"].notna().astype(int)
        X[f"P_{st}"] = pd.to_numeric(cat[f"P{i}"], errors="coerce")
        X[f"Q_{st}"] = pd.to_numeric(cat[f"Q{i}"], errors="coerce")
        X[f"operating_{st}"] = (t >= pd.Timestamp(STATION_START[st])).astype(int)
    la = X[[f"logamp_{s}" for s in STATIONS]]
    X["logamp_max"] = la.max(axis=1)
    X["logamp_mean"] = la.mean(axis=1)
    X["logamp_std"] = la.std(axis=1)
    X["n_stations"] = X[[f"seen_{s}" for s in STATIONS]].sum(axis=1)
    X["n_operating"] = X[[f"operating_{s}" for s in STATIONS]].sum(axis=1)
    X["seen_fraction"] = X["n_stations"] / X["n_operating"].clip(lower=1)
    X["strength_x_duration"] = X["logamp_max"].fillna(-1) - 0.8 * X["log_duration"]
    for a, b in [(12, 14), (12, 15), (12, 16), (14, 15), (14, 16), (15, 16)]:
        X[f"ratio_{a}_{b}"] = X[f"logamp_{a}"] - X[f"logamp_{b}"]

    # signal quality fields recorded by the analysts
    X["grade"] = cat["Grade"].map({"A": 3, "B": 2, "C": 1})
    X["traces"] = pd.to_numeric(cat["Traces"], errors="coerce")
    for c in ["12LPX", "12LPY", "12LPZ", "14SPZ", "14LPX", "14LPY", "14LPZ", "15SPZ",
              "15LPX", "15LPY", "15LPZ", "16SPZ", "16LPX", "16LPY", "16LPZ"]:
        X["ch_" + c] = pd.to_numeric(cat[c], errors="coerce").fillna(0)

    # tides: where were the Earth and Moon in their orbits?
    jd = days + 2440587.5
    r, D, Mp, F = moon_geometry(jd)
    r2, *_ = moon_geometry(jd + 0.25)
    X["earth_moon_km"] = r
    X["distance_rate"] = (r2 - r) * 4
    for n, ang in [("synodic", D), ("anomalistic", Mp), ("draconic", F)]:
        X[f"{n}_sin"] = np.sin(ang)
        X[f"{n}_cos"] = np.cos(ang)
    hour = t.dt.hour + t.dt.minute / 60
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    doy = t.dt.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    # how busy was the Moon around this event? (label-free context)
    hrs = days * 24
    order = np.argsort(hrs)
    sh = hrs[order]
    for w in (6, 24, 72):
        cnt = np.searchsorted(sh, sh + w) - np.searchsorted(sh, sh - w) - 1
        out = np.empty_like(cnt)
        out[order] = cnt
        X[f"neighbors_{w}h"] = out

    X = pd.concat([X, _recurrence(X, days), _fingerprint(X, days)], axis=1)
    return X


def _recurrence(X: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    """Deep-moonquake nests re-fire every anomalistic month (27.55 d) with the same
    'station fingerprint'. For each event, how well does a look-alike appear 1-3 months away?"""
    P = 27.5546
    order = np.argsort(days)
    ts = days[order]
    L = X[[f"logamp_{s}" for s in STATIONS]].to_numpy()[order]
    dur = X["log_duration"].to_numpy()[order]
    out = {}
    for k in (1, 2, 3):
        best = np.full(len(ts), 9.0)
        cnt = np.zeros(len(ts))
        for sgn in (1, -1):
            lo = np.searchsorted(ts, ts + sgn * k * P - 1.5)
            hi = np.searchsorted(ts, ts + sgn * k * P + 1.5)
            for i in range(len(ts)):
                a, b = lo[i], hi[i]
                if b <= a:
                    continue
                cnt[i] += b - a
                diff = L[a:b] - L[i]
                ok = ~np.isnan(diff)
                nc = ok.sum(1)
                dd = np.where(ok, diff ** 2, 0).sum(1)
                mismatch = (np.isnan(L[a:b]) != np.isnan(L[i])).sum(1)
                score = np.sqrt(dd / np.maximum(nc, 1)) + 0.5 * mismatch + 0.3 * np.abs(dur[a:b] - dur[i])
                score = np.where(nc > 0, score, 9.0)
                best[i] = min(best[i], score.min())
        o1, o2 = np.empty(len(ts)), np.empty(len(ts))
        o1[order], o2[order] = best, cnt
        out[f"recur_match_{k}m"], out[f"recur_count_{k}m"] = o1, o2
    R = pd.DataFrame(out, index=X.index)
    R["recur_best"] = R[[f"recur_match_{k}m" for k in (1, 2, 3)]].min(axis=1)
    R["recur_n_good"] = (R[[f"recur_match_{k}m" for k in (1, 2, 3)]] < 0.35).sum(axis=1)
    return R


def _fingerprint(X: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    """Look-alike events anywhere in the 8-year record (same station pattern + tidal phase)."""
    from sklearn.neighbors import NearestNeighbors
    ph = 2 * np.pi * (days % 27.5546) / 27.5546
    L = X[[f"logamp_{s}" for s in STATIONS]].fillna(-1.5).to_numpy()
    S = X[[f"seen_{s}" for s in STATIONS]].to_numpy() * 1.0
    du = X["log_duration"].to_numpy()[:, None]
    out = pd.DataFrame(index=X.index)
    for name, w in (("tidal", 1.2), ("any", 0.0)):
        F = np.hstack([L, S * 0.8, du * 0.6, w * np.sin(ph)[:, None], w * np.cos(ph)[:, None]])
        dist, idx = NearestNeighbors(n_neighbors=16).fit(F).kneighbors(F)
        dist, idx = dist[:, 1:], idx[:, 1:]
        gap = np.abs(days[idx] - days[:, None])
        out[f"fp_{name}_d1"] = dist[:, 0]
        out[f"fp_{name}_d5"] = dist[:, 4]
        out[f"fp_{name}_d15"] = dist[:, 14]
        out[f"fp_{name}_far"] = (gap > 5).mean(axis=1)
    return out


# ---------------------------------------------------------------- locations
def load_locations() -> dict[str, pd.DataFrame]:
    """Published source locations (NASA PDS) for the map."""
    dm = pd.read_csv(RAW / "nakamura_2005_dm_locations.csv")
    dm = dm.rename(columns={"A": "nest", "Lat": "lat", "Long": "lon", "Depth": "depth_km"})
    ai = pd.read_csv(RAW / "nakamura_1983_ai_locations.csv").rename(columns={"Lat": "lat", "Long": "lon", "AI": "name"})
    ai["time"] = (pd.to_datetime((1900 + ai["Y"]).astype(str) + "-01-01") + pd.to_timedelta(ai["JD"] - 1, unit="D")
                  + pd.to_timedelta(ai["Hour"] * 3600 + ai["Min"] * 60 + ai["Sec"], unit="s"))
    # Apollo 16 S-IVB: tracking was lost, location is NASA's estimate (~1.3N, 23.8W); time from the catalog
    ai = pd.concat([ai, pd.DataFrame([dict(name="16 S-IVB", lat=1.3, lon=-23.8,
                                           time=pd.Timestamp("1972-04-19 21:03"))])], ignore_index=True)
    ai["name"] = ai["name"].str.replace("LM", "Lunar Module").str.replace("S-IVB", "Saturn V S-IVB stage")
    ai["name"] = "Apollo " + ai["name"]
    sm = pd.read_csv(RAW / "nakamura_1979_sm_locations.csv").rename(columns={"Lat": "lat", "Long": "lon"})
    sm["time"] = (pd.to_datetime(sm["Year"].astype(str) + "-01-01") + pd.to_timedelta(sm["Day"] - 1, unit="D")
                  + pd.to_timedelta(sm["H"] * 3600 + sm["M"] * 60 + sm["S"], unit="s"))
    g = pd.read_csv(RAW / "gagnepian_2006_catalog.csv", dtype={"Date": str})
    met = g[g["Type"] == "M"].rename(columns={"Lat": "lat", "Long": "lon"}).copy()
    met["time"] = pd.to_datetime("19" + met["Date"].str.zfill(10), format="%Y%m%d%H%M")
    return dict(deep_nests=dm, artificial=ai, shallow=sm, meteoroid=met[["lat", "lon", "time"]])
