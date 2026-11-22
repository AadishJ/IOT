"""
iot_sim.py - Python port of the firmware's simulator and policies (main.cpp), line for line.

Same model as the ESP8266 hardware-in-the-loop sketch:
  * virtual 10 mAh battery (3.0-4.2 V) = 30 F capacitor, charged by a solar panel (ISC x irradiance)
  * energy cost per sample / per transmission, plus sleep power
  * four policies: fixed_rate, signal_only, energy_only, combined_v4
  * gateway reconstruction by linear interpolation between transmitted samples -> NRMSE / NMAE
All constants are in Cfg (defaults = the firmware's), so sensitivity sweeps are one keyword away.

The simulation arithmetic runs in float32, like the ESP8266, because the policies compare quantised
sensor values against thresholds such as ">= 0.3": float32 and float64 break those ties differently.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

METHODS = ["fixed_rate", "signal_only", "energy_only", "combined_v4"]
WEATHERS = ["sunny", "sunny_to_cloudy", "cloudy_to_sunny", "cloudy"]
_W_SUNNY = {0: (True, True), 1: (True, False), 2: (False, True), 3: (False, False)}


@dataclass(frozen=True)
class Cfg:
    dt: float = 30.0            # simulated seconds per tick (one row)
    days: int = 2
    batt_mah: float = 10.0
    v_empty: float = 3.0
    v_full: float = 4.2
    v_alive: float = 3.05
    soc0: float = 0.5
    isc: float = 6e-3           # A, panel short-circuit current at full sun
    p_sleep: float = 1.5e-4     # W
    e_sample: float = 0.14      # J per sample
    e_tx: float = 0.55          # J per transmission
    sunrise: float = 6.5
    sunset: float = 17.5
    fixed_interval: int = 10
    max_stale: int = 120        # combined_v4 mandatory row spacing
    soc_floor: float = 0.15
    soc_ref: float = 0.6
    tx_frac: float = 0.02
    max_interval: int = 30

    @property
    def n(self):
        return int(self.days * 86400 / self.dt)

    @property
    def cap_f(self):
        return self.batt_mah * 3.6 / (self.v_full - self.v_empty)


DEFAULT_CFG = Cfg()
RING = 80


def quantize(trace):
    """x100 rounded half away from zero (C lroundf), as the firmware stores its ground truth."""
    x = np.asarray(trace, dtype=np.float32) * np.float32(100.0)
    return (np.sign(x) * np.floor(np.abs(x).astype(np.float64) + 0.5)).astype(np.int64)


# ------------------------------------------------------------------ weather
def cloud_path(seed, n):
    """AR(1) cloud process (same recursion as the firmware: cloudX = 0.97*cloudX + 0.22*randn)."""
    eps = np.random.default_rng(seed).standard_normal(n)
    x = np.empty(n)
    c = 0.0
    for i in range(n):
        c = 0.97 * c + 0.22 * eps[i]
        x[i] = c
    return x


def irradiance(weather_idx, cx, cfg=DEFAULT_CFG):
    """Irradiance 0..1 per tick for a weather scenario, given a cloud path."""
    n = len(cx)
    t = np.arange(n) * cfg.dt
    day = np.minimum((t / 86400.0).astype(int), 1)
    sunny = np.array(_W_SUNNY[weather_idx])[day]
    cloud = np.where(sunny, np.clip(0.97 + 0.02 * cx, 0.88, 1.0),
                     np.clip(0.15 * np.exp(0.55 * cx), 0.03, 0.40))
    hour = np.mod(t / 3600.0, 24.0)
    x = (hour - cfg.sunrise) / (cfg.sunset - cfg.sunrise)
    clear = np.where((x <= 0) | (x >= 1), 0.0, np.sin(np.pi * np.clip(x, 0, 1)) ** 1.2)
    return clear * cloud


def build_all_irradiance(seeds, cfg=DEFAULT_CFG):
    """{(weather_idx, seed): irradiance array}. Every weather shares the same cloud path for a seed."""
    out = {}
    for s in seeds:
        cx = cloud_path(s, cfg.n)
        for w in range(4):
            out[(w, s)] = irradiance(w, cx, cfg)
    return out


# ------------------------------------------------------------------ float32 helpers
F = np.float32


def _clamp(x, a, b):
    return a if x < a else (b if x > b else x)


class _C32:
    """The Cfg constants as float32, evaluated in the same order as the C++ constant expressions."""
    def __init__(self, c):
        self.dt = F(c.dt)
        self.v_empty, self.v_full, self.v_alive = F(c.v_empty), F(c.v_full), F(c.v_alive)
        self.span_e = self.v_full - self.v_empty
        self.soc0 = F(c.soc0)
        self.isc, self.p_sleep = F(c.isc), F(c.p_sleep)
        self.e_sample, self.e_tx = F(c.e_sample), F(c.e_tx)
        self.cap_f = F(c.batt_mah) * F(3.6) / self.span_e
        self.soc_floor, self.soc_ref, self.tx_frac = F(c.soc_floor), F(c.soc_ref), F(c.tx_frac)
        self.fixed_interval, self.max_stale, self.max_interval = c.fixed_interval, c.max_stale, c.max_interval
        self.u1, self.u2, self.u3, self.u4 = F(0.025) * F(0.2), F(0.08) * F(0.2), F(0.2) * F(0.2), F(0.5) * F(0.2)


_ZERO, _ONE, _HALF = F(0.0), F(1.0), F(0.5)
_F100, _F30, _F10, _F5 = F(100.0), F(30.0), F(10.0), F(1.0) * F(5.0)
_C08, _C015, _C03, _C001 = F(0.8), F(0.15), F(0.3), F(0.01)
_T1800, _EPS9, _EPS3, _C25, _C20, _F4, _C03b = F(1800.0), F(1e-9), F(1e-3), F(2.5), F(2.0), F(4.0), F(0.3)


def _median32(a):
    b = sorted(a)
    n = len(b)
    return b[n // 2] if n & 1 else _HALF * (b[n // 2 - 1] + b[n // 2])


# ------------------------------------------------------------------ policies
class _Base:
    def __init__(self, n, c):
        self.n, self.c = n, c
        self.next_index = 0
        self.has_tx = False
        self.last_tx = _ZERO


class FixedRate(_Base):
    def step(self, i, v, t, soc):
        self.next_index = i + self.c.fixed_interval
        return True, True


class SignalOnly(_Base):
    def __init__(self, n, c):
        super().__init__(n, c)
        self.obs = []

    def step(self, i, v, t, soc):
        o = self.obs
        if len(o) < 10:
            o.append(v)
        else:
            o.pop(0)
            o.append(v)
        raw = _ZERO
        if len(o) > 1:
            for k in range(1, len(o)):
                raw = raw + abs(o[k] - o[k - 1])
            raw = raw / F(len(o) - 1)
        interval = 1 if raw >= _C08 else (30 if raw <= _C015 else 10)
        send = (not self.has_tx) or abs(v - self.last_tx) >= _C03
        if send:
            self.last_tx, self.has_tx = v, True
        self.next_index = i + interval
        return True, bool(send)


class EnergyOnly(_Base):
    def step(self, i, v, t, soc):
        b = soc * _F100
        interval = 30 if b < _F30 else (10 if b < F(70.0) else 5)
        send = (not self.has_tx) or abs(v - self.last_tx) >= _C03
        if send:
            self.last_tx, self.has_tx = v, True
        self.next_index = i + interval
        return True, bool(send)


class CombinedV4(_Base):
    def __init__(self, n, c):
        super().__init__(n, c)
        self.have_lh = False
        self.low = self.high = _ZERO
        self.rt = [_ZERO] * RING
        self.rv = [_ZERO] * RING
        self.rs = [_ZERO] * RING
        self.rpos = 0
        self.rn = 0

    def _is_mand(self, i):
        return (i % self.c.max_stale == 0 and i < self.n - 1) or i == self.n - 1

    def _nxt_mand(self, i):
        k = (i // self.c.max_stale + 1) * self.c.max_stale
        if k < self.n - 1:
            return k
        return self.n - 1 if i < self.n - 1 else self.n

    def _ring(self, k):                       # k = 0 newest
        return (self.rpos - 1 - k + 2 * RING) % RING

    def step(self, i, v, t, soc):
        c = self.c
        mand = self._is_mand(i)
        if (not mand) and soc < c.soc_floor:
            self.next_index = self._nxt_mand(i)
            return False, False
        if not self.have_lh:
            self.low = self.high = v
            self.have_lh = True
        else:
            self.low = min(self.low, v)
            self.high = max(self.high, v)
        scale = max(max(self.high - self.low, _EPS3 * max(abs(v), _ONE)), _EPS9)
        var = pred = _ZERO
        rt, rv, rs, rn = self.rt, self.rv, self.rs, self.rn
        if rn >= 2:
            m = min(rn, 10)
            rates, dts = [], []
            for k in range(m - 1, 0, -1):
                a, b = self._ring(k), self._ring(k - 1)
                dtt = max(rt[b] - rt[a], _EPS9)
                rates.append(abs(rv[b] - rv[a]) / dtt)
                dts.append(dtt)
            var = _median32(rates) * _median32(dts) / scale
            n0, n1 = self._ring(0), self._ring(1)
            slope = (rv[n0] - rv[n1]) / max(rt[n0] - rt[n1], _EPS9)
            st = _clamp(slope * (t - rt[n0]), -_HALF * scale, _HALF * scale)
            pred = abs(v - (rv[n0] + st)) / scale
        urg = max(var, pred)
        desired = 1 if urg >= c.u4 else 3 if urg >= c.u3 else 6 if urg >= c.u2 else 12 if urg >= c.u1 else c.max_interval

        r = soc / c.soc_ref
        for k in range(rn):                    # battery rising over the last >=30 min => surplus
            ix = self._ring(k)
            if t - rt[ix] >= _T1800:
                if soc - rs[ix] > _C001:
                    r = r + _C03b
                break
        flr = 1 if r >= _ONE else 3 if r >= F(0.8) else 6 if r >= F(0.6) else 12 if r >= F(0.4) else c.max_interval
        if urg >= c.u4 and r >= F(0.6):
            interval = max(desired, min(flr, 3))
        else:
            interval = max(desired, flr)
        interval = max(1, min(c.max_interval, interval))

        if mand:
            send = True
        else:
            thr = c.tx_frac * scale * _clamp(_ONE / max(min(r, _ONE), _EPS3), _ONE, _F4)
            if urg >= c.u2:
                thr = thr * _HALF
            send = bool(((not self.has_tx) or abs(v - self.last_tx) >= thr) and soc >= c.soc_floor)
        if send:
            self.last_tx, self.has_tx = v, True
        rt[self.rpos], rv[self.rpos], rs[self.rpos] = t, v, soc
        self.rpos = (self.rpos + 1) % RING
        if self.rn < RING:
            self.rn += 1
        self.next_index = min(i + interval, self._nxt_mand(i))
        return True, send


POLICIES = {"fixed_rate": FixedRate, "signal_only": SignalOnly,
            "energy_only": EnergyOnly, "combined_v4": CombinedV4}


# ------------------------------------------------------------------ one run
def simulate(trace, G, method, cfg=DEFAULT_CFG):
    """Run one policy on one trace under one irradiance array (float32 arithmetic). Returns a dict of metrics."""
    n = len(trace)
    c = _C32(cfg)
    q = quantize(trace)
    vals = [F(int(x)) / _F100 for x in q]
    Gf = [F(g) for g in G]
    pol = POLICIES[method](n, c)
    dt = c.dt
    V = c.v_empty + c.span_e * c.soc0
    e_in = e_clip = e_load = _ZERO
    alive_ticks = brownouts = n_acq = 0
    was_alive = True
    min_soc = _ONE
    tx_idx = []
    for i in range(n):
        t = F(i) * dt
        soc = _clamp((V - c.v_empty) / c.span_e, _ZERO, _ONE)
        alive = V > c.v_alive
        acq = send = False
        if alive and i >= pol.next_index:
            acq, send = pol.step(i, vals[i], t, soc)
        i_chg = c.isc * Gf[i]
        e_sleep = c.p_sleep * dt if alive else _ZERO
        e_event = (c.e_sample + (c.e_tx if send else _ZERO)) if acq else _ZERO
        e_in = e_in + V * i_chg * dt
        e_load = e_load + (e_sleep + e_event)
        V = V + (i_chg * dt - (e_sleep + e_event) / max(V, _C25)) / c.cap_f
        if V > c.v_full:
            e_clip = e_clip + (V - c.v_full) * c.cap_f * c.v_full
            V = c.v_full
        if V < _C20:
            V = _C20
        if acq:
            n_acq += 1
            if send:
                tx_idx.append(i)
        if alive:
            alive_ticks += 1
        alive_now = V > c.v_alive
        if was_alive and not alive_now:
            brownouts += 1
        was_alive = alive_now
        soc_now = _clamp((V - c.v_empty) / c.span_e, _ZERO, _ONE)
        if soc_now < min_soc:
            min_soc = soc_now

    nrmse, nmae, max_gap = reconstruction(q, tx_idx, cfg.dt)
    soc_end = _clamp((V - c.v_empty) / c.span_e, _ZERO, _ONE)
    return {"nrmse": nrmse, "nmae": nmae, "samples": n_acq, "tx": len(tx_idx),
            "load_J": float(e_load), "harvest_J": float(e_in), "clipped_J": float(e_clip),
            "uptime": 100.0 * alive_ticks / n, "brownouts": brownouts,
            "minSOC": 100.0 * float(min_soc), "finalSOC": 100.0 * float(soc_end),
            "max_gap_min": max_gap}


def reconstruction(q, tx_idx, dt):
    """Gateway view: linear interpolation between transmitted samples, hold before first / after last."""
    n = len(q)
    if not tx_idx:
        return float("nan"), float("nan"), n * dt / 60.0
    q = q.astype(float)
    idx = np.asarray(tx_idx)
    pred = np.interp(np.arange(n), idx, q[idx])      # left/right of the range = first/last transmitted value
    err = (pred - q) / 100.0
    span = max(float(np.float32(q.max() - q.min()) / np.float32(100.0)), 1e-12)
    nrmse = math.sqrt(float(np.mean(err * err))) / span * 100.0
    nmae = float(np.mean(np.abs(err))) / span * 100.0
    gaps = [idx[0], n - 1 - idx[-1]] + (list(np.diff(idx)) if len(idx) > 1 else [])
    return nrmse, nmae, max(gaps) * dt / 60.0