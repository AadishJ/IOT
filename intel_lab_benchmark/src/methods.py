"""Causal sampling policies for real sensor time series.

The combined policy uses normalized prediction error, signal variability,
and remaining abstract budget. Budget values are illustrative units, not joules.
"""
import numpy as np
import pandas as pd


def _trace(df, acquired, transmitted, interval, policy, budget_trace=None):
    out = df.copy()
    out["acquired"] = acquired
    out["transmit"] = transmitted
    out["sampling_interval"] = interval
    out["method"] = policy
    if budget_trace is not None:
        out["energy_budget_remaining"] = budget_trace
    return out


def fixed_rate(df: pd.DataFrame, interval: int = 10) -> pd.DataFrame:
    if interval < 1:
        raise ValueError("interval must be >= 1")
    n = len(df)
    acquired = np.zeros(n, dtype=bool)
    acquired[::interval] = True
    return _trace(df, acquired, acquired.copy(),
                  np.where(acquired, interval, np.nan), "fixed_rate")


def _robust_scale(values):
    """Robust scale based on IQR, with a safe fallback for constant signals."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return 1.0
    q25, q75 = np.percentile(a, [25, 75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(a))
    if scale <= 1e-12:
        scale = max(float(np.median(np.abs(a))) * 0.01, 1e-6)
    return scale


def _signal_features(observed, current, window=10):
    """Causal, scale-normalized variability and one-step prediction error."""
    history = np.asarray(observed[-max(window, 30):], dtype=float)
    current = float(current)
    combined = np.append(history, current)
    scale = _robust_scale(combined)

    if history.size == 0:
        return 0.0, 0.0, scale

    recent = np.asarray(observed[-window:], dtype=float)
    if recent.size >= 2:
        variability = float(np.median(np.abs(np.diff(recent))) / scale)
    else:
        variability = 0.0

    # Predict current value from prior acquired observations only.
    if history.size >= 2:
        # Local linear extrapolation, clipped to avoid unstable long extrapolation.
        slope = float(history[-1] - history[-2])
        predicted = float(history[-1] + np.clip(slope, -scale, scale))
        prediction_error = abs(current - predicted) / scale
    else:
        prediction_error = 0.0

    return variability, prediction_error, scale


def adaptive_policy(
    df: pd.DataFrame,
    mode: str,
    min_interval: int = 1,
    max_interval: int = 30,
    window: int = 10,
    stable_threshold: float = 0.015,
    rapid_threshold: float = 0.08,
    initial_energy: float = 100.0,
    sample_cost: float = 0.02,
    tx_cost: float = 0.08,
    tx_delta: float = 0.3,
) -> pd.DataFrame:
    """Modes: signal_only, energy_only, combined.

    Decisions use only acquired history and the current acquired reading.
    ``combined`` chooses the next interval using both normalized signal
    behavior and remaining budget. Energy is an abstract model.
    """
    if mode not in {"signal_only", "energy_only", "combined"}:
        raise ValueError("unknown mode")
    if min_interval < 1 or max_interval < min_interval:
        raise ValueError("invalid sampling interval bounds")
    if initial_energy <= 0:
        raise ValueError("initial_energy must be > 0")

    n = len(df)
    values = df["value"].to_numpy(dtype=float)
    acquired = np.zeros(n, dtype=bool)
    transmitted = np.zeros(n, dtype=bool)
    intervals = np.full(n, np.nan)
    budget_trace = np.full(n, np.nan)

    budget = float(initial_energy)
    next_i = 0
    observed = []
    last_tx = None
    last_tx_scale = None

    for i in range(n):
        if i < next_i:
            budget_trace[i] = budget
            continue

        current = float(values[i])
        acquired[i] = True
        variability, pred_error, scale = _signal_features(observed, current, window)
        energy_fraction = float(np.clip(budget / initial_energy, 0.0, 1.0))

        if mode == "signal_only":
            # Preserve a simple raw-change baseline, as in the previous version.
            recent = np.asarray((observed + [current])[-window:], dtype=float)
            raw_var = float(np.mean(np.abs(np.diff(recent)))) if recent.size > 1 else 0.0
            interval = (min_interval if raw_var >= 0.8 else
                        max_interval if raw_var <= 0.15 else 10)
        elif mode == "energy_only":
            interval = (max_interval if budget < 30 else
                        10 if budget < 70 else 5)
        else:
            # Step 1: choose signal urgency from normalized variability AND
            # how badly the latest reading violated a causal prediction.
            urgency = max(variability, pred_error)
            if urgency >= 0.50:
                desired = min_interval
            elif urgency >= 0.20:
                desired = max(min_interval, 3)
            elif urgency >= 0.08:
                desired = max(min_interval, 6)
            elif urgency >= 0.025:
                desired = max(min_interval, 12)
            else:
                desired = max_interval

            # Step 2: budget pressure imposes a progressively longer interval.
            # High uncertainty can partially override, but not ignore, pressure.
            if energy_fraction < 0.10:
                budget_floor = max_interval
            elif energy_fraction < 0.25:
                budget_floor = max(18, min_interval)
            elif energy_fraction < 0.45:
                budget_floor = max(10, min_interval)
            elif energy_fraction < 0.70:
                budget_floor = max(5, min_interval)
            else:
                budget_floor = min_interval

            if urgency >= 0.50 and energy_fraction >= 0.20:
                interval = max(desired, min(budget_floor, 3))
            else:
                interval = max(desired, budget_floor)

        interval = int(np.clip(interval, min_interval, max_interval))
        intervals[i] = interval

        # Sensing has a cost whether or not the reading is transmitted.
        budget = max(0.0, budget - sample_cost)

        if mode == "combined":
            # Adaptive threshold uses the sensor's robust scale, avoiding a
            # single absolute delta behaving differently for temperature/light.
            tx_threshold = max(0.02 * scale, 1e-9)
        else:
            tx_threshold = tx_delta

        should_transmit = (
            last_tx is None or abs(current - last_tx) >= tx_threshold
        )
        if should_transmit and budget >= tx_cost:
            transmitted[i] = True
            budget = max(0.0, budget - tx_cost)
            last_tx = current
            last_tx_scale = scale

        budget_trace[i] = budget
        observed.append(current)
        next_i = i + interval

    return _trace(df, acquired, transmitted, intervals, mode, budget_trace)


def acquired_rows(trace):
    return trace.loc[trace["acquired"]].copy()


def transmitted_rows(trace):
    return trace.loc[trace["transmit"]].copy()
