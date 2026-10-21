"""Combined V4: adaptive sensing/transmission with a funded freshness schedule.

* A mandatory-transmission schedule is built explicitly: rows 0, S, 2S, ... and
  the final row n-1 (no duplicates), so consecutive mandatory rows are <= S apart.
* The energy for every future mandatory event (sample + tx) is reserved up
  front. Optional sampling/transmission may never dip into that reserve.
* If initial_energy cannot fund the whole schedule, the run is flagged
  infeasible (trace.attrs["freshness_feasible"] = False, plus a warning).
* Between mandatory rows the policy stays adaptive (variability, prediction
  error, discretionary budget pacing).

Energy values are abstract simulation units, not joules.
"""
import warnings
import numpy as np

BASE_URGENCY = (0.025, 0.08, 0.2, 0.5)


def mandatory_schedule(n, max_stale_rows):
    """Rows that must be acquired AND transmitted. Includes 0 and n-1."""
    if n <= 0:
        return np.array([], dtype=int)
    rows = list(range(0, n - 1, int(max_stale_rows)))
    if not rows or rows[-1] != n - 1:
        rows.append(n - 1)
    return np.asarray(rows, dtype=int)


def combined_v4(df, initial_energy=100.0, sample_cost=0.02, tx_cost=0.08,
                horizon=None, min_interval=1, max_interval=30, window=10,
                tx_frac=0.02, urg_scale=0.2, max_stale_rows=120):
    n = len(df)
    if initial_energy < 0 or sample_cost < 0 or tx_cost < 0:
        raise ValueError("Energy and costs must be non-negative.")
    if min_interval < 1 or max_interval < min_interval:
        raise ValueError("Invalid sampling interval bounds.")
    if max_stale_rows < 1 or window < 2:
        raise ValueError("max_stale_rows must be >= 1 and window >= 2.")

    out = df.copy()
    acquired = np.zeros(n, bool)
    transmitted = np.zeros(n, bool)
    intervals = np.full(n, np.nan)
    budget_trace = np.full(n, float(initial_energy))

    sched = mandatory_schedule(n, max_stale_rows)
    is_mand = np.zeros(n, bool)
    is_mand[sched] = True
    mc = sample_cost + tx_cost                     # cost of one mandatory event
    required = len(sched) * mc
    eps = 1e-9
    feasible = initial_energy + eps >= required
    if not feasible:
        warnings.warn(
            f"Freshness guarantee INFEASIBLE: schedule needs {required:.2f} "
            f"units but initial_energy is {initial_energy:.2f}.")

    # mand_after[i] = number of mandatory rows with index > i
    csum = np.cumsum(is_mand)
    mand_after = (len(sched) - csum).astype(int) if n else csum

    horizon = n if horizon is None else max(int(horizon), 1)
    disc0 = max(initial_energy - required, 0.0)    # discretionary budget

    values = df["value"].to_numpy(float)
    ts_ns = df["timestamp"].astype("int64").to_numpy()
    times = (ts_ns - ts_ns[0]) / 1e9 if n else np.array([])
    u1, u2, u3, u4 = (b * urg_scale for b in BASE_URGENCY)

    spent_budget = float(initial_energy)
    hv, ht = [], []
    low = high = None
    last_tx_value = None
    next_index = 0
    mandatory_missed = 0

    def next_mandatory_after(i):
        k = np.searchsorted(sched, i, side="right")
        return int(sched[k]) if k < len(sched) else n

    for i in range(n):
        budget = spent_budget
        if i < next_index:
            budget_trace[i] = budget
            continue

        mand = bool(is_mand[i])
        reserve_after = mand_after[i] * mc

        # ---- affordability: mandatory rows are funded by the reserve ----
        if mand:
            if budget + eps < mc:                  # schedule not fundable
                mandatory_missed += 1
                budget_trace[i] = budget
                next_index = i + 1
                continue
        else:
            if budget - sample_cost + eps < reserve_after:
                budget_trace[i] = budget           # optional sample unaffordable
                next_index = next_mandatory_after(i)
                continue

        # ---- acquire ----
        acquired[i] = True
        budget -= sample_cost
        v, t = values[i], times[i]
        low = v if low is None else min(low, v)
        high = v if high is None else max(high, v)
        scale = max(high - low, 1e-3 * max(abs(v), 1.0), 1e-9)

        var = pred = 0.0
        if len(hv) >= 2:
            rv = np.asarray(hv[-window:]); rt = np.asarray(ht[-window:])
            dt = np.maximum(np.diff(rt), 1e-9)
            var = float(np.median(np.abs(np.diff(rv)) / dt) * np.median(dt)) / scale
            slope = (hv[-1] - hv[-2]) / max(ht[-1] - ht[-2], 1e-9)
            step = np.clip(slope * (t - ht[-1]), -0.5 * scale, 0.5 * scale)
            pred = abs(v - (hv[-1] + step)) / scale
        urgency = max(var, pred)

        if urgency >= u4: desired = min_interval
        elif urgency >= u3: desired = max(min_interval, 3)
        elif urgency >= u2: desired = max(min_interval, 6)
        elif urgency >= u1: desired = max(min_interval, 12)
        else: desired = max_interval

        # ---- pacing on DISCRETIONARY budget (reserve excluded) ----
        committed = (tx_cost if mand else 0.0) + reserve_after
        disc = max(budget - committed, 0.0)
        left = max(1.0 - i / horizon, 1.0 / horizon)
        r = (disc / disc0) / left if disc0 > 0 else 0.0
        if r >= 1.0: floor = min_interval
        elif r >= 0.8: floor = 3
        elif r >= 0.6: floor = 6
        elif r >= 0.4: floor = 12
        else: floor = max_interval
        if urgency >= u4 and r >= 0.6:
            interval = max(desired, min(floor, 3))
        else:
            interval = max(desired, floor)
        interval = int(np.clip(interval, min_interval, max_interval))

        # ---- transmit ----
        if mand:
            send = True                            # funded by reserve
        else:
            thr = tx_frac * scale * float(
                np.clip(1.0 / max(min(r, 1.0), 1e-3), 1.0, 4.0))
            if urgency >= u2:
                thr *= 0.5
            wants = last_tx_value is None or abs(v - last_tx_value) >= thr
            send = wants and (budget - tx_cost + eps >= reserve_after)
        if send:
            transmitted[i] = True
            budget -= tx_cost
            last_tx_value = v

        if -eps < budget < 0:
            budget = 0.0
        spent_budget = budget
        budget_trace[i] = budget
        hv.append(v); ht.append(t)

        # never step past the next mandatory row
        next_index = min(i + interval, next_mandatory_after(i))
        intervals[i] = next_index - i

    out["acquired"] = acquired
    out["transmit"] = transmitted
    out["sampling_interval"] = intervals
    out["method"] = "combined_v4"
    out["energy_budget_remaining"] = budget_trace
    out["mandatory"] = is_mand
    out.attrs["freshness_feasible"] = bool(feasible)
    out.attrs["mandatory_required_energy"] = float(required)
    out.attrs["mandatory_missed"] = int(mandatory_missed)
    out.attrs["max_stale_rows"] = int(max_stale_rows)
    return out


def verify_freshness(trace, max_stale_rows=120, initial_energy=100.0,
                     sample_cost=0.02, tx_cost=0.08):
    """Check the guarantee from the ACTUAL transmit/acquire flags."""
    n = len(trace)
    tx = np.flatnonzero(trace["transmit"].to_numpy())
    acq = trace["acquired"].to_numpy()
    spent = acq.sum() * sample_cost + len(tx) * tx_cost
    if len(tx) == 0:
        return {"n": n, "transmissions": 0, "guarantee_met": False}
    between = int(np.diff(tx).max()) if len(tx) > 1 else 0
    leading = int(tx[0])                           # rows before first tx
    trailing = int((n - 1) - tx[-1])
    worst = max(between, leading, trailing)
    final_sent = bool(trace["transmit"].iloc[-1])
    return {
        "n": n,
        "acquired": int(acq.sum()),
        "transmissions": int(len(tx)),
        "max_gap_between_tx": between,
        "leading_gap": leading,
        "trailing_gap": trailing,
        "worst_gap": worst,
        "final_row_transmitted": final_sent,
        "tx_without_acquire": bool((trace["transmit"] & ~trace["acquired"]).any()),
        "energy_spent": float(spent),
        "within_budget": bool(spent <= initial_energy + 1e-9),
        "min_budget_trace": float(trace["energy_budget_remaining"].min()),
        "budget_matches_spend": bool(abs(
            trace["energy_budget_remaining"].iloc[-1]
            - (initial_energy - spent)) < 1e-6),
        "guarantee_met": bool(worst <= max_stale_rows and final_sent),
    }