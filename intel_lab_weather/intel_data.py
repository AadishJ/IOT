"""
intel_data.py - load the Intel Berkeley Research Lab file and cut it into 2-day windows.

data.txt columns (space separated):  date time epoch moteid temperature humidity light voltage
The file has malformed / truncated lines and wild outliers (dying motes), so:
  * lines that do not parse are dropped
  * each variable is set to NaN outside a plausible range (per variable, the rest of the row is kept)
  * rows are placed on a regular grid by EPOCH number (about one reading per 31 s per mote)
  * gaps up to max_fill epochs are linearly interpolated, longer gaps are left as holes
  * a window is N consecutive epochs with no holes and at least min_valid_frac of real (non-interpolated) readings
    (this is mode "grid"; it needs a mote to be nearly gap-free for 2 days, which the real file rarely is)
  * mode "rows" (default in run_intel_benchmark.py): a window is N consecutive VALID readings, one reading = one tick,
    no interpolation; it may not contain a jump of more than max_epoch_gap epochs between two readings.
    The solar/weather clock of the simulator is synthetic anyway, so it does not need the real timestamps.
CHECK the printed per-variable min/max after cleaning and adjust RANGES (--range) if they look wrong.
"""
import numpy as np
import pandas as pd

COLS = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
VARIABLES = ["temperature", "humidity", "light"]
RANGES = {"temperature": (0.0, 50.0), "humidity": (0.0, 100.0), "light": (0.0, 2500.0)}


def load_raw(path, motes=None, chunk=500_000, max_epoch=1_000_000):
    """Read data.txt (or .gz) in chunks, keep only the wanted motes and the numeric columns."""
    use = COLS[2:]                                     # epoch moteid temperature humidity light voltage
    parts, n_lines = [], 0
    reader = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, usecols=use, on_bad_lines="skip",
                         dtype=str, chunksize=chunk)
    for df in reader:
        n_lines += len(df)
        for c in use:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["epoch", "moteid"])
        df = df[(df.epoch >= 0) & (df.epoch <= max_epoch)]
        if motes is not None:
            df = df[df.moteid.isin(motes)]
        df = df.astype({"epoch": np.int64, "moteid": np.int64})
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    print(f"read {n_lines} parsable lines, kept {len(out)} rows for motes {sorted(motes) if motes else 'all'}")
    return out


def clean(df, ranges=None, min_voltage=None):
    """Set out-of-range values to NaN, per variable. Optionally blank everything below min_voltage."""
    ranges = ranges or RANGES
    df = df.copy()
    for v in VARIABLES:
        lo, hi = ranges[v]
        bad = ~df[v].between(lo, hi)            # NaN counts as bad
        print(f"  {v:12s} kept {int((~bad).sum()):>8d} of {len(df)} rows in [{lo}, {hi}]")
        df.loc[bad, v] = np.nan
    if min_voltage is not None:
        low = df["voltage"] < min_voltage
        for v in VARIABLES:
            df.loc[low, v] = np.nan
        print(f"  blanked {int(low.sum())} rows with voltage < {min_voltage}")
    return df


def _grid(sub, var):
    """Regular epoch grid for one mote and one variable: (values with NaN holes, first epoch)."""
    s = sub[["epoch", var]].dropna().drop_duplicates("epoch").sort_values("epoch")
    if len(s) < 2:
        return None, None
    e0 = int(s.epoch.iloc[0])
    arr = np.full(int(s.epoch.iloc[-1]) - e0 + 1, np.nan)
    arr[s.epoch.values - e0] = s[var].values
    return arr, e0


def find_windows(df, mote, var, n, max_fill=6, min_valid_frac=0.9, min_range=1.0, max_windows=6, step=720):
    """Return [(start_epoch, np.array(n))] - up to max_windows evenly spread, non-overlapping windows."""
    arr, e0 = _grid(df[df.moteid == mote], var)
    if arr is None:
        return [], {"grid": 0, "valid": 0, "flat": 0, "longest_run": 0}
    observed = ~np.isnan(arr)
    filled = pd.Series(arr).interpolate(limit=max_fill, limit_area="inside").values
    holes = np.isnan(filled)
    ch = np.concatenate([[0], np.cumsum(holes)])
    co = np.concatenate([[0], np.cumsum(observed)])
    starts, flat, s = [], 0, 0
    while s + n <= len(filled):
        if ch[s + n] - ch[s] == 0 and (co[s + n] - co[s]) / n >= min_valid_frac:
            w = filled[s:s + n]
            if w.max() - w.min() >= min_range:
                starts.append(s)
                s += n
                continue
            flat += 1
        s += step
    info = {"grid": len(arr), "valid": len(starts), "flat": flat}
    if not starts:
        return [], info
    if len(starts) > max_windows:
        pick = np.unique(np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int))
        starts = [starts[i] for i in pick]
    info["longest_run"] = info.get("longest_run", 0)
    return [(e0 + st, filled[st:st + n].copy(), n, max_fill) for st in starts], info


def find_windows_rows(df, mote, var, n, max_epoch_gap=300, min_range=1.0, max_windows=6, step=720):
    """Windows of n consecutive valid readings (one reading = one tick), no interpolation.
    Returns ([(start_epoch, array, epoch_span, max_jump)], info). A jump of more than max_epoch_gap epochs between two
    consecutive readings ends a segment; windows never cross such a jump."""
    s = df.loc[df.moteid == mote, ["epoch", var]].dropna().drop_duplicates("epoch").sort_values("epoch")
    ep, x = s.epoch.values, s[var].values
    info = {"grid": len(x), "valid": 0, "flat": 0, "longest_run": 0}
    if len(x) == 0:
        return [], info
    brk = np.where(np.diff(ep) > max_epoch_gap)[0] + 1
    seg_a = np.concatenate([[0], brk])
    seg_b = np.concatenate([brk, [len(x)]])
    info["longest_run"] = int((seg_b - seg_a).max())
    starts, flat = [], 0
    for a, b in zip(seg_a, seg_b):
        k = int(a)
        while k + n <= b:
            w = x[k:k + n]
            if w.max() - w.min() >= min_range:
                starts.append(k)
                k += n
            else:
                flat += 1
                k += step
    info["valid"], info["flat"] = len(starts), flat
    if not starts:
        return [], info
    if len(starts) > max_windows:
        pick = np.unique(np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int))
        starts = [starts[i] for i in pick]
    return [(int(ep[k]), x[k:k + n].copy(), int(ep[k + n - 1] - ep[k] + 1), int(np.diff(ep[k:k + n]).max()))
            for k in starts], info


def diagnose(df, ranges):
    """Print what the raw file looks like per mote: coverage, gap sizes, outliers, voltage."""
    print("\nDIAGNOSIS (before cleaning)")
    for mote, sub in df.groupby("moteid"):
        ep = np.sort(sub.epoch.drop_duplicates().values)
        span = int(ep[-1] - ep[0] + 1)
        gaps = np.diff(ep) if len(ep) > 1 else np.array([0])
        q = np.percentile(gaps, [50, 95, 99])
        print(f"mote {mote}: {len(ep)} distinct epochs over a span of {span} ({100 * len(ep) / span:.0f}% present); "
              f"gap between readings: median {q[0]:.0f}, p95 {q[1]:.0f}, p99 {q[2]:.0f}, max {gaps.max()}; "
              f"jumps > 10: {int((gaps > 10).sum())}, > 60: {int((gaps > 60).sum())}, > 300: {int((gaps > 300).sum())}")
        for v in VARIABLES:
            x = sub[v].dropna()
            lo, hi = ranges[v]
            ok = x[(x >= lo) & (x <= hi)]
            pct = np.percentile(ok, [1, 50, 99]) if len(ok) else [np.nan] * 3
            vo = sub.loc[(sub[v] < lo) | (sub[v] > hi), "voltage"].median()
            vi = sub.loc[(sub[v] >= lo) & (sub[v] <= hi), "voltage"].median()
            print(f"   {v:12s} values {len(x)}, below {lo}: {int((x < lo).sum())}, above {hi}: {int((x > hi).sum())}; "
                  f"in-range 1/50/99 percentile {pct[0]:.1f} / {pct[1]:.1f} / {pct[2]:.1f}; "
                  f"median voltage of out-of-range rows {vo:.2f} vs in-range rows {vi:.2f}")