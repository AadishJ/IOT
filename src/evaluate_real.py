"""Evaluate causal sampling policies on Intel Lab sensor data.

Reports both local reconstruction (all acquired samples) and gateway
reconstruction (transmitted samples), plus an illustrative energy cost.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_data import load_intel_data
from methods import (
    fixed_rate, adaptive_policy, acquired_rows, transmitted_rows
)


def _reconstruct(full, sampled):
    if sampled.empty:
        return np.full(len(full), np.nan)
    origin = full["timestamp"].iloc[0].value
    x = (full["timestamp"].astype("int64").to_numpy() - origin) / 1e9
    sx = (sampled["timestamp"].astype("int64").to_numpy() - origin) / 1e9
    y = sampled["value"].to_numpy(float)
    # np.interp assumes sorted coordinates; loader is expected to sort data.
    if len(sx) == 1:
        return np.full(len(full), y[0])
    return np.interp(x, sx, y)


def _error_metrics(full, sampled, prefix):
    pred = _reconstruct(full, sampled)
    y = full["value"].to_numpy(float)
    if np.isnan(pred).all():
        return {f"{prefix}_mae": float("nan"), f"{prefix}_rmse": float("nan"),
                f"{prefix}_nmae_pct": float("nan"), f"{prefix}_nrmse_pct": float("nan")}
    err = pred - y
    span = max(float(np.max(y) - np.min(y)), 1e-12)
    return {
        f"{prefix}_mae": float(np.nanmean(np.abs(err))),
        f"{prefix}_rmse": float(np.sqrt(np.nanmean(err ** 2))),
        f"{prefix}_nmae_pct": float(np.nanmean(np.abs(err)) / span * 100),
        f"{prefix}_nrmse_pct": float(np.sqrt(np.nanmean(err ** 2)) / span * 100),
    }


def score(full, trace, name, sample_cost=0.02, tx_cost=0.08):
    acquired = acquired_rows(trace)
    sent = transmitted_rows(trace)
    n = len(full)
    ns = int(trace["acquired"].sum())
    nt = int(trace["transmit"].sum())
    result = {
        "method": name,
        "sensor_id": int(full["moteid"].iloc[0]),
        "variable": full.attrs.get("variable", "temperature"),
        "ground_truth_readings": n,
        "samples_acquired": ns,
        "transmissions": nt,
        "sample_reduction_pct": 100 * (1 - ns / n),
        "transmission_reduction_pct": 100 * (1 - nt / n),
        "estimated_energy_units": ns * sample_cost + nt * tx_cost,
    }
    # Local accuracy and gateway accuracy are distinct: a sensed reading that
    # was not transmitted cannot help reconstruct the signal at the gateway.
    result.update(_error_metrics(full, acquired, "local"))
    result.update(_error_metrics(full, sent, "gateway"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/real/data.txt")
    p.add_argument("--sensor-id", type=int, default=3)
    p.add_argument("--variable",
                   choices=["temperature", "humidity", "light", "voltage"],
                   default="temperature")
    p.add_argument("--max-rows", type=int, default=20000,
                   help="Chronological subset; 0 means all rows.")
    p.add_argument("--fixed-interval", type=int, default=10)
    p.add_argument("--output-dir", default="results/real")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_intel_data(args.input, args.sensor_id, args.variable)
    if args.max_rows > 0:
        df = df.iloc[:args.max_rows].copy()
    if len(df) < 3:
        raise SystemExit("Too few valid readings. Check path, sensor ID and variable.")
    df.attrs["variable"] = args.variable
    df.to_csv(outdir / "real_signal_clean.csv", index=False)

    traces = [
        ("fixed_rate", fixed_rate(df, args.fixed_interval)),
        ("signal_only", adaptive_policy(df, "signal_only")),
        ("energy_only", adaptive_policy(df, "energy_only")),
        ("combined_adaptive", adaptive_policy(df, "combined")),
    ]

    metrics = []
    for name, trace in traces:
        trace.to_csv(outdir / f"{name}_trace.csv", index=False)
        metrics.append(score(df, trace, name))

    m = pd.DataFrame(metrics)
    m.to_csv(outdir / "metrics_real.csv", index=False)
    print("\nREAL DATA RESULTS (energy is illustrative, not measured joules)")
    print(m.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    plt.figure(figsize=(13, 5))
    plt.plot(df["timestamp"], df["value"],
             label="Ground-truth readings", linewidth=1)
    for name, trace in traces:
        sent = transmitted_rows(trace)
        plt.scatter(sent["timestamp"], sent["value"], s=8, label=f"{name} sent", alpha=.7)
    plt.xlabel("Timestamp")
    plt.ylabel(args.variable)
    plt.title(f"Gateway-received samples: mote {args.sensor_id} ({args.variable})")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "real_sampling_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(m["method"], m["estimated_energy_units"])
    plt.ylabel("Estimated energy units (abstract)")
    plt.title("Illustrative cost comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(outdir / "real_energy_comparison.png", dpi=160)
    plt.close()
    print(f"\nSaved output to {outdir.resolve()}")


if __name__ == "__main__":
    main()
