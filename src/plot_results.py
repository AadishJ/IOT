from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_data import load_intel_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA_FILE = ROOT / "data" / "real" / "data.txt"
METRICS_FILE = RESULTS / "final_metrics.csv"

FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(METRICS_FILE)

required = {"method", "gateway_nrmse_pct", "estimated_energy_units"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in metrics CSV: {sorted(missing)}")

# Consistent method order for all comparisons.
preferred_order = [
    "fixed_rate",
    "signal_only",
    "energy_only",
    "combined_v3",
    "combined_v4",
]
methods = [m for m in preferred_order if m in df["method"].unique()]
df = df[df["method"].isin(methods)].copy()

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES / filename}")

# 1. Gateway reconstruction accuracy.
means = df.groupby("method")["gateway_nrmse_pct"].mean().reindex(methods)
plt.figure(figsize=(9, 5))
bars = plt.bar(means.index, means.values)
plt.ylabel("Gateway NRMSE (%)")
plt.xlabel("Method")
plt.title("Gateway Reconstruction Error — Full-Trace Evaluation")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, means.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
save_plot("01_gateway_nrmse.png")

# 2. Energy consumption and budget.
energy = df.groupby("method")["estimated_energy_units"].mean().reindex(methods)
plt.figure(figsize=(9, 5))
bars = plt.bar(energy.index, energy.values)
plt.axhline(100, linestyle="--", linewidth=1.5,
            label="100-unit budget")
plt.ylabel("Average estimated energy (simulation units)")
plt.xlabel("Method")
plt.title("Energy Consumption by Method")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, energy.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{value:.1f}", ha="center", va="bottom", fontsize=9)
save_plot("02_energy_consumption.png")

# 3. Sample and transmission reductions.
reduction_cols = [
    ("sample_reduction_pct", "Sample reduction"),
    ("transmission_reduction_pct", "Transmission reduction"),
]
available = [(col, label) for col, label in reduction_cols if col in df.columns]
if available:
    summary = df.groupby("method")[[c for c, _ in available]].mean().reindex(methods)
    x = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(10, 5))
    for j, (col, label) in enumerate(available):
        offset = (j - (len(available)-1)/2) * width
        plt.bar(x + offset, summary[col].values, width, label=label)
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Reduction relative to fixed-rate baseline (%)")
    plt.title("Sampling and Communication Savings")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    save_plot("03_efficiency_reductions.png")

# 4. Reconstruction accuracy for each sensor trace.
if {"sensor_id", "variable"}.issubset(df.columns):
    pivot = df.pivot_table(
        index=["sensor_id", "variable"],
        columns="method",
        values="gateway_nrmse_pct",
        aggfunc="mean",
    ).reindex(columns=methods)
    labels = [f"Mote {m} — {v}" for m, v in pivot.index]
    x = np.arange(len(labels))
    width = 0.82 / max(len(methods), 1)

    plt.figure(figsize=(max(12, len(labels)*0.8), 6))
    for j, method in enumerate(methods):
        plt.bar(
            x - 0.41 + width/2 + j*width,
            pivot[method].values,
            width,
            label=method,
        )
    plt.xticks(x, labels, rotation=55, ha="right")
    plt.ylabel("Gateway NRMSE (%)")
    plt.xlabel("Sensor trace")
    plt.title("Reconstruction Accuracy by Mote and Variable")
    plt.legend(ncol=min(3, len(methods)))
    plt.grid(axis="y", alpha=0.3)
    save_plot("04_accuracy_by_sensor.png")

# 5. Raw dataset overview: temperature, humidity, and light.
# Plot representative motes, with point thinning for readability.
motes = [1, 3, 10, 15]
variables = ["temperature", "humidity", "light"]
fig, axes = plt.subplots(3, 1, figsize=(13, 10))

for ax, variable in zip(axes, variables):
    raw = load_intel_data(str(DATA_FILE), None, variable)

    for mote in motes:
        series = raw[raw["moteid"] == mote].copy()
        if series.empty:
            continue

        series["timestamp"] = pd.to_datetime(
            series["timestamp"], errors="coerce"
        )
        series = series.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

        # Thin plotted points only; all data remain in the evaluation.
        if len(series) > 3000:
            step = int(np.ceil(len(series) / 3000))
            series = series.iloc[::step]

        ax.plot(
            series["timestamp"],
            series["value"],
            linewidth=0.8,
            label=f"Mote {mote}",
        )

    ax.set_title(f"Raw Intel Lab Dataset — {variable.title()}")
    ax.set_ylabel(variable.title())
    ax.set_xlabel("Timestamp")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4)

fig.suptitle("Sensor Readings Over Time (Representative Motes)", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "05_raw_sensor_overview.png",
            dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIGURES / '05_raw_sensor_overview.png'}")

# Save a concise summary table alongside the plots.
summary_cols = [
    c for c in [
        "gateway_nrmse_pct",
        "gateway_nmae_pct",
        "estimated_energy_units",
        "sample_reduction_pct",
        "transmission_reduction_pct",
    ]
    if c in df.columns
]
df.groupby("method")[summary_cols].mean().reindex(methods).to_csv(
    RESULTS / "summary_by_method.csv"
)

print("\nAverage metrics by method:")
print(
    df.groupby("method")[summary_cols]
      .mean()
      .reindex(methods)
      .round(3)
      .to_string()
)
print(f"\nCharts saved in: {FIGURES}")
