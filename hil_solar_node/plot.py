#!/usr/bin/env python3
"""
plot_hil.py - figures from the folder written by `run.py parse`.

  pip install matplotlib pandas
  python3 plot_hil.py --dir hil_results            # add --tick-ms N if you changed TICK_MS (default 70)

Reads  hil_results/summary_mean_std.csv  (+ trace_<signal>.csv if present)
Writes hil_results/figures/
    1_tradeoff_error_vs_energy.png   accuracy vs energy, one panel per signal
    2_metrics_by_weather.png         NRMSE / energy / uptime / longest gap, grouped by weather
    3_recorded_traces.png            the real sensor traces that every run replayed
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ["fixed_rate", "signal_only", "energy_only", "combined_v4"]
WEATHERS = ["sunny", "sunny_to_cloudy", "cloudy_to_sunny", "cloudy"]
SIGNALS = ["temperature", "humidity", "soil"]
COLORS = {"fixed_rate": "#7f7f7f", "signal_only": "#1f77b4", "energy_only": "#ff7f0e", "combined_v4": "#2ca02c"}
MARKERS = {"sunny": "o", "sunny_to_cloudy": "^", "cloudy_to_sunny": "v", "cloudy": "s"}
WLABEL = {"sunny": "sunny", "sunny_to_cloudy": "sunny→cloudy", "cloudy_to_sunny": "cloudy→sunny", "cloudy": "cloudy"}
UNIT = {"temperature": "°C", "humidity": "%RH", "soil": "% moisture"}


def load(d):
    s = pd.read_csv(os.path.join(d, "summary_mean_std.csv"))
    for c in [c for c in s.columns if c.endswith("_std")]:
        s[c] = s[c].fillna(0.0)
    return s


def fig_tradeoff(s, out):
    sigs = [x for x in SIGNALS if x in set(s.signal)]
    fig, axes = plt.subplots(1, len(sigs), figsize=(5.4 * len(sigs), 5.0), squeeze=False)
    for ax, sig in zip(axes[0], sigs):
        for _, r in s[s.signal == sig].iterrows():
            dead = r.uptime_mean < 99.5
            ax.errorbar(r.load_J_mean, r.nrmse_mean, xerr=r.load_J_std, yerr=r.nrmse_std,
                        fmt="none", ecolor=COLORS[r.method], alpha=0.35, lw=1)
            ax.scatter(r.load_J_mean, r.nrmse_mean, s=75, marker=MARKERS[r.weather], zorder=3,
                       facecolors="white" if dead else COLORS[r.method],
                       edgecolors=COLORS[r.method], linewidths=1.7)
        ax.set_title(sig.capitalize())
        ax.set_xlabel("energy used by the node (J, model)")
        ax.set_ylabel("reconstruction error, NRMSE (%)")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.annotate("better", xy=(0.04, 0.04), xycoords="axes fraction", color="#555", fontsize=9)
    h = [Line2D([0], [0], marker="o", ls="", markerfacecolor=COLORS[m], markeredgecolor=COLORS[m],
                markersize=9, label=m) for m in METHODS]
    h += [Line2D([0], [0], marker=MARKERS[w], ls="", markerfacecolor="#bbb", markeredgecolor="#555",
                 markersize=9, label=WLABEL[w]) for w in WEATHERS]
    h += [Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor="#333",
                 markersize=9, label="hollow = node ran out of battery (uptime < 100%)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Accuracy vs energy (lower-left is better)", fontsize=13)
    fig.tight_layout(rect=[0, 0.15, 1, 0.96])
    fig.savefig(out, dpi=200); plt.close(fig)


def fig_bars(s, out):
    sigs = [x for x in SIGNALS if x in set(s.signal)]
    metrics = [("nrmse", "NRMSE (%)  lower = better"), ("load_J", "energy used (J)"),
               ("uptime", "uptime (%)"), ("max_gap_min", "longest data gap (min)")]
    fig, axes = plt.subplots(len(metrics), len(sigs), figsize=(5.2 * len(sigs), 3.3 * len(metrics)), squeeze=False)
    x = np.arange(len(WEATHERS)); w = 0.2
    for j, sig in enumerate(sigs):
        d = s[s.signal == sig]
        for i, (m, lab) in enumerate(metrics):
            ax = axes[i][j]
            for k, meth in enumerate(METHODS):
                dd = d[d.method == meth].set_index("weather").reindex(WEATHERS)
                ax.bar(x + (k - 1.5) * w, dd[f"{m}_mean"].fillna(0), w,
                       yerr=dd[f"{m}_std"].fillna(0), color=COLORS[meth], capsize=2, label=meth)
            ax.set_xticks(x); ax.set_xticklabels([WLABEL[v] for v in WEATHERS], fontsize=8)
            ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
            if m == "uptime":
                ax.set_ylim(0, 108)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title(sig.capitalize())
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=200); plt.close(fig)


def fig_traces(d, tick_ms, out):
    sigs = [x for x in SIGNALS if os.path.exists(os.path.join(d, f"trace_{x}.csv"))]
    if not sigs:
        return False
    fig, axes = plt.subplots(len(sigs), 1, figsize=(10, 2.7 * len(sigs)), squeeze=False, sharex=True)
    for ax, sig in zip(axes[:, 0], sigs):
        t = pd.read_csv(os.path.join(d, f"trace_{sig}.csv"))
        ax.plot(t.tick * tick_ms / 60000.0, t.value, lw=1.2, color="#333")
        ax.set_ylabel(f"{sig} ({UNIT[sig]})"); ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("recording time (min)")
    sec = axes[0, 0].secondary_xaxis("top",
        functions=(lambda m: m * 60000.0 / tick_ms * 30 / 3600, lambda h: h * 3600 / 30 * tick_ms / 60000.0))
    sec.set_xlabel("simulated time (h)   [5760 ticks = 2 simulated days]")
    fig.tight_layout()
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hil_results")
    ap.add_argument("--tick-ms", type=float, default=70.0)
    a = ap.parse_args()
    s = load(a.dir)
    fd = os.path.join(a.dir, "figures"); os.makedirs(fd, exist_ok=True)
    fig_tradeoff(s, os.path.join(fd, "1_tradeoff_error_vs_energy.png"))
    fig_bars(s, os.path.join(fd, "2_metrics_by_weather.png"))
    got = fig_traces(a.dir, a.tick_ms, os.path.join(fd, "3_recorded_traces.png"))
    print("wrote figures to", fd, "" if got else "(no trace_*.csv found, skipped the trace figure)")