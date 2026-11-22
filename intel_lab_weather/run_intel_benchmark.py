#!/usr/bin/env python3
"""
run_intel_benchmark.py - the firmware's four policies on real Intel Lab traces, under the four weather scenarios.

  pip install numpy pandas matplotlib
  python3 run_intel_benchmark.py --data data/real/data.txt

What it does
  1. loads data.txt, cleans it, cuts each (mote, variable) into 2-day windows (5760 rows = the firmware's run length)
  2. for every window x weather (sunny, sunny->cloudy, cloudy->sunny, cloudy) x seed x method runs the SAME
     simulator as the ESP8266 sketch (iot_sim.py is a line-for-line float32 port; see the check in the notes)
  3. writes intel_results/: runs.csv, summary.csv, paired_vs_fixed.csv, win_rates.csv, figures/*.png

Statistics: seeds only re-draw the cloud noise (pseudo-replicates), so they are averaged inside each window first;
mean, std and standard error are then taken ACROSS windows (the independent units).
"""

import argparse, multiprocessing as mp, os, sys, time
from dataclasses import replace

import numpy as np
import pandas as pd

import intel_data as D
import iot_sim as S

_G = None
_CFG = None


def _init(seeds, cfg):
    global _G, _CFG
    _CFG = cfg
    _G = S.build_all_irradiance(seeds, cfg)


def _work(task):
    meta, trace = task
    rows = []
    for (w, seed), G in sorted(_G.items()):
        for m in S.METHODS:
            r = S.simulate(trace, G, m, _CFG)
            r.update(meta, weather=S.WEATHERS[w], seed=seed, method=m)
            rows.append(r)
    return rows


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data", default="../data/data.txt", help="data.txt or data.txt.gz"
    )
    ap.add_argument("--motes", type=int, nargs="+", default=[1, 3, 10, 15])
    ap.add_argument("--variables", nargs="+", default=D.VARIABLES, choices=D.VARIABLES)
    ap.add_argument(
        "--windows-per-mote",
        type=int,
        default=6,
        help="max 2-day windows per (mote, variable)",
    )
    ap.add_argument("--seeds", type=int, default=5, help="cloud-noise seeds 1..N")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="intel_results")
    ap.add_argument(
        "--mode",
        choices=["rows", "grid"],
        default="rows",
        help="rows: window = N consecutive valid readings, no interpolation (default); "
        "grid: N consecutive epochs, small gaps interpolated",
    )
    ap.add_argument(
        "--max-epoch-gap",
        type=int,
        default=300,
        help="rows mode: a jump bigger than this many epochs (~31 s each) between two readings ends a segment",
    )
    ap.add_argument(
        "--diagnose",
        action="store_true",
        help="print coverage / gap / outlier statistics of the raw file and exit",
    )
    ap.add_argument(
        "--max-fill",
        type=int,
        default=6,
        help="grid mode: longest gap (epochs) to interpolate",
    )
    ap.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.9,
        help="grid mode: min fraction of real readings in a window",
    )
    ap.add_argument(
        "--min-range",
        type=float,
        default=1.0,
        help="skip windows whose max-min is below this (units of the variable)",
    )
    ap.add_argument(
        "--min-voltage",
        type=float,
        default=None,
        help="optionally blank readings below this mote voltage",
    )
    ap.add_argument(
        "--range",
        nargs=3,
        action="append",
        metavar=("VAR", "LO", "HI"),
        help="override a plausible range",
    )
    ap.add_argument(
        "--box-weather",
        default="sunny",
        choices=S.WEATHERS,
        help="weather used in the boxplot figure",
    )
    ap.add_argument("--no-plots", action="store_true")
    # model overrides for sensitivity runs (defaults = firmware values)
    ap.add_argument("--p-sleep", type=float, help="W, default 1.5e-4")
    ap.add_argument("--e-sample", type=float, help="J, default 0.14")
    ap.add_argument("--e-tx", type=float, help="J, default 0.55")
    ap.add_argument("--batt-mah", type=float, help="default 10")
    ap.add_argument("--isc", type=float, help="A, default 6e-3")
    ap.add_argument(
        "--max-stale",
        type=int,
        help="combined_v4 mandatory spacing in rows, default 120",
    )
    return ap.parse_args()


def build_cfg(a):
    over = {
        k: v
        for k, v in dict(
            p_sleep=a.p_sleep,
            e_sample=a.e_sample,
            e_tx=a.e_tx,
            batt_mah=a.batt_mah,
            isc=a.isc,
            max_stale=a.max_stale,
        ).items()
        if v is not None
    }
    return replace(S.DEFAULT_CFG, **over), over


def collect_windows(a, cfg):
    ranges = dict(D.RANGES)
    for v, lo, hi in a.range or []:
        ranges[v] = (float(lo), float(hi))
    df = D.load_raw(a.data, a.motes)
    if a.diagnose:
        D.diagnose(df, ranges)
        sys.exit(0)
    print("cleaning:")
    df = D.clean(df, ranges, a.min_voltage)
    tasks, report = [], []
    for var in a.variables:
        for mote in a.motes:
            if a.mode == "rows":
                wins, info = D.find_windows_rows(
                    df,
                    mote,
                    var,
                    cfg.n,
                    a.max_epoch_gap,
                    a.min_range,
                    a.windows_per_mote,
                )
            else:
                wins, info = D.find_windows(
                    df,
                    mote,
                    var,
                    cfg.n,
                    a.max_fill,
                    a.min_valid_frac,
                    a.min_range,
                    a.windows_per_mote,
                )
            spans = [w[2] for w in wins]
            report.append(
                (
                    var,
                    mote,
                    info["grid"],
                    info["longest_run"],
                    info["valid"],
                    info["flat"],
                    len(wins),
                    f"{np.median(spans) / cfg.n:.2f}" if spans else "-",
                )
            )
            for k, (start, arr, sp, mj) in enumerate(wins):
                tasks.append(
                    (
                        {
                            "variable": var,
                            "mote": mote,
                            "window": k,
                            "start_epoch": start,
                            "epoch_span": sp,
                            "max_jump_epochs": mj,
                            "range": float(arr.max() - arr.min()),
                        },
                        arr,
                    )
                )
    print("\nwindows (2 days = %d rows, mode=%s):" % (cfg.n, a.mode))
    print(
        pd.DataFrame(
            report,
            columns=[
                "variable",
                "mote",
                "valid_readings",
                "longest_run",
                "valid_windows",
                "skipped_flat",
                "used",
                "epoch_span/rows",
            ],
        ).to_string(index=False)
    )
    if not tasks:
        print(
            "\nNo window found. 'longest_run' = the longest stretch of readings without a jump of more than "
            "--max-epoch-gap epochs; it must reach %d. Try a larger --max-epoch-gap (e.g. 1000), more motes, "
            "or run with --diagnose to see the gap structure." % cfg.n
        )
    return tasks


def summarise(df, outdir):
    key = ["variable", "mote", "window", "weather", "method"]
    metrics = [
        "nrmse",
        "nmae",
        "samples",
        "tx",
        "load_J",
        "harvest_J",
        "clipped_J",
        "uptime",
        "brownouts",
        "minSOC",
        "finalSOC",
        "max_gap_min",
    ]
    pw = (
        df.groupby(key)[metrics].mean().reset_index()
    )  # average the seeds inside each window
    g = pw.groupby(["variable", "weather", "method"])[metrics].agg(
        ["mean", "std", "sem", "count"]
    )
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    summary = g.reset_index()
    summary["n_windows"] = summary["nrmse_count"]
    summary = summary.drop(columns=[c for c in summary.columns if c.endswith("_count")])
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # README-style reductions vs fixed_rate, computed pairwise per window
    fx = pw[pw.method == "fixed_rate"].set_index(
        ["variable", "mote", "window", "weather"]
    )
    p = pw.set_index(["variable", "mote", "window", "weather"]).join(
        fx[["samples", "tx", "load_J"]], rsuffix="_fixed"
    )
    p["sampling_reduction_%"] = 100 * (1 - p.samples / p.samples_fixed)
    p["tx_reduction_%"] = 100 * (1 - p.tx / p.tx_fixed)
    p["energy_saving_%"] = 100 * (1 - p.load_J / p.load_J_fixed)
    paired = (
        p.reset_index()
        .groupby(["variable", "weather", "method"])[
            ["sampling_reduction_%", "tx_reduction_%", "energy_saving_%"]
        ]
        .mean()
        .reset_index()
    )
    paired.to_csv(os.path.join(outdir, "paired_vs_fixed.csv"), index=False)

    # how often combined_v4 has lower NRMSE than each other method, window by window
    pv = pw.pivot_table(
        index=["variable", "mote", "window", "weather"],
        columns="method",
        values="nrmse",
    )
    up = pw.pivot_table(
        index=["variable", "mote", "window", "weather"],
        columns="method",
        values="uptime",
    )
    wins = []
    for (var, wea), idx in pv.groupby(level=["variable", "weather"]).groups.items():
        for other in [m for m in S.METHODS if m != "combined_v4"]:
            a_, b_ = pv.loc[idx, "combined_v4"], pv.loc[idx, other]
            ok = a_.notna() & b_.notna()
            alive = (
                ok & (up.loc[idx, "combined_v4"] >= 99.5) & (up.loc[idx, other] >= 99.5)
            )
            wins.append(
                {
                    "variable": var,
                    "weather": wea,
                    "vs": other,
                    "win_rate_all": (
                        float((a_[ok] < b_[ok]).mean()) if ok.any() else np.nan
                    ),
                    "n_all": int(ok.sum()),
                    "win_rate_both_alive": (
                        float((a_[alive] < b_[alive]).mean()) if alive.any() else np.nan
                    ),
                    "n_both_alive": int(alive.sum()),
                }
            )
    pd.DataFrame(wins).to_csv(os.path.join(outdir, "win_rates.csv"), index=False)
    return pw, summary, paired


def print_tables(summary, paired):
    pd.set_option("display.width", 200)
    for var in summary.variable.unique():
        d = summary[summary.variable == var]
        for col, title in [
            ("nrmse_mean", "NRMSE % (lower is better)"),
            ("load_J_mean", "energy used, J (model)"),
            ("uptime_mean", "uptime %"),
            ("max_gap_min_mean", "longest data gap, min"),
        ]:
            p = d.pivot(index="method", columns="weather", values=col).reindex(
                index=S.METHODS, columns=S.WEATHERS
            )
            print(f"\n== {var}: {title} (mean over windows) ==")
            print(p.round(2).to_string())
        q = paired[paired.variable == var]
        for col in ["sampling_reduction_%", "tx_reduction_%"]:
            p = q.pivot(index="method", columns="weather", values=col).reindex(
                index=S.METHODS, columns=S.WEATHERS
            )
            print(f"\n== {var}: {col} vs fixed_rate ==")
            print(p.round(1).to_string())


def make_figures(pw, summary, outdir, box_weather):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    colors = {
        "fixed_rate": "#7f7f7f",
        "signal_only": "#1f77b4",
        "energy_only": "#ff7f0e",
        "combined_v4": "#2ca02c",
    }
    markers = {
        "sunny": "o",
        "sunny_to_cloudy": "^",
        "cloudy_to_sunny": "v",
        "cloudy": "s",
    }
    wl = {
        "sunny": "sunny",
        "sunny_to_cloudy": "sunny→cloudy",
        "cloudy_to_sunny": "cloudy→sunny",
        "cloudy": "cloudy",
    }
    vars_ = [v for v in D.VARIABLES if v in set(summary.variable)]

    fig, axes = plt.subplots(
        1, len(vars_), figsize=(5.4 * len(vars_), 5.0), squeeze=False
    )
    for ax, var in zip(axes[0], vars_):
        for _, r in summary[summary.variable == var].iterrows():
            dead = r.uptime_mean < 99.5
            ax.errorbar(
                r.load_J_mean,
                r.nrmse_mean,
                xerr=r.load_J_sem,
                yerr=r.nrmse_sem,
                fmt="none",
                ecolor=colors[r.method],
                alpha=0.35,
                lw=1,
            )
            ax.scatter(
                r.load_J_mean,
                r.nrmse_mean,
                s=75,
                marker=markers[r.weather],
                zorder=3,
                facecolors="white" if dead else colors[r.method],
                edgecolors=colors[r.method],
                linewidths=1.7,
            )
        ax.set_title(var.capitalize())
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("energy used by the node (J, model)")
        ax.set_ylabel("reconstruction error, NRMSE (%)")
    h = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markerfacecolor=colors[m],
            markeredgecolor=colors[m],
            markersize=9,
            label=m,
        )
        for m in S.METHODS
    ]
    h += [
        Line2D(
            [0],
            [0],
            marker=markers[w],
            ls="",
            markerfacecolor="#bbb",
            markeredgecolor="#555",
            markersize=9,
            label=wl[w],
        )
        for w in S.WEATHERS
    ]
    h += [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markerfacecolor="white",
            markeredgecolor="#333",
            markersize=9,
            label="hollow = node ran out of battery (uptime < 100%)",
        )
    ]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Intel Lab traces: accuracy vs energy (lower-left is better; bars = standard error across windows)"
    )
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])
    fig.savefig(os.path.join(fd, "1_tradeoff_error_vs_energy.png"), dpi=200)
    plt.close(fig)

    metrics = [
        ("nrmse", "NRMSE (%)  lower = better"),
        ("load_J", "energy used (J)"),
        ("uptime", "uptime (%)"),
        ("max_gap_min", "longest data gap (min)"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(vars_),
        figsize=(5.2 * len(vars_), 3.3 * len(metrics)),
        squeeze=False,
    )
    x = np.arange(4)
    w = 0.2
    for j, var in enumerate(vars_):
        d = summary[summary.variable == var]
        for i, (m, lab) in enumerate(metrics):
            ax = axes[i][j]
            for k, meth in enumerate(S.METHODS):
                dd = d[d.method == meth].set_index("weather").reindex(S.WEATHERS)
                ax.bar(
                    x + (k - 1.5) * w,
                    dd[f"{m}_mean"].fillna(0),
                    w,
                    yerr=dd[f"{m}_sem"].fillna(0),
                    color=colors[meth],
                    capsize=2,
                    label=meth,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([wl[v] for v in S.WEATHERS], fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            if m == "uptime":
                ax.set_ylim(0, 108)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title(var.capitalize())
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(fd, "2_metrics_by_weather.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(vars_), figsize=(5.0 * len(vars_), 4.4), squeeze=False
    )
    for ax, var in zip(axes[0], vars_):
        d = pw[(pw.variable == var) & (pw.weather == box_weather)]
        data = [d[d.method == m].nrmse.dropna().values for m in S.METHODS]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=[m.replace("_", "\n") for m in S.METHODS],
            showfliers=True,
        )
        for patch, m in zip(bp["boxes"], S.METHODS):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.7)
        ax.set_title(f"{var.capitalize()} ({wl[box_weather]})")
        ax.set_ylabel("NRMSE per window (%)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Window-to-window spread of reconstruction error")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "3_nrmse_spread_across_windows.png"), dpi=200)
    plt.close(fig)
    print("figures:", fd)


def main():
    a = parse_args()
    cfg, over = build_cfg(a)
    tasks = collect_windows(a, cfg)
    if not tasks:
        sys.exit("no usable windows found (see the message above)")
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(1, a.seeds + 1))
    n_runs = len(tasks) * 4 * len(seeds) * len(S.METHODS)
    print(
        f"\n{len(tasks)} windows x 4 weathers x {len(seeds)} seeds x {len(S.METHODS)} methods = {n_runs} runs"
        f"{'  (model overrides: ' + str(over) + ')' if over else ''}, {a.jobs} worker(s)"
    )
    t0, rows = time.time(), []
    with mp.Pool(a.jobs, initializer=_init, initargs=(seeds, cfg)) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, tasks), 1):
            rows.extend(r)
            if k % max(1, len(tasks) // 10) == 0 or k == len(tasks):
                print(
                    f"  {k}/{len(tasks)} windows done, {time.time() - t0:.0f} s",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    pw, summary, paired = summarise(df, a.out)
    print_tables(summary, paired)
    if not a.no_plots:
        make_figures(pw, summary, a.out, a.box_weather)
    print(
        f"\nwrote {a.out}/ (runs.csv, summary.csv, paired_vs_fixed.csv, win_rates.csv, figures/) in {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
