"""Evaluate existing baselines and Combined V4.

Run from GPT/:
python src/eval_v4.py --motes 3 10 \
    --variables temperature humidity light \
    --out results/v4_2/motes3_10.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from real_data import load_intel_data
from methods import fixed_rate, adaptive_policy
from evaluate_real import score
from main import combined_v4, verify_freshness


def get_policies(df, params):
    return [
        ("fixed_rate", fixed_rate(df, 10)),
        ("signal_only", adaptive_policy(df, "signal_only")),
        ("energy_only", adaptive_policy(df, "energy_only")),
        ("combined_v3", adaptive_policy(df, "combined")),
        ("combined_v4", combined_v4(df, **params)),
    ]


def get_diagnostics(trace, max_stale_rows=120, tx_cost=0.08,
                    sample_cost=0.02, initial_energy=100.0):
    """Independently verify freshness and energy from actual trace flags."""
    n = len(trace)

    tx = trace["transmit"].fillna(False).to_numpy(dtype=bool)
    acquired = trace["acquired"].fillna(False).to_numpy(dtype=bool)
    tx_rows = np.flatnonzero(tx)

    if n == 0 or len(tx_rows) == 0:
        leading_gap = n
        trailing_gap = n
        max_between = 0
        max_gap = n
        median_gap = float(n)
        last_tx = -1
    else:
        gaps = np.diff(tx_rows)
        leading_gap = int(tx_rows[0])
        trailing_gap = int(n - 1 - tx_rows[-1])
        max_between = int(gaps.max()) if len(gaps) else 0
        max_gap = max(leading_gap, trailing_gap, max_between)
        median_gap = float(np.median(gaps)) if len(gaps) else float(n)
        last_tx = int(tx_rows[-1])

    final_sent = bool(n > 0 and tx[-1])
    tx_without_acquire = bool(np.any(tx & ~acquired))

    acquired_count = int(acquired.sum())
    tx_count = int(tx.sum())

    # Recompute energy independently from the action flags.
    energy_spent = acquired_count * sample_cost + tx_count * tx_cost
    remaining_energy = initial_energy - energy_spent

    freshness_met = bool(
        n > 0
        and len(tx_rows) > 0
        and max_gap <= max_stale_rows
        and final_sent
        and not tx_without_acquire
    )

    result = {
        "max_tx_gap_rows": max_gap,
        "max_between_tx_rows": max_between,
        "leading_gap_rows": leading_gap,
        "median_tx_gap_rows": median_gap,
        "trailing_gap_rows": trailing_gap,
        "final_row_transmitted": final_sent,
        "last_tx_row": last_tx,
        "tx_without_acquire": tx_without_acquire,
        "independent_energy_spent": float(energy_spent),
        "independent_energy_remaining": float(remaining_energy),
        "independent_within_budget": bool(remaining_energy >= -1e-9),
        "meets_gap_bound": freshness_met,
    }

    # Compare recomputed energy with the policy's reported energy.
    if "energy_budget_remaining" in trace.columns and n > 0:
        reported = float(trace["energy_budget_remaining"].iloc[-1])
        result["budget_matches_spend"] = bool(
            np.isfinite(reported)
            and abs(reported - remaining_energy) < 1e-6
        )

    if n > 0 and trace["method"].iloc[0] == "combined_v4":
        result["freshness_guarantee_met"] = freshness_met
        result["freshness_feasible"] = trace.attrs.get(
            "freshness_feasible", None
        )
        result["mandatory_missed"] = trace.attrs.get(
            "mandatory_missed", None
        )

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/real/data.txt")
    p.add_argument("--motes", type=int, nargs="+", default=[1, 3, 10, 15])
    p.add_argument("--variables", nargs="+",
                   default=["temperature", "humidity", "light"])
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--out", default="results/v4_2/all_metrics.csv")
    p.add_argument("--tx-frac", type=float, default=0.02)
    p.add_argument("--urg-scale", type=float, default=0.2)
    p.add_argument("--max-stale", type=int, default=120)
    args = p.parse_args()

    params = {"tx_frac": args.tx_frac, "urg_scale": args.urg_scale,
              "max_stale_rows": args.max_stale}

    results = []
    for variable in args.variables:
        print(f"\nLoading {variable} data...")
        data = load_intel_data(args.input, None, variable)
        for mote in args.motes:
            df = (data[data.moteid == mote].iloc[:args.max_rows]
                  .reset_index(drop=True))
            if len(df) < 1000:
                print(f"Skipping mote {mote}, {variable}: only {len(df)} rows")
                continue
            df.attrs["variable"] = variable
            print(f"Evaluating mote {mote}, {variable}, {len(df)} readings...")
            for name, trace in get_policies(df, params):
                row = score(df, trace, name)
                row.update(
                    get_diagnostics(
                        trace,
                        max_stale_rows=args.max_stale,
                        tx_cost=0.08,
                        sample_cost=0.02,
                        initial_energy=100.0,
                    )
                )
                results.append(row)
    if not results:
        print("No valid traces were evaluated.")
        return

    res = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    cols = ["sensor_id", "variable", "method", "samples_acquired",
            "transmissions", "estimated_energy_units", "gateway_nrmse_pct",
            "gateway_nmae_pct", "sample_reduction_pct",
            "transmission_reduction_pct", "max_tx_gap_rows",
            "trailing_gap_rows", "final_row_transmitted",
            "meets_gap_bound", "freshness_guarantee_met",
            "budget_exhausted_at_row"]
    cols = [c for c in cols if c in res.columns]
    print("\nDETAILED RESULTS")
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    metrics = [c for c in ["estimated_energy_units", "gateway_nrmse_pct",
                           "gateway_nmae_pct", "sample_reduction_pct",
                           "transmission_reduction_pct"] if c in res.columns]
    print("\nAVERAGE RESULTS BY METHOD")
    print(res.groupby("method")[metrics].mean().round(3).to_string())

    v4 = res[res["method"] == "combined_v4"]
    if len(v4):
        ok = int(v4["freshness_guarantee_met"].sum())
        print(f"\nV4 freshness guarantee met on {ok}/{len(v4)} traces")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()