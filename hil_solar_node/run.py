#!/usr/bin/env python3
"""
hil_collect.py - drive the ESP8266 benchmark over serial and turn the log into tables.

  pip install pyserial pandas
  python hil_collect.py run   --port /dev/ttyUSB0 --signals 0 1 2 --out hil_results   # record + sweep
  python hil_collect.py parse --log hil_results/raw_log.txt --out hil_results # (re)build the tables from a saved log

run   : for each signal: sends g<n>, waits for you to press Enter, sends c (records the real
        sensor, ~6.7 min at TICK_MS=70), waits for CAPTURE_DONE, sends the sweep (b = 16 combos x 5
        seeds), then re-runs 16 combos and checks the results are identical (determinism check).
        Every serial line is saved to raw_log.txt.
parse : reads RESULT and TRACE lines from the raw log and writes
        results_all.csv        one row per run (signal, weather, method, seed)
        summary_mean_std.csv   mean/std over seeds per (signal, weather, method)
        trace_<signal>.csv     the recorded sensor trace (feed this to your Python sim)
"""

import argparse, os, sys, time

COLS = [
    "weather",
    "signal",
    "method",
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
    "seed",
]


def wait_for(ser, token, log, timeout_s):
    """Read lines until one starts with `token`; return all RESULT lines seen meanwhile."""
    t0 = time.time()
    results = []
    while time.time() - t0 < timeout_s:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").rstrip("\r\n")
        log.write(line + "\n")
        if line.startswith("RESULT,"):
            results.append(line)
        if line.startswith(("CAPTURE_DONE", "SWEEP_DONE", "ERR")):
            print(line[:100])
        if line.startswith("ERR"):
            sys.exit("firmware error: " + line)
        if line.startswith(token):
            log.flush()
            return results
    sys.exit(f"timeout waiting for {token}")


def run(a):
    import serial

    os.makedirs(a.out, exist_ok=True)
    ser = serial.Serial(
        a.port, 115200, timeout=1
    )  # opening the port resets most NodeMCU boards
    time.sleep(1)
    ser.reset_input_buffer()
    t_start = time.time()
    with open(os.path.join(a.out, "raw_log.txt"), "a") as log:
        print("waiting for the board to finish setup (~10 s)...")
        time.sleep(12)
        for g in a.signals:
            name = ["temperature", "humidity", "soil"][g]
            ser.write(f"g{g}\n".encode())
            time.sleep(1)
            input(
                f"\n[{name}] Get ready for your disturbances, then press Enter to START recording... "
            )
            ser.write(b"c\n")
            print(f"recording {name} ... do your disturbances now")
            wait_for(ser, "CAPTURE_DONE", log, a.capture_timeout_h * 3600)
            print(f"trace recorded, running sweep '{a.sweep}'")
            t0 = time.time()
            ser.write(f"{a.sweep}\n".encode())
            res1 = wait_for(ser, "SWEEP_DONE", log, 3600)
            print(f"sweep took {time.time() - t0:.0f} s, {len(res1)} runs")
            # determinism check: re-run the 16 combos at the seed the sweep ended on; must be identical
            seed = res1[-1].rsplit(",", 1)[1]
            ref = sorted(l for l in res1 if l.rsplit(",", 1)[1] == seed)
            ser.write(b"a\n")
            res2 = sorted(wait_for(ser, "SWEEP_DONE", log, 3600))
            print(
                f"[{name}] determinism check (seed {seed}):",
                (
                    "PASS"
                    if ref == res2
                    else "FAIL - weather is not reproducible on this core, results are NOT comparable"
                ),
            )
    print(
        f"done in {(time.time() - t_start) / 60:.1f} min. now run: python hil_collect.py parse --log",
        os.path.join(a.out, "raw_log.txt"),
        "--out",
        a.out,
    )


def parse(a):
    import pandas as pd

    os.makedirs(a.out, exist_ok=True)
    rows, traces = [], {}
    for line in open(a.log, errors="replace"):
        line = line.strip()
        if line.startswith("RESULT,"):
            f = line.split(",")[1:]
            if len(f) == len(COLS):
                rows.append(f)
        elif line.startswith("TRACE,"):
            _, sig, i, v = line.split(",")
            traces.setdefault(sig, {})[int(i)] = float(
                v
            )  # last recording of a signal wins
    if not rows:
        sys.exit(
            "no RESULT lines found (is this the right log, and is the firmware the seed-column version?)"
        )
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # if a log contains the same run twice (re-run), keep the last one
    df = df.drop_duplicates(subset=["signal", "weather", "method", "seed"], keep="last")
    df.to_csv(os.path.join(a.out, "results_all.csv"), index=False)
    num = [c for c in COLS[3:] if c != "seed"]
    summ = df.groupby(["signal", "weather", "method"])[num].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ = summ.reset_index()
    summ["n_seeds"] = df.groupby(["signal", "weather", "method"]).size().values
    summ.to_csv(os.path.join(a.out, "summary_mean_std.csv"), index=False)
    for sig, d in traces.items():
        pd.DataFrame(sorted(d.items()), columns=["tick", "value"]).to_csv(
            os.path.join(a.out, f"trace_{sig}.csv"), index=False
        )
    order_m = ["fixed_rate", "signal_only", "energy_only", "combined_v4"]
    order_w = ["sunny", "sunny_to_cloudy", "cloudy_to_sunny", "cloudy"]
    pd.set_option("display.width", 200)
    for sig in df.signal.unique():
        d = summ[summ.signal == sig]
        for col, title in [
            ("nrmse_mean", "NRMSE % (lower better)"),
            ("load_J_mean", "energy used, J"),
            ("uptime_mean", "uptime %"),
            ("brownouts_mean", "brown-outs"),
            ("max_gap_min_mean", "longest data gap, min"),
        ]:
            p = d.pivot(index="method", columns="weather", values=col).reindex(
                index=order_m, columns=order_w
            )
            print(f"\n== {sig}: {title} (mean over seeds) ==")
            print(p.round(2).to_string())
    print(f"\nwrote tables to {a.out}  ({len(df)} runs)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", required=True)
    r.add_argument("--signals", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--out", default="hil_results")
    r.add_argument("--capture-timeout-h", type=float, default=6.0)
    r.add_argument(
        "--sweep",
        choices=["a", "b"],
        default="b",
        help="b = 16 combos x seeds 1..5 (default), a = 16 combos, seed 1 only (faster)",
    )
    q = sub.add_parser("parse")
    q.add_argument("--log", required=True)
    q.add_argument("--out", default="hil_results")
    a = p.parse_args()
    run(a) if a.cmd == "run" else parse(a)
