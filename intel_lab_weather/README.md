# Intel Lab benchmark (firmware policies, four weather scenarios)

Runs the four policies from the ESP8266 sketch (fixed_rate, signal_only, energy_only, triAS) on real
Intel Lab traces, under the same solar/battery model as the hardware demo.

    pip install -r requirements.txt
    python3 run_intel_benchmark.py --data /path/to/data.txt            # data.txt.gz also works
    python3 run_intel_benchmark.py --data /path/to/data.txt --windows-per-mote 2 --seeds 2   # quick test

Files
- iot_sim.py             float32 port of main.cpp: model, 4 policies, reconstruction metrics
- intel_data.py          loader, cleaning, 2-day windows
- run_intel_benchmark.py runner, tables, figures

Outputs (intel_results/): runs.csv, summary.csv, paired_vs_fixed.csv, win_rates.csv, figures/

If it finds no windows, run `--diagnose` first (coverage, gap sizes, outliers per mote), then adjust.
The real file is patchy, so the default mode cuts a window from 5760 consecutive VALID readings
(one reading = one tick, no interpolation) and never across a jump of more than --max-epoch-gap epochs (default 300,
about 2.6 h). `epoch_span/rows` in the windows table shows how stretched the real time axis is; runs.csv has
`max_jump_epochs` per window so you can filter or compare, e.g. rerun with --max-epoch-gap 100.

Useful options
- --diagnose             print raw-file statistics per mote and exit
- --mode rows|grid       rows (default) or the strict gap-free epoch grid with small-gap interpolation
- --max-epoch-gap 300    largest jump between consecutive readings allowed inside a window (rows mode)
- --motes 1 3 10 15 --variables temperature humidity light --windows-per-mote 6 --seeds 5
- --range VAR LO HI      override a plausible range, e.g. --range light 0 4000
- --min-voltage 2.4      blank readings from low-voltage motes
- --e-tx 1.1 --p-sleep 3e-4 --batt-mah 20 --out intel_results_sens   sensitivity runs (model constants)