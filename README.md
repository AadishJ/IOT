# Adaptive Sensing and Communication for Energy-Efficient IoT Systems

This prototype evaluates four causal sampling policies on the Intel Berkeley
Research Lab sensor dataset: fixed-rate, signal-only adaptive, energy-only
adaptive, and combined adaptive sampling.

## Setup

Copy the dataset into `data/real/` so this file exists:

```text
data/real/data.txt
```

Optional accompanying files are `mote_locs.txt` and `connectivity.txt`; this
experiment does not need them.

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the experiment from the project root:

```bash
python src/evaluate_real.py
```

Defaults: mote 3, temperature, first 20,000 valid readings, fixed interval 10.
To select a different mote or sensor variable:

```bash
python src/evaluate_real.py --sensor-id 11 --variable humidity
python src/evaluate_real.py --sensor-id 3 --variable light --max-rows 10000
```

To process all valid rows for the selected mote, use `--max-rows 0`. This may
take longer. The script uses the real timestamps and sorts readings in time.

## Outputs

Written to `results/real/`:
- `metrics_real.csv`: four-policy comparison
- `real_signal_clean.csv`: selected, cleaned sensor series
- `*_trace.csv`: detailed per-row policy traces
- `real_sampling_comparison.png`: real signal and selected samples
- `real_energy_comparison.png`: illustrative energy-cost comparison

## Important limitations

- The controller makes decisions only from observations it has acquired so
  far. It does not inspect future readings to choose its next sampling time.
- The Intel Lab dataset has real sensor measurements and voltage readings, but
  does not provide direct per-sample/per-transmission energy consumption for
  this experiment.
- `estimated_energy_units` uses abstract costs (0.02 per acquired sample and
  0.08 per transmission). It is a model-based comparison, not measured joules
  and not battery-life evidence.
- The signal-only, energy-only, and combined policies are simple research
  prototypes, not faithful reproductions of the cited papers. Their thresholds
  are initial values and should be documented and tested for sensitivity.
- The data may be irregularly sampled and contain missing values. This script
  sorts observations and drops missing values for the chosen variable. The
  fixed interval is in rows, not a fixed wall-clock duration.
- Results from one mote and one time window are preliminary. Repeat on multiple
  motes, variables, time windows, and parameter settings before making claims.
- The voltage column is preserved as data but is not treated as energy consumed.

## Research workflow

1. Run the default experiment.
2. Check `metrics_real.csv` and the plots.
3. Repeat for several mote IDs and variables.
4. Run parameter sensitivity experiments and report average and variability.
5. If possible, calibrate an energy model from documented hardware power
   specifications or direct measurements.

## Dataset

Intel Berkeley Research Lab sensor data: https://db.lcs.mit.edu/labdata/labdata.html
