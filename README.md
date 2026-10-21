# Adaptive Sensing and Communication for Energy-Efficient IoT Systems

An experimental study of adaptive sensor sampling and wireless transmission strategies for energy-efficient Internet of Things (IoT) systems using real-world sensor data.

## Overview

IoT sensor nodes continuously collect environmental measurements and communicate them to a gateway. Sampling and transmitting every reading can consume unnecessary energy, while reducing communication too aggressively can degrade the accuracy of the reconstructed signal at the gateway.

This project investigates an adaptive sensing and communication strategy that balances three objectives:

- **Energy efficiency:** reduce the number of sensor acquisitions and network transmissions.
- **Reconstruction accuracy:** preserve the quality of the signal reconstructed at the gateway.
- **Data freshness:** ensure that the gateway receives updates within a specified maximum interval.

The proposed approach, **Combined V4**, uses signal variability, prediction error, and the remaining energy budget to guide sensing and transmission decisions. It also reserves energy for mandatory transmissions to enforce a maximum transmission-gap constraint when the configured energy budget is sufficient.

The approach is evaluated against fixed-rate and adaptive baseline methods using the Intel Berkeley Research Lab sensor dataset.

## Research objectives

The primary objectives are:

1. Investigate whether adaptive sampling can reduce the number of sensor acquisitions without significantly degrading reconstruction accuracy.
2. Evaluate whether adaptive transmission policies can reduce communication overhead.
3. Combine signal-dependent decisions with energy-aware scheduling.
4. Enforce a maximum transmission gap through an explicit mandatory-transmission schedule.
5. Compare reconstruction quality, energy consumption, sampling reduction, transmission reduction, and freshness compliance across sensor traces.

The central research question is:

> Can a combined adaptive sensing and communication policy reduce energy expenditure and communication overhead while maintaining acceptable gateway reconstruction accuracy and bounded data staleness?

## Proposed approach: Combined V4

Combined V4 makes sensing and transmission decisions using information about the observed signal and the remaining energy budget.

### 1. Signal variability

The algorithm estimates recent signal variability using changes between previously acquired measurements. More variable signals can justify more frequent sensing because long intervals between observations may miss meaningful changes.

### 2. Prediction error

A local prediction is derived from recent measurements. The difference between the current reading and its predicted value contributes to an urgency score. Higher prediction error can trigger more frequent sampling or transmission.

### 3. Energy-aware sampling

The algorithm tracks simulated energy expenditure and adjusts its sampling interval according to the remaining discretionary budget.

The policy attempts to use fewer acquisitions when energy is constrained while preserving resources for required future transmissions.

### 4. Adaptive transmission

A newly acquired measurement can be transmitted when the signal changes sufficiently relative to the last transmitted value. Transmission decisions also account for the urgency of the current signal and the available energy.

### 5. Mandatory freshness schedule

The algorithm explicitly constructs a schedule of mandatory transmission rows, separated by no more than the configured maximum gap. The final row is included, avoiding duplicate entries when it coincides with an existing scheduled row.

Energy is reserved for the mandatory schedule before discretionary sampling and transmission costs are incurred. If the initial budget cannot fund the mandatory schedule, the run is marked infeasible rather than being treated as a successful freshness guarantee.

The default maximum gap used in the experiments is 120 rows.

**Important:** a row-based freshness limit is a bound on the number of dataset observations, not necessarily a fixed duration in seconds. Actual time-based freshness depends on the timestamps of the sensor readings.

## Experimental baselines

The evaluator compares five methods:

| Method | Description |
|---|---|
| `fixed_rate` | Periodic sampling and transmission baseline. |
| `signal_only` | Adaptive policy based on signal characteristics. |
| `energy_only` | Adaptive policy based on energy considerations. |
| `combined_v3` | Earlier combined adaptive policy. |
| `combined_v4` | Combined policy with energy reservation and mandatory freshness scheduling. |

All methods should be evaluated on the same sensor traces and with consistent metric definitions to enable a fair comparison.

## Dataset

The project uses the **Intel Berkeley Research Lab sensor dataset**.

Original dataset page: https://db.csail.mit.edu/labdata/labdata.html

The dataset contains environmental sensor readings collected by distributed sensor motes. The local dataset directory contains:

- `data/real/data.txt` — sensor measurements.
- `data/real/mote_locs.txt` — mote location information.
- `data/real/connectivity.txt` — connectivity information.

The measurement records contain the following fields:

| Field | Description |
|---|---|
| `date` | Measurement date |
| `time` | Measurement time |
| `epoch` | Dataset epoch or record identifier |
| `moteid` | Sensor mote identifier |
| `temperature` | Temperature measurement |
| `humidity` | Relative humidity measurement |
| `light` | Light measurement |
| `voltage` | Sensor voltage measurement |

The experiments focus on temperature, humidity, and light. Selected motes are evaluated independently, subject to the availability of sufficient valid measurements.

### Data preprocessing

The data loader in `src/real_data.py` is responsible for loading and preparing the dataset for evaluation. Invalid or unavailable measurements should be handled consistently before metrics are calculated.

The evaluator may skip traces that contain fewer than the configured minimum number of readings.

## Energy model

The implementation uses a simplified, simulated energy-cost model.

The default parameters are:

| Parameter | Value |
|---|---:|
| Initial energy budget | 100.0 units |
| Sampling cost | 0.02 units per acquired reading |
| Transmission cost | 0.08 units per transmission |
| Maximum transmission gap | 120 rows |

The simulated energy expenditure is calculated as:

\[
E_{\mathrm{spent}}
=
N_{\mathrm{acquired}}E_{\mathrm{sample}}
+
N_{\mathrm{transmitted}}E_{\mathrm{tx}}
\]

where:

- \(N_{\mathrm{acquired}}\) is the number of acquired readings.
- \(N_{\mathrm{transmitted}}\) is the number of transmitted readings.
- \(E_{\mathrm{sample}}\) is the cost of one acquisition.
- \(E_{\mathrm{tx}}\) is the additional cost of one transmission.

These values are abstract simulation units. They are **not measurements of joules, battery capacity, radio power, or real hardware energy consumption**.

The model assumes that each acquired reading incurs a sampling cost and that each transmission incurs an additional transmission cost. Real-world energy consumption would require a calibrated hardware or radio-energy model.

## Evaluation metrics

### Gateway reconstruction accuracy

Reconstruction quality is evaluated at the gateway using the readings actually transmitted by the sensor.

The gateway reconstructs the signal from transmitted measurements rather than from every reading acquired locally.

The primary error metrics are:

- **Normalized Mean Absolute Error (NMAE):** measures average absolute reconstruction error relative to the signal's normalization scale.
- **Normalized Root Mean Squared Error (NRMSE):** measures reconstruction error while penalizing larger deviations more strongly.

Lower NMAE and NRMSE indicate better reconstruction accuracy.

The exact normalization convention is defined by the evaluation implementation in `src/evaluate_real.py`.

### Sampling reduction

Sampling reduction measures the percentage reduction in acquired readings relative to the fixed-rate reference:

\[
R_{\mathrm{sample}}
=
100\left(1-\frac{N_{\mathrm{adaptive}}}
{N_{\mathrm{reference}}}\right)
\]

### Transmission reduction

Transmission reduction measures the percentage reduction in transmitted readings relative to the fixed-rate reference:

\[
R_{\mathrm{tx}}
=
100\left(1-\frac{T_{\mathrm{adaptive}}}
{T_{\mathrm{reference}}}\right)
\]

Higher reductions indicate fewer acquisitions or transmissions relative to the reference.

### Freshness compliance

Freshness is verified using actual transmission indices.

The verification considers:

- The gap before the first transmission.
- Gaps between consecutive transmissions.
- The trailing gap between the last transmission and the final dataset row.
- Whether the final row is transmitted.
- Whether any transmission occurs without a corresponding acquisition.

For a configured maximum gap \(S\), the freshness constraint is satisfied only when the checked gaps do not exceed \(S\), the final row is transmitted, and transmission integrity is maintained.

### Energy-budget compliance

Energy expenditure is independently recomputed from acquisition and transmission counts. The result is compared with the configured initial energy budget and, where available, the policy's reported remaining-energy trace.

A successful evaluation should verify both that energy expenditure remains within budget and that the reported remaining energy agrees with the independently calculated expenditure.

## Repository structure

The primary source files are:

```text
GPT/
├── data/
│   └── real/
│       ├── data.txt
│       ├── connectivity.txt
│       └── mote_locs.txt
├── results/
│   ├── final_metrics.csv
│   ├── summary_by_method.csv
│   ├── figures/
│   └── verification/
├── src/
│   ├── main.py
│   ├── evaluate.py
│   ├── evaluate_real.py
│   ├── methods.py
│   ├── real_data.py
│   └── plot_results.py
├── README.md
├── requirements.txt
└── .gitignore
```

`src/plot_results.py` is used to generate visualizations when it has been created in the repository. Generated files under `results/` depend on which evaluation and plotting commands have been run.

### Source file responsibilities

- `src/main.py` — Combined V4 adaptive sensing and transmission policy, including mandatory transmission scheduling and freshness verification.
- `src/evaluate.py` — runs the baseline comparison and writes the evaluation metrics.
- `src/evaluate_real.py` — computes reconstruction and other evaluation metrics.
- `src/methods.py` — contains the baseline sampling and transmission policies.
- `src/real_data.py` — loads and prepares the Intel Lab dataset.
- `src/plot_results.py` — generates research plots from the evaluation results, if present.

## Installation

Python 3.10 or newer is recommended, subject to the versions supported by the dependencies.

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If plotting dependencies are not included in `requirements.txt`, install them with:

```bash
python -m pip install matplotlib pandas numpy
```

## Running the evaluation

Run commands from the repository root.

### Evaluate selected motes

The following command evaluates the selected motes across temperature, humidity, and light:

```bash
python src/evaluate.py \
  --motes 1 3 10 15 \
  --variables temperature humidity light \
  --max-rows 20000 \
  --max-stale 120 \
  --out results/final_metrics.csv
```

The `--max-rows` option limits the number of rows considered per selected mote and variable. Increase it to process longer traces.

### Evaluate all available rows for the selected motes

```bash
python src/evaluate.py \
  --motes 1 3 10 15 \
  --variables temperature humidity light \
  --max-rows 100000000 \
  --max-stale 120 \
  --out results/final_metrics.csv
```

The evaluator still applies its own preprocessing and minimum-valid-reading rules. Check the console output to confirm which traces were evaluated or skipped.

### Generate summary metrics and charts

If `src/plot_results.py` is present, run:

```bash
python src/plot_results.py
```

The plotting script reads `results/final_metrics.csv` and saves charts under `results/figures/`, along with a summary table when configured to do so.

The charts should be regenerated whenever the evaluation dataset, method implementation, or evaluation parameters change.

## Reproducibility and verification checklist

Before recording results in a report or publication:

1. Record the dataset, selected motes, variables, and maximum number of rows used.
2. Keep the energy-cost parameters and freshness bound consistent across comparisons.
3. Verify that all methods are evaluated on the same eligible traces.
4. Confirm that gateway reconstruction uses transmitted measurements only.
5. Calculate freshness from the actual transmission flags, including leading and trailing gaps.
6. Independently recompute energy expenditure from the action counts.
7. Preserve the generated metrics CSV and the exact code used for the run.
8. Regenerate all figures from the final metrics rather than mixing results from earlier algorithm versions.

## Results interpretation

The purpose of the evaluation is to understand the trade-off among reconstruction quality, sensing activity, communication overhead, and energy consumption.

No single metric is sufficient to establish that a method is best overall:

- Fixed-rate sampling may provide better reconstruction accuracy while consuming more energy.
- Aggressive adaptive policies may reduce energy use but introduce larger reconstruction errors or freshness violations.
- Combined V4 is designed to balance signal-dependent decisions with a reserved energy budget for mandatory transmissions.

The most appropriate method depends on the application's constraints. When the freshness bound and energy budget are mandatory, methods that violate either constraint should not be considered feasible solutions under that configuration.

Numerical results should be reported from the latest reproducible evaluation output. Earlier experiments and partial-trace results should not be mixed with full-trace results without explicitly identifying the difference in experimental setup.

## Limitations

The current research prototype has several limitations:

1. **Simulated energy:** energy consumption is represented by abstract cost units rather than physical measurements.
2. **Row-based freshness:** a maximum gap in rows does not guarantee a fixed wall-clock delay when sample timestamps are irregular.
3. **Dataset coverage:** results from the Intel Lab dataset may not generalize to other environments, sensor types, or communication networks.
4. **Reconstruction trade-offs:** maintaining a bounded transmission gap does not guarantee the lowest reconstruction error for every sensor trace.
5. **Parameter dependence:** the results depend on energy costs, the initial budget, urgency thresholds, and the maximum freshness gap.
6. **Evaluation scope:** conclusions apply to the traces and configurations actually tested. Broader claims require further experiments, ideally including held-out traces and additional datasets.

## Future work

Potential extensions include:

- Calibrating the energy model against real sensor hardware.
- Expressing freshness requirements in elapsed time rather than row count.
- Evaluating performance on additional sensor datasets and unseen traces.
- Investigating reconstruction-error outliers for difficult sensor traces.
- Studying sensitivity to energy budgets and maximum freshness intervals.
- Adding statistical comparisons and repeated experiments where appropriate.
- Evaluating the policy on an actual or emulated wireless sensor network.

## Acknowledgements

This project uses the Intel Berkeley Research Lab sensor dataset.

Original dataset: https://db.csail.mit.edu/labdata/labdata.html
