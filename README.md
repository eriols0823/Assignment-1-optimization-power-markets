# 46750 - Assignment 1: Demand-Side Flexibility in Active Distribution Grids

Code for Group Assignment 1 of *46750 - Optimization in Modern Power Systems* (DTU): a `gurobipy` model of a
flexible consumer with PV and a grid connection, solved for each question of the assignment. Every number and
figure of the report is produced by the commands in Section 2 and saved under `results/` (figures in
`results/figures/`).

**Submission of the code.** The complete project code is handed in either as a single `.zip` file in DTU Learn
or as a link to the group's GitHub repository on the front page of the report; the report says which one.
A linked repository must be accessible to the graders (public, or private with the teaching team invited) and
must not be modified after the deadline: the last commit before the deadline is the one graded.
Chosen option: this GitHub repository, public and linked on the front page of the report.

## 1. Setup

Use one of the two options (both install the same packages; the code needs Python 3.10 or later, `gurobipy` and a
Gurobi licence). Every command below has been tested with Python 3.12 and the free size-limited licence that comes
with `pip install gurobipy`.

**Option A - pip and venv**
```bash
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate.bat         # Windows cmd
# venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

**Option B - conda**
```bash
conda env create -f environment.yaml
conda activate 46750-a1
```

Check that it works: `python main.py` solves the base case of `Q1_caseA` and writes it to `results/Q1_caseA/`.

## 2. Running the code

Run all commands from the repository root. Each question writes its tables (CSV, TXT) to `results/<case>/`
and its report figures to `results/figures/`; `--show` opens the figures in a window instead.

**Question 1** (`--question Q1_caseA` or `Q1_caseB`)
```bash
python main.py --question Q1_caseA              # base case
python main.py --question Q1_caseA --scenarios  # + example sensitivity scenarios
python main.py --question Q1_caseA --validate   # Question 1.(f): price ladder and dual check
python main.py --question Q1_caseB --validate   # the same for case B
```

| Command | Tables (`results/<case>/`) | Figures (`results/<case>/`) |
|---|---|---|
| `--question Q1_caseA` (or `Q1_caseB`) | `<case>_hourly.csv` (schedule and duals of every constraint), `<case>_summary.txt` | `schedule.png`, `duals.png`, `inputs.png` |
| `... --validate` | `<case>_validation.csv` (price-ladder prediction against the solver, hour by hour) | `lambda_range.png` (range of valid values of the balance dual) |
| `... --scenarios` | `<case>_<scenario>_hourly.csv`, `<case>_<scenario>_summary.txt` | `scenarios_cost.png` |

`--validate` (1.(f)) compares the solved load and PV with the price ladder of 1.(d), hour by hour, and checks
whether the dual of the power balance is unique by solving again with a little energy added and removed.

**Question 2**
```bash
python main.py --q2                                     # all of Question 2, (b) to (e), in one command (about 20 s)
python main.py --question Q2_linear --sweep             # 2.(b): base case and sweep of c^L
python main.py --question Q2_quadratic --sweep          # 2.(c): base case and sweep of c^Q
python main.py --question Q2_quadratic --compare        # 2.(d): linear vs quadratic vs Question 1 model
python main.py --question Q2_quadratic --nonseparable   # 2.(e) bonus: accumulated-deviation disutility
```

| Command | Tables (`results/...`) | Figures (`results/figures/`) |
|---|---|---|
| `--question Q2_linear --sweep` | `Q2_linear/`: `Q2_linear_hourly.csv`, `Q2_linear_summary.txt`, `sweep_cL.csv`, `sweep_cL_report_table.csv`, `sweep_cL_change_points.csv` | `q2b_schedule.png`, `q2b_sweep.png` |
| `--question Q2_quadratic --sweep` | `Q2_quadratic/`: `Q2_quadratic_hourly.csv`, `Q2_quadratic_summary.txt`, `sweep_cQ.csv`, `sweep_cQ_report_table.csv` | `q2c_schedule.png`, `q2c_sweep.png` |
| `--question Q2_quadratic --compare` | `Q2_comparison/`: `comparison.csv`, `comparison_loads.csv` | `q2d_profiles.png` |
| `--question Q2_quadratic --nonseparable` | `Q2_nonseparable/`: base case, `experiment.csv` | `q2e_schedule.png`, `q2e_profiles.png`, `q2e_experiment.png` |

The sweeps also run built-in sanity checks and stop with an error if one fails: the load never exceeds the
reference, the solver matches the closed-form (cost-benefit or first-order-condition) load in every hour of every
run, the change points of the linear sweep lie at the predicted price thresholds, the quadratic solution is
unique (same load with another solver seed), and the bonus model with rho = 0 reproduces 2.(c).

**Question 3** (`--question Q3` or `Q3_battery`)
```bash
python main.py --question Q3                  # 3.(d)-(e): minimum daily energy, quadratic disutility
python main.py --question Q3 --sensitivity    # 3.(f): sweeps of E^min, c^Q and L^max
python main.py --question Q3_battery          # 3.(g): the same consumer with a battery, the value of the battery and its sensitivity
```

| Command | Tables (`results/...`) | Figures |
|---|---|---|
| `--question Q3` | `Q3/`: `Q3_hourly.csv`, `Q3_summary.txt` (incl. mu^E as `dual[min_energy]`) | `results/Q3/`: `schedule.png`, `duals.png`, `inputs.png` |
| `--question Q3 --sensitivity` | `Q3/`: `sensitivity_E_min.csv`, `sensitivity_c_Q.csv`, `sensitivity_L_max.csv` | `results/Q3/`: `q3f_emin_sweep.png`, `q3f_emin_profiles.png`, `q3f_cq_profiles.png`, `q3f_lmax_profiles.png` |
| `--question Q3_battery` | `Q3_battery/`: `Q3_battery_hourly.csv` (incl. `charge`, `discharge`, `soc` and their duals), `Q3_battery_summary.txt`, `q3g_sensitivity.csv`, `q3g_end_of_day.csv`, `q3g_water_value.csv`, `q3g_pi_end_sweep.csv` | `results/figures/`: `q3g_battery_schedule.png`, `q3g_sensitivity.png` |

With a battery, `main.py` also solves the case without battery and prints the value of the battery, the value
pi^end of the energy left at the end of the day (computed from the next-day data by `terminal_value()` in
`src/model.py`), the final state of charge and the largest overlap of charging and discharging (3.(g).iv).
It then runs the sensitivity analysis of 3.(g).v: the value of the battery when one parameter changes at a time
(price spread and level, `c^Q`, capacity, power, efficiency; `q3g_sensitivity.csv`). The parameters and their
values are in `Q3_BATTERY_SWEEPS` in `main.py`, and the scenarios come from `src/scenarios.py`. Finally it compares
our pi^end with the end-of-day alternatives of 3.(g).i (trend of the last two prices, last price, free end, cyclic)
and with a benchmark over day D and hours 0-10 of D+1 (`solve_extended_horizon()` in `src/model.py`, written in a
compact form so that it fits the 200-variable limit of the size-limited Gurobi licence): `q3g_end_of_day.csv` (with
the value of the battery reported by each alternative and its exact value, with the energy left at the end valued
with the benchmark), the
value of the energy stored at the end of the day `q3g_water_value.csv`, and the values of pi^end that change the
decisions `q3g_pi_end_sweep.csv`.

## 3. Repository structure

```
main.py                  Entry point: load data -> build model -> solve -> save results and figures, one function per experiment
src/
  data_loader.py         load_question("Q1_caseA") -> InputData (all parameters, with units)
  model.py               FlexibleConsumerModel (Questions 1 and 3, with the battery of 3.(g)) and the Question 2 models:
                         build() / solve() -> Results; terminal_value() and solve_extended_horizon() for 3.(g)
  scenarios.py           Helpers that derive scenarios from a base InputData (prices, tariffs, PV, preferences,
                         disutility, battery)
  plotting.py            Figures for inputs, schedules, duals, scenario comparisons, sweeps and the battery
data/                    Input data, see Section 6
results/                 Written by main.py (tables per case, figures in results/figures/)
requirements.txt, environment.yaml, LICENSE, .gitignore
```

## 4. Code conventions

* Every variable family is stored in `self.var[<name>]` and every constraint family in `self.con[<name>]`.
  `solve()` returns the hourly values of each variable family as a column of `results.hourly`, the dual value
  (Gurobi attribute `Pi`) of each hourly constraint as a column `dual_<name>`, and the dual of every single
  constraint in `results.duals`. Families named `import`, `export`, `load`, `pv` make the standard plots work.
* Bounds that need a dual are written as explicit constraints (`addConstr`), not as variable bounds, because
  Gurobi reports a variable bound in the reduced cost and not in `Pi`.
* Sign convention: the objective is maximised and the constraints are written as `h = 0` and `g <= 0`, so each
  `dual_<name>` is directly the multiplier `mu >= 0` (or the free `lambda` of the balance) of the report.
* Quadratic terms sit in the objective; `QCPDual` is set for the rare case of a quadratic constraint.
* Scenarios are derived in code from the base data (`src/scenarios.py`), never by editing the JSON files.
* Every function has a docstring with inputs, outputs and units.

## 5. Code of each question

**Question 1** (hourly consumption decision with a linear utility)

* `src/model.py`: `FlexibleConsumerModel.build()` writes the LP of 1.(a) over the 24 hours: the variables
  `load`, `pv`, `import`, `export`, the objective `sum_t u^L l_t - p^imp_t q^imp_t + p^exp_t q^exp_t - c^PV q^PV_t`
  and the constraints `balance`, `load_lo`, `load_up`, `pv_lo`, `pv_up`, `imp_nonneg`, `exp_nonneg`, all in the
  form of 1.(b), so that each `dual_<name>` is the multiplier of the report (lambda_t and the six mu_t).
  `solve()` returns the schedule, the objective split into utility and procurement cost, and every dual.
* `main.py`: `run_base_case` (schedule, duals and figures of any case) and `run_q1_validation` (1.(f): price
  ladder of 1.(d) against the solver, and uniqueness of the balance dual).
* `src/plotting.py`: `plot_inputs`, `plot_schedule`, `plot_duals` and `plot_lambda_range`.

**Question 2** (separable and non-separable disutility)

* `src/model.py`: `DisutilityModel` builds the Question 1 variables and constraints (with a zero utility) and
  replaces the objective by `-disutility - procurement cost`; subclasses only define the disutility.
  `LinearDisutilityModel` (2.(b)) writes `c^L |l_t - l^ref_t|` as an LP with deviation variables
  `d+_t, d-_t >= 0`. `QuadraticDisutilityModel` (2.(c)) uses `c^Q (l_t - l^ref_t)^2` directly. The bonus
  `AccumulatedDeviationModel` (2.(e)) adds the accumulated deviation `e_t = rho e_{t-1} + (l_t - l^ref_t)`, with
  the disutility `sum_t c^Q (l_t - l^ref_t)^2 + c^S e_t^2` (defaults `c^Q = 0.05`, `c^S = 0.06`, `rho = 0.8`).
  The results carry the extra hourly columns `deviation` (and `dev_up`, `dev_down` or `accumulated`) and
  `meta["disutility"]`.
* `src/scenarios.py`: `set_disutility(data, linear=, quadratic=)` overrides the disutility coefficients.
* `src/plotting.py`: `plot_sweep` (sweep of a coefficient) and `plot_load_profiles` (loads of several models
  against the reference, optionally with the price), used for the Question 2 figures. The schedules use the
  common `plot_schedule`.
* `main.py`: `run_q2_sweep` (2.(b), 2.(c): base case, sweep, report table, figures, checks),
  `run_q2_comparison` (2.(d)) and `run_q2_nonseparable` (2.(e): base case, comparison with 2.(b) and 2.(c),
  experiment on the order of the prices at equal mean and amplitude), plus the closed-form predictions
  `predicted_load_linear` and `predicted_load_quadratic` used by the checks. Question 2 solves the base case with
  its own model and writes its figures to `results/figures/` instead of the standard plots of Question 1.

**Question 3** (minimum daily energy and a small battery)

* `src/model.py`, 3.(a)-(e): `FlexibleConsumerModel.build()` uses the quadratic disutility of 2.(c) as the
  objective when the data contain `c^Q`, and adds the constraint `min_energy`, `E^min - sum_t l_t <= 0`, when
  they contain `E^min`. Its dual is mu^E (`results.duals["min_energy"]`).
* `src/model.py`, 3.(g): when the data contain a battery, `build()` adds the variables `charge`, `discharge`,
  `soc`, the state-of-charge constraint `soc_dyn` and the limits `ch_lo`, `ch_up`, `dis_lo`, `dis_up`, `soc_lo`,
  `soc_up`, and the term `pi^end (e_23 - e^init)` in the objective. `terminal_value()` computes pi^end from the
  next-day data. The option `end_of_day` ("value", "cyclic" or "free") and `pi_end` give the alternatives
  compared in 3.(g).v. `solve_extended_horizon()` is the benchmark of 3.(g).v over day D and hours 0-10 of D+1.
* `src/scenarios.py`: `set_load_preferences` (E^min, L^max), `set_disutility` (c^Q), `set_battery` (capacity,
  power, efficiency), `scale_prices` and `shift_prices` (spread and level of the prices, also on the next day).
* `main.py`: `run_q3_sensitivity` (3.(f)), `run_q3_battery` (3.(g).iv), `run_q3_battery_sensitivity` and
  `run_q3_end_of_day` (3.(g).v). The parameters and values of the 3.(g).v sweeps are in `Q3_BATTERY_SWEEPS`.
* `src/plotting.py`: `plot_battery_schedule` and `plot_battery_sensitivity` for 3.(g); 3.(f) reuses `plot_sweep`
  and `plot_load_profiles`.

## 6. Input data

All input data lives in `data/`: two **shared** files plus one small **case** file per
question. The shared files form a catalogue of every appliance used anywhere in the assignment;
each `params_<case>.json` composes the consumer of one case by picking appliances from the
catalogue by ID. All time series have 24 hourly values (hour 0 to 23).

**Units.** Every field name carries its unit. Fields ending in `_kW` are power ratings (kW);
fields ending in `_kWh_per_hour` (or `_KWh_per_hour` - same unit despite the capital K) are hourly
energy bounds (kWh consumed within one hour); fields ending in `_kWh` are energy amounts;
`_DKK_per_kWh` are prices/costs and `_DKK_per_kWh2` is the quadratic disutility coefficient.
**Capacity factors and ratios** (`hourly_profile_ratio`, `reference_load_capacity_factor`,
`initial_state_of_charge_ratio`, and the `_next_day` profiles) are **dimensionless fractions in
[0, 1] of the corresponding maximum** - multiply by it to get physical values. With 1-hour periods,
a power rating in kW and an energy amount in kWh/h are numerically interchangeable; the loader
does every conversion for you and exposes physical quantities:
`pv_available = max_power_kW x hourly_profile_ratio` (kWh/h),
`reference_load = max_load_KWh_per_hour x reference_load_capacity_factor` (kWh/h), and
`battery_initial_soc_kWh = storage_capacity_kWh x initial_state_of_charge_ratio` (kWh).

| Question in the assignment | Case | Appliances (from the catalogue) |
|---|---|---|
| Question 1 - hourly consumption decision (utility u, PV cost c_PV), case A: c_PV < u | `Q1_caseA` | `PV_01_A` + `FL_01` (a load with `consumption_utility_DKK_per_kWh`; no reference profile, no daily requirement) |
| Question 1, case B: a more expensive PV, c_PV > u | `Q1_caseB` | `PV_01_B` + `FL_01` (`PV_01_B` differs from `PV_01_A` only in `marginal_cost_DKK_per_kWh`) |
| Question 2 - linear disutility of deviating from a reference profile | `Q2_linear` | `PV_01_A` + `FL_02_L` (reference profile, `linear_disutility_DKK_per_kWh`; same hourly bounds as Question 1; no utility) |
| Question 2 - quadratic disutility | `Q2_quadratic` | `PV_01_A` + `FL_02_Q` (as `FL_02_L` with `quadratic_disutility_DKK_per_kWh2` instead) |
| Question 3 - minimum daily energy requirement | `Q3` | `PV_01_A` + `FL_03` (as `FL_02_Q` plus `min_total_energy_per_day_kWh`, the E_min of the daily requirement) |
| Question 3.(g) - the same consumer with a small battery | `Q3_battery` | `PV_01_A` + `FL_03` + `BAT_01` |

No data is shipped for the bonus question 2.(e): its parameters (`rho = 0.8`, `c^Q = 0.05`, `c^S = 0.06`) were
chosen by the group, starting from the `Q2_quadratic` case, and are set in code (`AccumulatedDeviationModel`).
All the values chosen by the group (sweeps, scenarios, parameters) are listed in the appendix of the report.

`params_<case>.json` - the case file
: `consumers[]`: `consumer_id`, `connection_bus` (a `bus_id` of `bus_params.json`), `list_appliances`
  (IDs of the consumer's DERs, loads and storages in `appliance_params.json`). `hyperparameters` is
  unused in Assignment 1 (`null`).

`appliance_params.json` - the shared catalogue, one list per appliance type (`null` if none)
: **DERs**: `DER_id`, `DER_type` (`"PV"`), `max_power_kW` (peak power, kW), `marginal_cost_DKK_per_kWh` (cost of every kWh produced), `hourly_profile_ratio` (dimensionless fraction of `max_power_kW` available each hour; available energy in kWh/h = `max_power_kW` x ratio), `hourly_profile_ratio_next_day` (see *Next-day forecasts* below)
: **loads**: `load_id`, `load_type`, `max_load_KWh_per_hour` and `min_load_KWh_per_hour` (hourly consumption bounds, kWh/h), `consumption_utility_DKK_per_kWh` (value of every kWh consumed, Question 1), `linear_disutility_DKK_per_kWh` and `quadratic_disutility_DKK_per_kWh2` (Questions 2-3), `min_total_energy_per_day_kWh` (the E_min of Question 3), `reference_load_capacity_factor` (dimensionless fraction of `max_load_KWh_per_hour`; preferred consumption in kWh/h = max load x factor), `reference_hourly_profile_ratio_next_day` (see *Next-day forecasts*). Fields a load does not use are `null`. The hourly bounds are the same for every load: deviations from the reference are limited only by them.
: **storages**: `storage_id`, `storage_type` (`"battery"`), `storage_capacity_kWh`, `max_charging_power_kW`, `max_discharging_power_kW`, `charging_efficiency` and `discharging_efficiency` (fraction of the energy transferred that survives each conversion - a full round trip keeps η_ch·η_dis), `initial_state_of_charge_ratio` (fraction of `storage_capacity_kWh` in the battery at the start of the day). **No final state of charge is given on purpose**: how to treat the battery's state of charge at the end of the day (constraint or objective) is your modeling choice in Question 3.(g).
: **heat_pumps**: not used in Assignment 1 (`null`)

`bus_params.json` - grid and market conditions at the connection bus
: `bus_id`, `import_tariff_DKK_per_kWh`, `export_tariff_DKK_per_kWh`, `energy_price_DKK_per_kWh` (24 values), `energy_price_next_day_DKK_per_kWh` (see *Next-day forecasts*), `max_import_kW`, `max_export_kW` (`null` = no grid limit in Assignment 1)

**Next-day forecasts.** Three fields carry the forecast for the *following* day:
`energy_price_next_day_DKK_per_kWh` (in `bus_params.json`), `hourly_profile_ratio_next_day` (in the
DER entries) and `reference_hourly_profile_ratio_next_day` (in the load entries). `load_question()`
exposes them as `energy_price_next_day`, `pv_available_next_day` and `reference_load_next_day`. We use
them only in Question 3.(g): the next-day prices give the value pi^end of the energy left in the battery
(`terminal_value()`), and the three of them define the benchmark of 3.(g).v (`solve_extended_horizon()`).

`load_question()` in `src/data_loader.py` shows exactly how each attribute is derived from these files.

**Modifying or adding data.** For sensitivity analyses, prefer deriving scenarios in code
(`src/scenarios.py`) over editing the JSON files - it keeps the base case intact and the experiment
reproducible. If you do add appliances or case files, follow the same structure (a new entry in the
catalogue, a new `params_<case>.json`) and document them here.
