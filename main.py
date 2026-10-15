"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios
    python main.py --question Q1_caseB --validate   # Question 1.(f): price ladder and dual check
    python main.py --q2                     # all of Question 2 (b)-(e) in one command
    python main.py --question Q2_linear --sweep     # Question 2.(b): base case, sweep of c^L, tables, figures, checks
    python main.py --question Q2_quadratic --sweep  # Question 2.(c): same for c^Q
    python main.py --question Q2_quadratic --compare        # Question 2.(d): linear vs quadratic vs Question 1 model
    python main.py --question Q2_quadratic --nonseparable   # Question 2.(e) bonus: accumulated-deviation model
    python main.py --question Q3_battery    # Question 3.(g): battery, value of the battery, its sensitivity and figures
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import (AccumulatedDeviationModel, FlexibleConsumerModel, LinearDisutilityModel,
                       QuadraticDisutilityModel, Results, solve_extended_horizon)
from src.plotting import (plot_battery_schedule, plot_battery_sensitivity, plot_duals, plot_inputs, plot_lambda_range, plot_load_profiles,
                          plot_scenario_comparison, plot_schedule, plot_sweep)
from src.scenarios import scale_prices, scale_pv, set_battery, set_disutility, set_tariffs, shift_prices

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"     # every figure used in the report

# Question 2.(b) and 2.(c): the model of each case and the sweep of its disutility coefficient
Q2_CASES = {
    "Q2_linear": dict(
        model=LinearDisutilityModel, arg="linear", name="c_L", fig="q2b", sym="$c^L$", unit="DKK/kWh",
        values=np.round(np.arange(0.005, 3.2, 0.01), 3),     # offset by 0.005: no run lies on a price threshold
        report=[1.0, 1.43, 1.475, 1.55, 1.65, 1.80, 1.95, 2.05, 2.50],
        intervals=["< 1.41", "1.41-1.45", "1.45-1.50", "1.50-1.60", "1.60-1.70", "1.70-1.90", "1.90-2.00",
                   "2.00-2.10", "> 2.10"]),
    "Q2_quadratic": dict(
        model=QuadraticDisutilityModel, arg="quadratic", name="c_Q", fig="q2c", sym="$c^Q$", unit="DKK/kWh$^2$",
        values=np.round(np.arange(0.01, 3.21, 0.01), 3),
        report=[0.2, 0.5, 1.0, 1.43, 2.0, 3.0], intervals=None),
}


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    is_q2 = question in Q2_CASES            # Question 2 has its own figures (results/figures)
    if not is_q2:
        plot_inputs(data, save_to=out / "inputs.png")

    model = Q2_CASES.get(question, {}).get("model", FlexibleConsumerModel)(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    if not is_q2:
        plot_schedule(results, data, save_to=out / "schedule.png")
        plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_q3_battery(out: Path, results: Results) -> None:
    """Question 3.(g): value of the battery against the same case without battery (3.(d)), and its figure."""
    data = load_question(results.question)
    without = FlexibleConsumerModel(load_question("Q3")).build().solve()
    print(f"Value of the battery: {results.objective - without.objective:.4f} DKK/day "
          f"(objective {results.objective:.4f} with, {without.objective:.4f} without); "
          f"pi_end = {results.meta['pi_end']:.4f} DKK/kWh, final state of charge {results.hourly['soc'].iloc[-1]:.3f} kWh, "
          f"max overlap of charge and discharge {results.meta['overlap_max']:.1e} kWh/h")
    plot_battery_schedule(results, data, without, save_to=FIGURES_DIR / "q3g_battery_schedule.png")


# Question 3.(g).v: parameters varied one at a time (base case marked in the figure), with the
# function that applies each value to the data
Q3_BATTERY_SWEEPS = {
    "price_spread": dict(values=[0.0, 0.5, 1.0, 1.5, 2.0], label="price spread (x base, same mean)",
                         apply=lambda d, v: scale_prices(d, v, keep_mean=True)),
    "price_level": dict(values=[-0.5, 0.0, 0.5, 1.0], label="price level (DKK/kWh added, same spread)",
                        apply=shift_prices),
    "c_Q": dict(values=[0.02, 0.1, 0.3, 1.0, 3.0, 10.0], label="$c^Q$ (DKK/kWh$^2$)",
                apply=lambda d, v: set_disutility(d, quadratic=v)),
    "capacity": dict(values=[1.0, 2.0, 4.0, 8.0, 16.0], label="capacity $E^{bat}$ (kWh)",
                     apply=lambda d, v: set_battery(d, capacity_kWh=v)),
    "power": dict(values=[0.5, 1.0, 1.5, 3.0, 6.0], label="power $P^{ch} = P^{dis}$ (kW)",
                  apply=lambda d, v: set_battery(d, power_kW=v)),
    "efficiency": dict(values=[0.8, 0.85, 0.9, 0.95, 1.0], label=r"efficiency $\eta^{ch} = \eta^{dis}$",
                       apply=lambda d, v: set_battery(d, efficiency=v)),
}


def battery_run(data, **model_args) -> dict:
    """Solve ``data`` with and without its battery; value of the battery = difference of the objectives."""
    with_bat = FlexibleConsumerModel(data, **model_args).build().solve()
    without = FlexibleConsumerModel(replace(data, battery_capacity_kWh=None)).build().solve()
    hr = with_bat.hourly
    return dict(value_DKK=with_bat.objective - without.objective, pi_end=with_bat.meta["pi_end"],
                final_soc_kWh=hr["soc"].iloc[-1], discharged_kWh=hr["discharge"].sum(),
                overlap_max=with_bat.meta["overlap_max"], hourly=hr)


def run_q3_battery_sensitivity(out: Path) -> None:
    """Question 3.(g).v: value of the battery when one parameter changes. Saves q3g_sensitivity.csv and its figure."""
    base = load_question("Q3_battery")
    rows = []
    for name, sw in Q3_BATTERY_SWEEPS.items():
        for v in sw["values"]:
            r = battery_run(sw["apply"](base, v))
            rows.append(dict(parameter=name, value=v, **{k: x for k, x in r.items() if k != "hourly"}))
    table = pd.DataFrame(rows)
    table.to_csv(out / "q3g_sensitivity.csv", index=False)
    print("\nValue of the battery, one parameter at a time:\n", table.round(3).to_string(index=False))
    plot_battery_sensitivity(table, {n: sw["label"] for n, sw in Q3_BATTERY_SWEEPS.items()},
                             save_to=FIGURES_DIR / "q3g_sensitivity.png")


def end_of_day_alternatives(d) -> dict[str, dict]:
    """The end-of-day representations compared in 3.(g).v, as arguments of FlexibleConsumerModel.

    "trend" extrapolates the last two import prices of the day by one hour, a heuristic that needs no forecast."""
    p_imp = d.energy_price + d.import_tariff
    trend = p_imp[-1] + (p_imp[-1] - p_imp[-2])
    return {"ours": {},
            "trend": dict(pi_end=trend),
            "last price": dict(pi_end=p_imp[-1]),
            "last price / eta_ch": dict(pi_end=p_imp[-1] / d.battery_charging_efficiency),
            "free end": dict(end_of_day="free"),
            "cyclic": dict(end_of_day="cyclic")}


def schedule_change(hourly, ref_hourly) -> float:
    """Largest change of charging or discharging in any hour (kWh/h) between two schedules."""
    return max(np.abs(np.asarray(hourly[c])[:24] - np.asarray(ref_hourly[c])).max() for c in ("charge", "discharge"))


def run_q3_end_of_day(out: Path) -> None:
    """Question 3.(g).v: our pi^end against the alternatives of 3.(g).i and against the extended-horizon benchmark.

    Saves q3g_end_of_day.csv (one row per alternative and the benchmark), q3g_water_value.csv (value of the energy
    stored at the end of the day) and q3g_pi_end_sweep.csv (the decisions for other values of pi^end).
    """
    base = load_question("Q3_battery")
    ref = battery_run(base)
    rows = []
    for name, args in end_of_day_alternatives(base).items():
        r = battery_run(base, **args)
        rows.append(dict(end=name, pi_end=r["pi_end"], final_soc_kWh=r["final_soc_kWh"], value_DKK=r["value_DKK"],
                         schedule_change=schedule_change(r["hourly"], ref["hourly"])))

    # Benchmark: day D + hours 0-10 of D+1, and the value W(x) of starting D+1 with x kWh in the battery
    ext = solve_extended_horizon(base)
    xs = np.arange(0.0, base.battery_capacity_kWh + 1e-9, 1.0)
    W = np.array([solve_extended_horizon(base, today=False, start_soc=float(x))["objective"] for x in xs])
    water = pd.DataFrame({"soc_kWh": xs, "value_DKK": W - W[0], "marginal_value_DKK_per_kWh": np.r_[np.nan, np.diff(W)]})
    water.to_csv(out / "q3g_water_value.csv", index=False)
    # Exact value of the battery: our schedule with the stored-energy term pi^end (e_23 - e^init) replaced
    # by the exact value W(e_23) - W(e^init)
    e23, e0 = ref["final_soc_kWh"], base.battery_initial_soc_kWh
    exact = ref["value_DKK"] - ref["pi_end"] * (e23 - e0) + np.interp(e23, xs, W) - np.interp(e0, xs, W)
    rows.insert(1, dict(end="benchmark (exact value)", pi_end=np.nan, final_soc_kWh=ext["soc"][23], value_DKK=exact,
                        schedule_change=schedule_change(ext, ref["hourly"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "q3g_end_of_day.csv", index=False)
    print(f"\nEnd of the day (benchmark: {ext['n_variables']} variables):\n", table.round(3).to_string(index=False))
    print("Value of the energy stored at the end of the day (benchmark):\n", water.round(3).to_string(index=False))

    # Thresholds: the smallest pi^end that changes the decisions, and the smallest that makes the battery
    # recharge in the last hour
    sweep = []
    for p in np.round(np.arange(1.40, 2.001, 0.01), 2):
        r = battery_run(base, pi_end=float(p))
        sweep.append(dict(pi_end=p, final_soc_kWh=r["final_soc_kWh"], value_DKK=r["value_DKK"],
                          schedule_change=schedule_change(r["hourly"], ref["hourly"]),
                          charge_last_hour=r["hourly"]["charge"].iloc[-1]))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "q3g_pi_end_sweep.csv", index=False)
    print(f"Decisions change for pi_end >= {sweep.loc[sweep['schedule_change'] > 0.05, 'pi_end'].min():.2f} DKK/kWh; "
          f"the battery recharges in the last hour for pi_end >= "
          f"{sweep.loc[sweep['charge_last_hour'] > 0.05, 'pi_end'].min():.2f} DKK/kWh")


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_q1_validation(question: str, out: Path, results: Results) -> None:
    """Question 1.(f): compare the solved schedule with the price ladder of 1.(d).ii-iii, hour by hour,
    and check whether the dual of the power balance (DKK/kWh) is unique.

    The ladder is derived for case A (c_PV < u); in case B the hours where it fails show how the
    ladder changes. Saves <question>_validation.csv (one row per hour) and lambda_range.png to ``out``.
    """
    data = load_question(question)
    hr = results.hourly
    u = data.consumption_utility
    c_pv = data.pv_marginal_cost
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff

    model = FlexibleConsumerModel(data).build()  # solved again below with a small change
    step = 0.001  # kWh

    rows = []
    for t in data.hours:
        pv_max = data.pv_available[t]

        # Price ladder of 1.(d): expected load, PV and dual of the balance in this hour
        if u >= p_imp[t]:  # regime 2: consuming is worth more than buying -> max load, import
            regime = 2
            load = data.load_max_kWh
            pv = pv_max if c_pv < p_imp[t] else 0  # PV is used only if cheaper than importing
            lam = p_imp[t]
        elif u <= p_exp[t]:  # regime 1: selling is worth more than consuming -> min load, export
            regime = 1
            load = data.load_min_kWh
            pv = pv_max
            lam = p_exp[t]
        else:  # regime 3: neither -> no trading, the load follows the own PV
            regime = 3
            load = pv_max
            pv = pv_max
            lam = u

        # The dual of the balance is the value of one extra kWh in this hour. Give the consumer a
        # little extra energy, then take a little away. If the two values differ, every value in
        # between is a valid dual, so the dual is not unique.
        balance = model.con["balance"][t]
        balance.RHS = step
        model.m.optimize()
        lam_low = (model.m.ObjVal - results.objective) / step
        balance.RHS = -step
        model.m.optimize()
        lam_high = (results.objective - model.m.ObjVal) / step
        balance.RHS = 0

        rows.append({
            "hour": t, "p_imp": p_imp[t], "p_exp": p_exp[t], "pv_max": pv_max, "regime": regime,
            "load_ladder": load, "load": hr.loc[t, "load"], "pv_ladder": pv, "pv": hr.loc[t, "pv"],
            "import": hr.loc[t, "import"], "export": hr.loc[t, "export"],
            "lambda_ladder": lam, "lambda": hr.loc[t, "dual_balance"],
            "lambda_low": lam_low, "lambda_high": lam_high,
        })

    table = pd.DataFrame(rows).set_index("hour").round(4)
    table.to_csv(out / f"{question}_validation.csv")
    plot_lambda_range(table, data, save_to=out / "lambda_range.png")

    differs = table.index[(table["load"] != table["load_ladder"]) | (table["pv"] != table["pv_ladder"])]
    not_unique = table.index[table["lambda_high"] > table["lambda_low"]]
    print(f"Price ladder matches the solver in {len(table) - len(differs)}/{len(table)} hours, differs in hours {list(differs)}")
    print(f"Dual of the balance is not unique in hours {list(not_unique)}")


### Question 2 ###
def run_metrics(res: Results, data, tol: float = 1e-6) -> dict:
    """Daily metrics of one Question 2 run: costs in DKK, energies in kWh, counts in hours."""
    hr = res.hourly
    dev = hr["load"] - data.reference_load
    at_bound = (hr["load"] <= data.load_min_kWh + tol) | (hr["load"] >= data.load_max_kWh - tol)
    return {
        "procurement_cost_DKK": res.meta["procurement_cost"],
        "disutility_DKK": res.meta["disutility"],
        "surplus_DKK": res.objective,
        "energy_kWh": hr["load"].sum(),
        "abs_deviation_kWh": dev.abs().sum(),
        "hours_above_reference": int((dev > tol).sum()),
        "hours_deviating": int((dev.abs() > tol).sum()),
        "hours_bound_binding": int((at_bound & (dev.abs() > tol)).sum()),
        "import_kWh": hr["import"].sum(),
        "export_kWh": hr["export"].sum(),
        "pv_used_kWh": hr["pv"].sum(),
    }


def predicted_load_linear(data, c: float) -> np.ndarray:
    """Hourly load predicted by the cost-benefit rule of 2.(b).ii for a linear disutility c^L = c."""
    p_imp, p_exp = data.energy_price + data.import_tariff, data.energy_price - data.export_tariff
    m_pv = np.minimum(p_imp, np.maximum(data.pv_marginal_cost, p_exp))   # marginal cost of the load covered by PV
    ref, pv, lmin = data.reference_load, data.pv_available, data.load_min_kWh
    return np.array([ref[t] if ref[t] <= lmin or c > p_imp[t]
                     else max(lmin, min(pv[t], ref[t])) if c > m_pv[t] else lmin for t in data.hours])


def predicted_load_quadratic(data, c: float) -> np.ndarray:
    """Hourly load predicted by the closed-form first-order conditions of 2.(c).ii for c^Q = c."""
    p_imp, p_exp = data.energy_price + data.import_tariff, data.energy_price - data.export_tariff
    m_pv = np.maximum(data.pv_marginal_cost, p_exp)
    ref, pv = data.reference_load, data.pv_available
    out = np.empty(data.n_hours)
    for t in data.hours:
        l_star = ref[t] - p_imp[t] / (2 * c)                              # above the PV: imported at p_imp
        if l_star < pv[t]:
            l_star = min(ref[t] - m_pv[t] / (2 * c), pv[t])               # up to the PV: marginal cost m_pv
        out[t] = min(max(l_star, data.load_min_kWh), data.load_max_kWh)
    return out


def run_q2_sweep(question: str, out: Path) -> None:
    """Question 2.(b) and 2.(c): base case, sweep of the disutility coefficient, report table, figures and checks.

    Saves sweep_<c>.csv and sweep_<c>_report_table.csv to ``out`` and the figures to results/figures.
    """
    cfg = Q2_CASES[question]
    base = load_question(question)

    def solve(c: float):
        d = set_disutility(base, **{cfg["arg"]: c})
        return cfg["model"](d).build().solve(), d

    base_res, _ = solve(getattr(base, f"{cfg['arg']}_disutility"))
    print(pd.Series(run_metrics(base_res, base)).round(3), "\n")
    plot_schedule(base_res, base, save_to=FIGURES_DIR / f"{cfg['fig']}_schedule.png")

    # sweep: metrics and hourly loads of every run
    rows, loads = [], {}
    for c in cfg["values"]:
        r, d = solve(c)
        rows.append({cfg["name"]: c, **run_metrics(r, d)})
        loads[c] = r.hourly["load"].to_numpy()
    sweep = pd.DataFrame(rows)
    loads = pd.DataFrame(loads, index=pd.Index(base.hours, name="hour"))
    sweep.round(3).to_csv(out / f"sweep_{cfg['name'].replace('_', '')}.csv", index=False)
    print(len(sweep), "runs")

    # table of the report: representative values of the coefficient
    report = pd.DataFrame([{cfg["name"]: c, **run_metrics(*solve(c))} for c in cfg["report"]])
    if cfg["intervals"]:
        report.insert(0, "interval", cfg["intervals"])
    report.round(3).to_csv(out / f"sweep_{cfg['name'].replace('_', '')}_report_table.csv", index=False)

    # figure and checks
    L, cs = loads.to_numpy(), loads.columns.to_numpy(dtype=float)
    changed = [j for j in range(1, L.shape[1]) if np.abs(L[:, j] - L[:, j - 1]).max() > 1e-6]
    tol = 1e-6
    assert sweep["hours_above_reference"].max() == 0, "the load is above the reference in some run"
    print(f"1. max hours above the reference over the sweep: {sweep['hours_above_reference'].max()}")
    if question == "Q2_linear":
        # price thresholds at which the schedule is predicted to change (2.(b).ii): m_PV and, where ref > PV, p_imp
        p_imp, p_exp = base.energy_price + base.import_tariff, base.energy_price - base.export_tariff
        cpv, ref, pv = base.pv_marginal_cost, base.reference_load, base.pv_available
        m_pv = np.minimum(p_imp, np.maximum(cpv, p_exp))
        thresholds = {}
        for t in base.hours:
            if ref[t] <= base.load_min_kWh:
                continue
            lab = "$c^{PV}$" if m_pv[t] == cpv else f"$p^{{exp}}_{{{t}}}$"
            thresholds.setdefault(round(float(m_pv[t]), 4), set()).add(lab)
            if ref[t] > pv[t]:
                thresholds.setdefault(round(float(p_imp[t]), 4), set()).add(f"$p^{{imp}}_{{{t}}}$")
        pd.DataFrame([{"between": f"{cs[j-1]:.3f} - {cs[j]:.3f}",
                       "hours": list(np.where(np.abs(L[:, j] - L[:, j - 1]) > tol)[0]),
                       "new load": list(np.round(L[np.abs(L[:, j] - L[:, j - 1]) > tol, j], 2))}
                      for j in changed]).to_csv(out / "sweep_cL_change_points.csv", index=False)
        both = ((base_res.hourly["dev_up"] > tol) & (base_res.hourly["dev_down"] > tol)).sum()
        err = max(np.abs(loads[c].to_numpy() - predicted_load_linear(base, c)).max() for c in loads.columns)
        at_threshold = all(any(cs[j - 1] < v < cs[j] for v in thresholds) for j in changed)
        print(f"2. hours with d+ > 0 and d- > 0 (base case): {both}")
        print(f"3. max |load - predicted load| over all hours and runs: {err:.1e}")
        print(f"4. every change point is at a predicted threshold: {at_threshold}")
        assert both == 0 and err < 1e-6 and at_threshold
        plot_sweep(sweep, "c_L", "$c^L$ (DKK/kWh)", "Sweep over $c^L$", step=True, thresholds=thresholds,
                   xlim=(1.0, 2.5), save_to=FIGURES_DIR / "q2b_sweep.png")
    else:
        err = max(np.abs(loads[c].to_numpy() - predicted_load_quadratic(base, c)).max() for c in loads.columns)
        moves = np.abs(np.diff(L[np.where(base.reference_load > 0)[0], :], axis=1))
        frac_moving = (moves > 1e-9).mean()
        check = QuadraticDisutilityModel(base).build()
        check.m.Params.Seed = 42                      # strictly convex QP: the optimal load is unique
        seed_diff = np.abs(check.solve().hourly["load"].to_numpy() - base_res.hourly["load"].to_numpy()).max()
        print(f"2. max |load - predicted load| over all hours and runs: {err:.1e}")
        print(f"3. fraction of (hour, step) pairs where the load moved: {frac_moving:.3f}; "
              f"largest single-step jump: {moves.max():.4f} kWh")
        print(f"4. max |load difference| after re-solving with another seed: {seed_diff:.1e}")
        assert err < 1e-3 and frac_moving > 0.7 and seed_diff < 1e-6
        plot_sweep(sweep, "c_Q", "$c^Q$ (DKK/kWh$^2$)", "Sweep over $c^Q$", save_to=FIGURES_DIR / "q2c_sweep.png")


def run_q2_comparison() -> None:
    """Question 2.(d): the linear and quadratic models (and the Question 1 model) with the same coefficient value.

    Saves comparison.csv and comparison_loads.csv to results/Q2_comparison and q2d_profiles.png to results/figures.
    """
    out = RESULTS_DIR / "Q2_comparison"
    out.mkdir(parents=True, exist_ok=True)
    base_lin, base_quad = load_question("Q2_linear"), load_question("Q2_quadratic")
    # The base case uses each model's own base coefficient; the scenarios use the same value in both models
    scenarios = {
        "Base (c^L=1.43, c^Q=1.00)": (base_lin.linear_disutility, base_quad.quadratic_disutility),
        "Low c = 0.50": (0.50, 0.50),
        "High c = 3.20": (3.20, 3.20),
    }
    runs, rows = {}, []
    for name, (cL, cQ) in scenarios.items():
        d_lin, d_quad = set_disutility(base_lin, linear=cL), set_disutility(base_quad, quadratic=cQ)
        # Question 1 model on the same prices and PV, with utility u^L = c^L and no reference profile
        d_q1 = replace(base_lin, consumption_utility=float(cL), linear_disutility=None)
        for model, build in [("Q1 utility", FlexibleConsumerModel(d_q1)), ("Linear", LinearDisutilityModel(d_lin)),
                             ("Quadratic", QuadraticDisutilityModel(d_quad))]:
            res = runs[(name, model)] = build.build().solve()
            hr, dev = res.hourly, res.hourly["load"] - base_lin.reference_load
            rows.append({"Scenario": name, "Model": model,
                         "Cost (DKK)": res.meta["procurement_cost"], "Energy (kWh)": hr["load"].sum(),
                         "|dev| (kWh)": dev.abs().sum(), "Import (kWh)": hr["import"].sum(),
                         "Export (kWh)": hr["export"].sum(),
                         "Hours at ref": int(((dev.abs() < 1e-4) & (base_lin.reference_load > 0)).sum()),
                         "Hours at 0": int(((hr["load"] < 1e-4) & (base_lin.reference_load > 1e-4)).sum())})
    comparison = pd.DataFrame(rows).set_index(["Scenario", "Model"]).round(3)
    comparison.to_csv(out / "comparison.csv")
    pd.DataFrame({f"{n} | {m}": r.hourly["load"] for (n, m), r in runs.items()}).round(3).to_csv(out / "comparison_loads.csv")
    print(comparison, "\n")
    panels = {n.replace("c^L", "$c^L$").replace("c^Q", "$c^Q$"):
              {"linear": runs[(n, "Linear")].hourly["load"], "quadratic": runs[(n, "Quadratic")].hourly["load"]}
              for n in scenarios}
    plot_load_profiles(panels, base_lin, save_to=FIGURES_DIR / "q2d_profiles.png")


def run_q2_nonseparable() -> None:
    """Question 2.(e) bonus: accumulated-deviation model, comparison with 2.(b) and 2.(c), price-order experiment.

    Saves the base case and experiment.csv to results/Q2_nonseparable and q2e_*.png to results/figures.
    """
    out = RESULTS_DIR / "Q2_nonseparable"
    out.mkdir(parents=True, exist_ok=True)
    base_lin, base_quad = load_question("Q2_linear"), load_question("Q2_quadratic")
    models = {"Linear": (base_lin, LinearDisutilityModel), "Quadratic": (base_quad, QuadraticDisutilityModel),
              "Accumulated": (base_quad, AccumulatedDeviationModel)}

    # check: with rho = 0 the model is 2.(c) with c^Q = cQ + cS
    chk = AccumulatedDeviationModel(base_quad, cQ=0.5, cS=0.5, rho=0.0).build().solve()
    diff = np.abs(chk.hourly["load"].to_numpy() - QuadraticDisutilityModel(base_quad).build().solve().hourly["load"].to_numpy()).max()
    print(f"max |load difference| vs 2.(c) for rho = 0: {diff:.1e}")
    assert diff < 1e-4                                  # solver tolerance of the QP

    # base case: schedule, load profiles of the three models, comparison table
    res = {mn: model(d).build().solve() for mn, (d, model) in models.items()}
    res["Accumulated"].save(out)
    plot_schedule(replace(res["Accumulated"], question=out.name), base_quad, save_to=FIGURES_DIR / "q2e_schedule.png")
    plot_load_profiles({"Base case: load profiles of the three models":
                        {"linear (b)": res["Linear"].hourly["load"], "quadratic (c)": res["Quadratic"].hourly["load"],
                         "accumulated (e)": res["Accumulated"].hourly["load"]}},
                       base_quad, save_to=FIGURES_DIR / "q2e_profiles.png")
    base_table = pd.DataFrame({mn: {**run_metrics(r, models[mn][0])} for mn, r in res.items()}).T
    print(base_table[["procurement_cost_DKK", "energy_kWh", "abs_deviation_kWh", "import_kWh", "export_kWh",
                      "disutility_DKK", "hours_above_reference"]].round(2), "\n")

    # experiment: same mean price and amplitude, only the order of the prices changes
    t = np.arange(24)
    pbar, a = base_quad.energy_price.mean(), 0.8                       # mean price and +-amplitude (DKK/kWh)
    profiles = {"Flat": np.full(24, pbar),
                "Alternating": pbar + a * (-1.0) ** t,                                # high, low, high, low, ...
                "Blocks of 4 h": pbar + a * np.where((t // 4) % 2 == 0, 1.0, -1.0)}   # 4 h high, 4 h low, ...
    rows, runs = [], {}
    for pn, p in profiles.items():
        for mn, (d0, model) in models.items():
            d = replace(d0, energy_price=p)
            r = runs[(pn, mn)] = model(d).build().solve()
            dev = r.hourly["load"] - d.reference_load
            rows.append({"Profile": pn, "Model": mn, "Surplus (DKK)": r.objective, "|dev| (kWh)": dev.abs().sum(),
                         "Hours above ref.": int((dev > 1e-6).sum())})
    exp = pd.DataFrame(rows).set_index(["Profile", "Model"])
    flat = exp.loc["Flat", "Surplus (DKK)"]                              # each model against its own flat-price run
    exp["Gain vs flat (DKK)"] = exp["Surplus (DKK)"] - exp.index.get_level_values("Model").map(flat).to_numpy()
    exp.round(3).to_csv(out / "experiment.csv")
    print(exp.round(2), "\n")
    names = ["Alternating", "Blocks of 4 h"]
    plot_load_profiles({f"{pn} prices": {mn.lower(): runs[(pn, mn)].hourly["load"] for mn in models} for pn in names},
                       base_quad, prices={f"{pn} prices": profiles[pn] for pn in names},
                       save_to=FIGURES_DIR / "q2e_experiment.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    parser.add_argument("--validate", action="store_true", help="Question 1.(f): price ladder and dual check")
    parser.add_argument("--q2", action="store_true", help="run all of Question 2: both sweeps, comparison and bonus")
    parser.add_argument("--sweep", action="store_true", help="Question 2.(b)/(c): sweep of the disutility coefficient (Q2_linear or Q2_quadratic)")
    parser.add_argument("--compare", action="store_true", help="Question 2.(d): linear vs quadratic vs Question 1 model")
    parser.add_argument("--nonseparable", action="store_true", help="Question 2.(e) bonus: accumulated-deviation model")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    if args.q2:
        for question in Q2_CASES:
            out = RESULTS_DIR / question
            out.mkdir(parents=True, exist_ok=True)
            run_base_case(question, out, False)
            run_q2_sweep(question, out)
        run_q2_comparison()
        run_q2_nonseparable()
        print(f"\nOutputs written to {RESULTS_DIR} and {FIGURES_DIR}")
        return

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)

    base = run_base_case(args.question, out, args.show)
    if base is not None and "soc" in base.hourly:      # Question 3.(g): a case with a battery
        run_q3_battery(out, base)
        run_q3_battery_sensitivity(out)
        run_q3_end_of_day(out)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    if args.validate and base is not None:
        run_q1_validation(args.question, out, base)
    if args.sweep and args.question in Q2_CASES:
        run_q2_sweep(args.question, out)
    if args.compare:
        run_q2_comparison()
    if args.nonseparable:
        run_q2_nonseparable()
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
