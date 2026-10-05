"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios
    python main.py --question Q1_caseB --validate   # Question 1.(f): price ladder and dual check

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_lambda_range, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    parser.add_argument("--validate", action="store_true", help="Question 1.(f): price ladder and dual check")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    if args.validate and base is not None:
        run_q1_validation(args.question, out, base)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
