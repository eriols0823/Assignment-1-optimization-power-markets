"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} (net utility {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)


def plot_lambda_range(table, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Dual of the power balance per hour (DKK/kWh): the value from Gurobi (line) and all valid values (shaded)."""
    h = data.hours
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(h, table["lambda_low"], table["lambda_high"], step="mid", color="C0", alpha=0.3, label="all valid values")
    ax.step(h, table["lambda"], where="mid", color="C0", lw=2, label="value from Gurobi")
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", label="import price")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", label="export price")
    ax.axhline(data.consumption_utility, color="red", lw=1, label="utility")
    ax.axhline(data.pv_marginal_cost, color="orange", lw=1, label="PV cost")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual of the power balance - {data.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


# ---------------------------------------------------------------------- Question 2
MODEL_COLORS = {"linear": "C3", "quadratic": "C0", "accumulated": "C2"}   # colour by the first word of a label


def plot_sweep(sweep, x: str, xlabel: str, title: str, step: bool = False, thresholds: dict | None = None,
               xlim: tuple | None = None, save_to: Path | str | None = None) -> plt.Figure:
    """Sweep of a disutility coefficient: energy and deviation (top), procurement cost and disutility (bottom).

    ``step`` draws piecewise-constant curves (linear disutility); ``thresholds`` maps a coefficient value to the
    set of labels of the price it equals, drawn as dotted vertical lines.
    """
    draw = (lambda ax, *a, **k: ax.step(*a, where="post", **k)) if step else (lambda ax, *a, **k: ax.plot(*a, **k))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    draw(ax1, sweep[x], sweep["energy_kWh"], color="C0", lw=2, label="daily energy consumed")
    draw(ax1, sweep[x], sweep["abs_deviation_kWh"], color="C1", lw=2, label="total absolute deviation")
    ax1.set(ylabel="kWh", title=title)
    draw(ax2, sweep[x], sweep["procurement_cost_DKK"], color="C3", lw=2, label="procurement cost")
    draw(ax2, sweep[x], sweep["disutility_DKK"], color="C4", lw=2, label="total disutility")
    ax2.set(xlabel=xlabel, ylabel="DKK")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if xlim:
            ax.set_xlim(*xlim)
        for v in thresholds or {}:
            ax.axvline(v, color="grey", lw=0.8, ls=":")
    for v, labs in (thresholds or {}).items():
        ax1.annotate(" = ".join(sorted(labs)), (v, 1.0), xycoords=("data", "axes fraction"),
                     rotation=90, fontsize=7, va="top", ha="right", color="grey")
    return _finish(fig, save_to)


def plot_load_profiles(panels: dict[str, dict], data: InputData, prices: dict | None = None,
                       save_to: Path | str | None = None) -> plt.Figure:
    """One panel per entry of ``panels`` ({title: {label: hourly load}}): the loads of several models against the
    reference load and the available PV. ``prices`` ({title: hourly price}) adds the price on a second axis."""
    n, h = len(panels), np.arange(data.n_hours)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n + 0.5), sharex=True, sharey=True, squeeze=False)
    for ax, (title, loads) in zip(axes[:, 0], panels.items()):
        ax.fill_between(h, data.pv_available, step="mid", color="orange", alpha=0.2, lw=0, label="PV available")
        ax.step(h, data.reference_load, where="mid", color="0.55", lw=4, label="reference")   # thick: stays visible under the others
        for label, load in loads.items():
            ax.step(h, load, where="mid", color=MODEL_COLORS[label.split()[0]], lw=2, label=label)
        ax.set_title(title, fontsize=11); ax.set_ylabel("kWh/h"); ax.grid(alpha=0.3)
        ax.set_xticks(range(0, data.n_hours, 2))
        if prices:
            ax2 = ax.twinx()
            ax2.step(h, prices[title], where="mid", color="k", lw=0.8, ls=":")
            ax2.set_ylabel("price (DKK/kWh)")
    axes[-1, 0].set_xlabel("hour")
    axes[0, 0].legend(fontsize=9, loc="upper left")
    return _finish(fig, save_to)


def plot_battery_schedule(results: Results, data: InputData, results_without: Results | None = None,
                          save_to: Path | str | None = None) -> plt.Figure:
    """Question 3.(g): load, charging and discharging (top) and state of charge with the import price (bottom).

    ``results_without`` (optional) is the same case without battery (Question 3.(d)), drawn as a dashed load.
    The state of charge e_t is the energy at the END of hour t, so it is drawn at t + 0.5, starting from
    the initial state of charge at -0.5.
    """
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    if results_without is not None:
        ax1.step(h, results_without.hourly["load"], where="mid", color="grey", ls="--", label="load without battery")
    ax1.step(h, hr["load"], where="mid", color="C0", label="load with battery")
    if "reference_load" in hr:
        ax1.step(h, hr["reference_load"], where="mid", color="black", lw=0.8, label="reference load")
    ax1.bar(h, hr["charge"], color="C2", alpha=0.6, label="charge")
    ax1.bar(h, -hr["discharge"], color="C3", alpha=0.6, label="discharge (negative)")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set(ylabel="kWh/h", title=f"Schedule with battery - {results.question} (objective {results.objective:.2f} DKK)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)

    ax2.plot(np.r_[-0.5, h + 0.5], np.r_[data.battery_initial_soc_kWh, hr["soc"]], color="C4", marker="o", ms=3,
             label="state of charge at the end of the hour")
    ax2.axhline(data.battery_capacity_kWh, color="C4", ls=":", lw=0.8, label="capacity")
    ax2.set(xlabel="hour", ylabel="kWh")
    ax2b = ax2.twinx()
    ax2b.step(h, hr["price"] + data.import_tariff, where="mid", color="grey", ls=":", label="import price")
    ax2b.set_ylabel("DKK/kWh")
    lines, labels = ax2.get_legend_handles_labels()
    l2, lb2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + l2, labels + lb2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    return _finish(fig, save_to)


def plot_battery_sensitivity(table, labels: dict[str, str], save_to: Path | str | None = None) -> plt.Figure:
    """Question 3.(g).v: value of the battery against each varied parameter, one panel per parameter.

    ``table`` has the columns parameter, value and value_DKK; ``labels`` maps each parameter to its axis label.
    The base case (value of the battery 3.27 DKK) is the dotted line in every panel.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    base = table.loc[(table["parameter"] == "capacity") & (table["value"] == 4.0), "value_DKK"].iloc[0]
    for ax, (name, label) in zip(axes.flat, labels.items()):
        sub = table[table["parameter"] == name]
        ax.plot(sub["value"], sub["value_DKK"], color="C0", lw=2, marker="o", ms=6)
        ax.axhline(base, color="grey", lw=0.8, ls=":")
        if name == "c_Q":
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("value of the battery (DKK/day)")
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    return _finish(fig, save_to)
