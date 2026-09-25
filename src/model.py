"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` implements the hourly prosumer problem of Question 1 (linear utility, PV, grid with
tariffs); the other questions are variations of it (a different objective, an extra constraint).
Subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            if self.meta.get("utility") is not None:
                f.write(f"utility   : {self.meta['utility']:.4f} DKK\n")
            if self.meta.get("procurement_cost") is not None:
                f.write(f"cost      : {self.meta['procurement_cost']:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        split = ""
        if self.meta.get("utility") is not None and self.meta.get("procurement_cost") is not None:
            split = f" (utility {self.meta['utility']:.2f} - cost {self.meta['procurement_cost']:.2f})"
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK{split}\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Declare the variables, objective and constraints of the Question 1 problem.

        Over the 24 hours, maximise the net utility
            sum_t ( uL * load_t - p_imp_t * import_t + p_exp_t * export_t - cPV * pv_t )
        subject to the hourly power balance, the load bounds, the PV availability limits and the
        non-negativity of imports and exports. Every family is stored in ``self.var`` /
        ``self.con`` so that ``solve()`` returns its primal and dual values automatically.
        """
        d, m, T = self.data, self.m, self.T

        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff

        # --- Legend: report (LaTeX) -> code ---------------------------------------------
        # Parameters (d = self.data; arrays are indexed by hour, scalars are not)
        #   p_t            d.energy_price[t]              DKK/kWh
        #   tau^imp        d.import_tariff                DKK/kWh
        #   tau^exp        d.export_tariff                DKK/kWh
        #   p^imp_t        p_imp[t]                       DKK/kWh   (p_t + tau^imp)
        #   p^exp_t        p_exp[t]                       DKK/kWh   (p_t - tau^exp)
        #   u^L            d.consumption_utility          DKK/kWh   (scalar)
        #   c^PV           d.pv_marginal_cost             DKK/kWh   (scalar)
        #   L^min, L^max   d.load_min_kWh, d.load_max_kWh kWh/h     (scalars)
        #   PV^max_t       d.pv_available[t]              kWh/h
        # Variables (declared below, used as self.var["<name>"][t])
        #   l_t            self.var["load"][t]            kWh/h
        #   q^PV_t         self.var["pv"][t]              kWh/h
        #   q^imp_t        self.var["import"][t]          kWh/h
        #   q^exp_t        self.var["export"][t]          kWh/h
        # Constraint families (used as self.con["<name>"]; their duals come out as dual_<name>)
        #   balance, load_lo, load_up, pv_lo, pv_up, imp_nonneg, exp_nonneg
        #   -> lambda_t, mu^L_lo, mu^L_up, mu^PV_lo, mu^PV_up, mu^imp, mu^exp of Question 1.(b)

        # --- Decision variables --------------------------------------------------------
        # All four families are free (lb = -inf): every bound is an explicit constraint below, so
        # that each one has a dual value (a variable bound would report its sensitivity in RC, not
        # in Pi). The names "import", "export", "load", "pv" are the ones src/plotting.py expects.
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="import")
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="export")
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv")

        # --- Objective: daily net utility (DKK), maximised --------------------------------
        m.setObjective(
            gp.quicksum(
                d.consumption_utility * self.var["load"][t]
                - p_imp[t] * self.var["import"][t]
                + p_exp[t] * self.var["export"][t]
                - d.pv_marginal_cost * self.var["pv"][t]
                for t in T),
            GRB.MAXIMIZE)

        # --- Constraints -------------------------------------------------------------
        # Sign convention (same as Question 1.(b), primal feasibility of the KKT conditions):
        # the balance is written as h(x) = 0 and every inequality as g(x) <= 0, term by term as in
        # the report. The objective is a maximisation, so Gurobi's dual Pi = d(objective)/d(rhs) is
        # >= 0 for every "<= 0" constraint. Hence, in results.hourly, dual_<name> is directly the
        # multiplier of the report: lambda_t (free) for the balance, mu >= 0 for the inequalities.
        # (A constraint written as ">= 0" would return the opposite sign, <= 0.)
        # h: balance                     l - q^PV - q^imp + q^exp = 0
        self.con["balance"] = m.addConstrs(
            (self.var["load"][t] - self.var["pv"][t] - self.var["import"][t] + self.var["export"][t] == 0 for t in T),
            name="balance")
        # g: load bounds                 L^min - l <= 0   and   l - L^max <= 0
        self.con["load_lo"] = m.addConstrs(
            (d.load_min_kWh - self.var["load"][t] <= 0 for t in T), name="load_lo")
        self.con["load_up"] = m.addConstrs(
            (self.var["load"][t] - d.load_max_kWh <= 0 for t in T), name="load_up")
        # g: PV limits                   -q^PV <= 0   and   q^PV - PV^max <= 0
        self.con["pv_lo"] = m.addConstrs(
            (-self.var["pv"][t] <= 0 for t in T), name="pv_lo")
        self.con["pv_up"] = m.addConstrs(
            (self.var["pv"][t] - d.pv_available[t] <= 0 for t in T), name="pv_up")
        # g: non-negativity of grid flows   -q^imp <= 0   and   -q^exp <= 0
        self.con["imp_nonneg"] = m.addConstrs(
            (-self.var["import"][t] <= 0 for t in T), name="imp_nonneg")
        self.con["exp_nonneg"] = m.addConstrs(
            (-self.var["export"][t] <= 0 for t in T), name="exp_nonneg")

        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: call build() before solve(), and make sure build() adds them."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        """Collect the primal values, duals, objective and its split into utility and cost.

        ``hourly`` has one column per variable family and one ``dual_<name>`` column per constraint
        family; ``meta['utility']`` and ``meta['procurement_cost']`` (DKK) add up to the objective.
        """
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        # objective = utility - procurement cost
        utility = None if d.consumption_utility is None else d.consumption_utility * hourly["load"].sum()
        p_imp = hourly["price"] + d.import_tariff
        p_exp = hourly["price"] - d.export_tariff
        cost = (p_imp * hourly["import"] - p_exp * hourly["export"] + d.pv_marginal_cost * hourly["pv"]).sum()

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars, "utility": utility, "procurement_cost": cost},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
