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

from dataclasses import dataclass, field, replace
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


def terminal_value(d: InputData) -> float:
    """Value pi^end (DKK/kWh) of the energy left in the battery at the end of the day (Question 3.(g).i).

    A kWh left at midnight is worth what it would cost to replace it the following day before it
    is needed: the cheapest effective import price of the night hours before the morning price
    peak of day D+1, divided by the charging efficiency. It is capped by what the kWh saves when
    discharged at the most expensive hour of day D+1:

        t_peak = argmax_{t in 0..11} p_{D+1,t},   night = {0, ..., t_peak - 1}
        pi_end = min( min_{t in night} p^imp_{D+1,t} / eta_ch ,  eta_dis * max_t p^imp_{D+1,t} )
    """
    if d.energy_price_next_day is None:
        raise ValueError(f"{d.question}: the input data has no next-day prices to compute pi^end")
    p_next = np.asarray(d.energy_price_next_day)
    p_imp_next = p_next + d.import_tariff
    t_peak = int(np.argmax(p_next[:12]))                 # morning price peak of the following day
    night = p_imp_next[:max(t_peak, 1)]                  # night hours before it (at least hour 0)
    replace_cost = night.min() / d.battery_charging_efficiency
    use_value = d.battery_discharging_efficiency * p_imp_next.max()
    return float(min(replace_cost, use_value))


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False,
                 end_of_day: str = "value", pi_end: float | None = None):
        """``end_of_day`` and ``pi_end`` only matter with a battery (Question 3.(g)). ``end_of_day`` is
        "value" (our choice: + pi^end (e_23 - e^init) in the objective, pi^end from terminal_value() unless
        ``pi_end`` is given), "cyclic" (e_23 >= e^init) or "free" (no condition); the last two are the
        alternatives of 3.(g).i, compared in 3.(g).v."""
        self.data = data
        self.end_of_day = end_of_day
        self.pi_end = pi_end if end_of_day == "value" else 0.0     # the alternatives give no value to e_23
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
        has_battery = d.battery_capacity_kWh is not None    # Question 3.(g) only

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
        # Battery (Question 3.(g), only when the data has one)
        #   E^bat, P^ch, P^dis      d.battery_capacity_kWh, d.battery_max_charge_kW, d.battery_max_discharge_kW
        #   eta^ch, eta^dis         d.battery_charging_efficiency, d.battery_discharging_efficiency
        #   e^init, pi^end          d.battery_initial_soc_kWh, terminal_value(d)
        #   p^ch_t, p^dis_t, e_t    self.var["charge"][t], self.var["discharge"][t], self.var["soc"][t]
        #   constraints: soc_dyn (h = 0), ch_lo, ch_up, dis_lo, dis_up, soc_lo, soc_up

        # --- Decision variables --------------------------------------------------------
        # All four families are free (lb = -inf): every bound is an explicit constraint below, so
        # that each one has a dual value (a variable bound would report its sensitivity in RC, not
        # in Pi). The names "import", "export", "load", "pv" are the ones src/plotting.py expects.
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="import")
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="export")
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv")
        if has_battery:
            self.var["charge"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="charge")
            self.var["discharge"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="discharge")
            self.var["soc"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="soc")

        # --- Objective: daily net utility (DKK), maximised --------------------------------
        # The consumption utility is only given in Question 1, so in the data it can be empty
        # (None). We save it as u_L and stop with a clear message if it is missing, so that below
        # we are sure we multiply by a number. This also avoids the VS Code warning that None
        # cannot be multiplied.
        u_L = d.consumption_utility
        if u_L is not None:                         # Question 1: linear utility of the load
            m.setObjective(
                gp.quicksum(
                    u_L * self.var["load"][t]
                    - p_imp[t] * self.var["import"][t]
                    + p_exp[t] * self.var["export"][t]
                    - d.pv_marginal_cost * self.var["pv"][t]
                    for t in T),
                GRB.MAXIMIZE)
        elif d.quadratic_disutility is not None:    # Questions 2.(c) and 3: quadratic disutility
            c_Q, ref = d.quadratic_disutility, d.reference_load
            m.setObjective(
                gp.quicksum(
                    - c_Q * (self.var["load"][t] - ref[t]) * (self.var["load"][t] - ref[t])
                    - p_imp[t] * self.var["import"][t]
                    + p_exp[t] * self.var["export"][t]
                    - d.pv_marginal_cost * self.var["pv"][t]
                    for t in T),
                GRB.MAXIMIZE)
        else:
            raise ValueError(f"{d.question}: the input data has no consumption utility or quadratic disutility")

        # Question 3.(g): value of the net change of stored energy, + pi^end * (e_23 - e^init).
        # The constant -pi^end * e^init does not change the decisions, but keeps the objective
        # comparable with the case without battery (the term is zero if the battery ends as it started).
        if has_battery:
            if self.pi_end is None:
                self.pi_end = terminal_value(d)
            m.update()                   # without it, getObjective() would still return the empty objective
            m.setObjective(
                m.getObjective() + self.pi_end * (self.var["soc"][T[-1]] - d.battery_initial_soc_kWh),
                GRB.MAXIMIZE)

        # --- Constraints -------------------------------------------------------------
        # Sign convention (same as Question 1.(b), primal feasibility of the KKT conditions):
        # the balance is written as h(x) = 0 and every inequality as g(x) <= 0, term by term as in
        # the report. The objective is a maximisation, so Gurobi's dual Pi = d(objective)/d(rhs) is
        # >= 0 for every "<= 0" constraint. Hence, in results.hourly, dual_<name> is directly the
        # multiplier of the report: lambda_t (free) for the balance, mu >= 0 for the inequalities.
        # (A constraint written as ">= 0" would return the opposite sign, <= 0.)
        # h: balance                     l - q^PV - q^imp + q^exp (+ p^ch - p^dis with a battery) = 0
        def battery_net(t):              # energy the battery takes from the balance in hour t
            return self.var["charge"][t] - self.var["discharge"][t] if has_battery else 0
        self.con["balance"] = m.addConstrs(
            (self.var["load"][t] - self.var["pv"][t] - self.var["import"][t] + self.var["export"][t]
             + battery_net(t) == 0 for t in T),
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
        # g: minimum daily energy (Question 3 only)   E^min - sum_t l_t <= 0   -> mu^E
        if d.min_daily_energy_kWh is not None:
            self.con["min_energy"] = m.addConstr(
                d.min_daily_energy_kWh - gp.quicksum(self.var["load"][t] for t in T) <= 0, name="min_energy")

        # Battery (Question 3.(g).i), same sign convention
        if has_battery:
            ch, dis, soc = self.var["charge"], self.var["discharge"], self.var["soc"]
            eta_ch, eta_dis = d.battery_charging_efficiency, d.battery_discharging_efficiency
            # h: state of charge   e_t - e_{t-1} - eta^ch p^ch_t + p^dis_t / eta^dis = 0,  e_{-1} = e^init
            # (its dual is the value of a kWh stored in hour t)
            self.con["soc_dyn"] = m.addConstrs(
                (soc[t] - (soc[t - 1] if t > T[0] else d.battery_initial_soc_kWh)
                 - eta_ch * ch[t] + dis[t] / eta_dis == 0 for t in T),
                name="soc_dyn")
            # g: power limits      -p^ch <= 0,  p^ch - P^ch <= 0,  -p^dis <= 0,  p^dis - P^dis <= 0
            self.con["ch_lo"] = m.addConstrs((-ch[t] <= 0 for t in T), name="ch_lo")
            self.con["ch_up"] = m.addConstrs((ch[t] - d.battery_max_charge_kW <= 0 for t in T), name="ch_up")
            self.con["dis_lo"] = m.addConstrs((-dis[t] <= 0 for t in T), name="dis_lo")
            self.con["dis_up"] = m.addConstrs((dis[t] - d.battery_max_discharge_kW <= 0 for t in T), name="dis_up")
            # g: capacity          -e_t <= 0,  e_t - E^bat <= 0
            self.con["soc_lo"] = m.addConstrs((-soc[t] <= 0 for t in T), name="soc_lo")
            self.con["soc_up"] = m.addConstrs((soc[t] - d.battery_capacity_kWh <= 0 for t in T), name="soc_up")
            # g: cyclic end, only for the alternative of 3.(g).v     e^init - e_23 <= 0
            if self.end_of_day == "cyclic":
                self.con["soc_end"] = m.addConstr(d.battery_initial_soc_kWh - soc[T[-1]] <= 0, name="soc_end")

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
        battery = {}
        if "soc" in hourly:                                  # Question 3.(g)
            battery = {"pi_end": self.pi_end,
                       "terminal_value": self.pi_end * (hourly["soc"].iloc[-1] - d.battery_initial_soc_kWh),
                       "overlap_max": float(np.minimum(hourly["charge"], hourly["discharge"]).max())}

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars, "utility": utility, "procurement_cost": cost, **battery},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")


# ------------------------------------------------------------- Question 3.(g).v
def solve_extended_horizon(d: InputData, hours_next: int = 11, today: bool = True,
                           start_soc: float | None = None) -> dict:
    """Benchmark of 3.(g).v: the battery problem of 3.(g) over day D and the first ``hours_next`` hours of D+1.

    Same objective and constraints as ``FlexibleConsumerModel`` with a battery, in a compact form so that the
    longer horizon fits the size-limited Gurobi licence (200 variables in a QP): the state of charge is an
    expression of the charging and discharging variables instead of a variable, PV variables exist only in
    hours with PV, and the bounds are variable bounds (no duals are needed here). With the base data,
    hours_next = 11 (hours 0-10 of D+1, including the morning price peak) gives 196 variables.

    E^min is imposed on day D only, since day D+1 is incomplete. The energy left at the end of the horizon has
    no value; the battery has used it at the morning peak by then. With ``today=False`` only the hours of D+1
    are solved, starting from ``start_soc``: the objective as a function of start_soc is the value of the energy
    stored at the end of day D (the "water value" curve of 3.(g).i).

    Returns the objective (DKK), the number of variables and the hourly charge, discharge and state of charge.
    """
    nxt = slice(0, hours_next)
    if today:
        price = np.r_[d.energy_price, d.energy_price_next_day[nxt]]
        pv_max = np.r_[d.pv_available, d.pv_available_next_day[nxt]]
        ref = np.r_[d.reference_load, d.reference_load_next_day[nxt]]
    else:
        price, pv_max, ref = d.energy_price_next_day[nxt], d.pv_available_next_day[nxt], d.reference_load_next_day[nxt]
    H = range(len(price))
    p_imp, p_exp = price + d.import_tariff, price - d.export_tariff

    m = gp.Model("extended_horizon")
    m.Params.OutputFlag = 0
    load = m.addVars(H, lb=d.load_min_kWh, ub=d.load_max_kWh, name="load")
    imp = m.addVars(H, name="import")
    exp = m.addVars(H, name="export")
    pv = {t: m.addVar(ub=pv_max[t], name=f"pv[{t}]") for t in H if pv_max[t] > 0}
    ch = m.addVars(H, ub=d.battery_max_charge_kW, name="charge")
    dis = m.addVars(H, ub=d.battery_max_discharge_kW, name="discharge")

    soc, e = [], d.battery_initial_soc_kWh if start_soc is None else start_soc
    for t in H:
        e = e + d.battery_charging_efficiency * ch[t] - dis[t] / d.battery_discharging_efficiency
        soc.append(e)
        m.addConstr(e >= 0)
        m.addConstr(e <= d.battery_capacity_kWh)
        m.addConstr(load[t] + ch[t] - dis[t] == pv.get(t, 0) + imp[t] - exp[t])
    if today:
        m.addConstr(gp.quicksum(load[t] for t in range(24)) >= d.min_daily_energy_kWh)
    m.setObjective(gp.quicksum(-d.quadratic_disutility * (load[t] - ref[t]) * (load[t] - ref[t])
                               - p_imp[t] * imp[t] + p_exp[t] * exp[t] - d.pv_marginal_cost * pv.get(t, 0)
                               for t in H), GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Extended horizon ended with status {_status_name(m.Status)}")
    return dict(objective=m.ObjVal, n_variables=m.NumVars,
                charge=np.array([ch[t].X for t in H]), discharge=np.array([dis[t].X for t in H]),
                soc=np.array([s.getValue() for s in soc]))


# ---------------------------------------------------------------------- Question 2
class DisutilityModel(FlexibleConsumerModel):
    """Question 2: the Question 1 problem with the utility replaced by a disutility of the deviation
    from the reference load.

    ``build()`` creates all variables and constraints of Question 1 (with a zero utility, since the
    Question 2 data has none) and then replaces the objective by ``-disutility - procurement cost``.
    Subclasses only define ``_disutility()``: a gurobi expression (DKK), possibly with extra variables.
    """

    def build(self) -> "DisutilityModel":
        d0 = self.data
        if d0.reference_load is None:
            raise ValueError(f"{d0.question}: a disutility model needs a reference load profile")
        self.data = replace(d0, consumption_utility=0.0)
        super().build()
        self.data = d0
        d, m, T = self.data, self.m, self.T
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        procurement = gp.quicksum(p_imp[t] * self.var["import"][t] - p_exp[t] * self.var["export"][t]
                                  + d.pv_marginal_cost * self.var["pv"][t] for t in T)
        m.setObjective(-self._disutility() - procurement, GRB.MAXIMIZE)
        m.update()
        return self

    def _disutility(self) -> gp.LinExpr | gp.QuadExpr:
        raise NotImplementedError

    def _extract_results(self, status: str) -> Results:
        res = super()._extract_results(status)
        res.hourly["deviation"] = res.hourly["load"] - self.data.reference_load
        res.meta["utility"] = None      # no utility term: objective = -disutility - procurement cost
        res.meta["disutility"] = -res.objective - res.meta["procurement_cost"]
        return res


class LinearDisutilityModel(DisutilityModel):
    """2.(b): D_t = c^L |l_t - l^ref_t|, written as an LP with d+_t, d-_t >= 0 and l_t - l^ref_t = d+_t - d-_t."""

    def _disutility(self):
        d, m, T = self.data, self.m, self.T
        if d.linear_disutility is None:
            raise ValueError(f"{d.question}: needs a linear disutility coefficient")
        self.var["dev_up"] = m.addVars(T, lb=0.0, name="dev_up")
        self.var["dev_down"] = m.addVars(T, lb=0.0, name="dev_down")
        self.con["deviation"] = m.addConstrs(
            (self.var["load"][t] - d.reference_load[t] - self.var["dev_up"][t] + self.var["dev_down"][t] == 0
             for t in T), name="deviation")
        return gp.quicksum(d.linear_disutility * (self.var["dev_up"][t] + self.var["dev_down"][t]) for t in T)


class QuadraticDisutilityModel(DisutilityModel):
    """2.(c): D_t = c^Q (l_t - l^ref_t)^2, a convex quadratic term in the objective (no extra variables)."""

    def _disutility(self):
        d, T = self.data, self.T
        if d.quadratic_disutility is None:
            raise ValueError(f"{d.question}: needs a quadratic disutility coefficient")
        return gp.quicksum(d.quadratic_disutility * (self.var["load"][t] - d.reference_load[t]) ** 2 for t in T)


class AccumulatedDeviationModel(DisutilityModel):
    """2.(e) bonus: non-separable disutility D = sum_t c^Q dev_t^2 + c^S e_t^2 with e_t = rho e_{t-1} + dev_t.

    The defaults are the base-case parameters of the report; with rho = 0 the model is 2.(c) with c^Q + c^S.
    """

    def __init__(self, data: InputData, cQ: float = 0.05, cS: float = 0.06, rho: float = 0.8, **kwargs):
        super().__init__(data, **kwargs)
        self.cQ, self.cS, self.rho = cQ, cS, rho

    def _disutility(self):
        d, m, T = self.data, self.m, self.T
        e = self.var["acc"] = m.addVars(T, lb=-GRB.INFINITY, name="acc")      # accumulated deviation (kWh)
        self.con["accumulate"] = m.addConstrs(
            (e[t] == (self.rho * e[t - 1] if t > 0 else 0) + self.var["load"][t] - d.reference_load[t] for t in T),
            name="accumulate")
        return gp.quicksum(self.cQ * (self.var["load"][t] - d.reference_load[t]) ** 2 + self.cS * e[t] ** 2
                           for t in T)

    def _extract_results(self, status: str) -> Results:
        res = super()._extract_results(status)
        res.hourly["accumulated"] = res.hourly.pop("acc")
        return res
