"""Derive sensitivity scenarios from a base ``InputData`` without touching the JSON files.

Each helper returns a *new* ``InputData`` (the base data is never modified), so scenarios can
be chained::

    from src.data_loader import load_question
    from src.scenarios import scale_prices, set_tariffs, scale_pv

    base = load_question("Q1_caseA")
    high_spread = scale_prices(base, factor=2.0, keep_mean=True)      # same mean, doubled spread
    no_export_tariff = set_tariffs(high_spread, export_tariff=0.0)

Add your own helpers here (e.g. shifting the price peak towards the PV peak, changing the
minimum daily energy or the reference profile) and document them in the README.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data_loader import InputData


def scale_prices(data: InputData, factor: float, keep_mean: bool = False) -> InputData:
    """Multiply hourly prices by ``factor``. With ``keep_mean=True`` only the spread around the
    mean is scaled, which isolates the effect of price variability from the effect of price level.
    The next-day prices, when given, are scaled in the same way (they set pi^end in Question 3.(g))."""
    def scale(p):
        new_p = p.mean() + factor * (p - p.mean()) if keep_mean else factor * p
        return np.clip(new_p, 0.0, None)                             # prices stay non-negative
    next_day = data.energy_price_next_day
    return replace(data, energy_price=scale(data.energy_price),
                   energy_price_next_day=None if next_day is None else scale(next_day))


def shift_prices(data: InputData, delta: float) -> InputData:
    """Add ``delta`` DKK/kWh to every hourly price (changes the mean, keeps the spread). The next-day
    prices, when given, are shifted in the same way (they set pi^end in Question 3.(g))."""
    next_day = data.energy_price_next_day
    return replace(data, energy_price=np.clip(data.energy_price + delta, 0.0, None),
                   energy_price_next_day=None if next_day is None else np.clip(next_day + delta, 0.0, None))


def roll_prices(data: InputData, hours: int) -> InputData:
    """Shift the price profile in time by ``hours`` (positive = later). Useful to study the
    temporal correlation between the price peak and the PV peak."""
    return replace(data, energy_price=np.roll(data.energy_price, hours))


def set_tariffs(data: InputData, import_tariff: float | None = None, export_tariff: float | None = None) -> InputData:
    """Override the import and/or export grid tariff (DKK/kWh)."""
    return replace(
        data,
        import_tariff=data.import_tariff if import_tariff is None else import_tariff,
        export_tariff=data.export_tariff if export_tariff is None else export_tariff,
    )


def scale_pv(data: InputData, factor: float) -> InputData:
    """Scale the available PV production (e.g. 0.5 for a cloudy day, 0.0 for no PV)."""
    profile = np.clip(data.pv_profile * factor, 0.0, 1.0)
    return replace(data, pv_profile=profile, pv_available=data.pv_max_kW * profile)


def set_load_preferences(
    data: InputData,
    load_max_kWh: float | None = None,
    load_min_kWh: float | None = None,
    min_daily_energy_kWh: float | None = None,
) -> InputData:
    """Override the hourly load bounds and/or the minimum daily energy requirement."""
    return replace(
        data,
        load_max_kWh=data.load_max_kWh if load_max_kWh is None else load_max_kWh,
        load_min_kWh=data.load_min_kWh if load_min_kWh is None else load_min_kWh,
        min_daily_energy_kWh=data.min_daily_energy_kWh if min_daily_energy_kWh is None else min_daily_energy_kWh,
    )


def set_disutility(data: InputData, linear: float | None = None, quadratic: float | None = None) -> InputData:
    """Override the linear (DKK/kWh) and/or quadratic (DKK/kWh^2) disutility coefficient of Question 2."""
    return replace(
        data,
        linear_disutility=data.linear_disutility if linear is None else float(linear),
        quadratic_disutility=data.quadratic_disutility if quadratic is None else float(quadratic),
    )


def set_battery(data: InputData, capacity_kWh: float | None = None, power_kW: float | None = None,
                efficiency: float | None = None) -> InputData:
    """Override the battery of Question 3.(g): energy capacity (the initial state of charge keeps its
    share of the capacity), charging and discharging power (both), and charging and discharging
    efficiency (both, so the round trip is efficiency^2)."""
    cap = data.battery_capacity_kWh
    new_cap = cap if capacity_kWh is None else float(capacity_kWh)
    return replace(
        data,
        battery_capacity_kWh=new_cap,
        battery_initial_soc_kWh=data.battery_initial_soc_kWh * new_cap / cap,
        battery_max_charge_kW=data.battery_max_charge_kW if power_kW is None else float(power_kW),
        battery_max_discharge_kW=data.battery_max_discharge_kW if power_kW is None else float(power_kW),
        battery_charging_efficiency=data.battery_charging_efficiency if efficiency is None else float(efficiency),
        battery_discharging_efficiency=data.battery_discharging_efficiency if efficiency is None else float(efficiency),
    )
