# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""
PV Battery Self-Consumption Analysis
Simulates household PV-battery system for:
- One typical day (15-min resolution)
- One typical month (31 days)
Compares reference meter vs. model with ±50W error.
"""

import numpy as np
import matplotlib.pyplot as plt
from demandlib import bdew

model_error_low = 0
model_error_high = 0
target_annual_load_kWh = 3500
season = "summer"

# Load H0 profile from BDew
def load_h0_profile(target_annual_load_kWh=3500):
    e_slp = bdew.ElecSlp(year=2025)
    scaled_profiles = e_slp.get_scaled_profiles({"h0": target_annual_load_kWh})
    h0_profile_kWh = scaled_profiles["h0"]
    return h0_profile_kWh

h0_profile_kWh_scaled = load_h0_profile(target_annual_load_kWh=target_annual_load_kWh)
h0_profile_year = load_h0_profile(target_annual_load_kWh)
intervals_per_day = 96  # 15-min intervals
num_days = len(h0_profile_year) // intervals_per_day
h0_profile_year_reshaped = np.array(h0_profile_year[:num_days*intervals_per_day]).reshape((num_days, intervals_per_day))
h0_profile_kWh_scaled = np.mean(h0_profile_year_reshaped, axis=0)
# PV profile for 1.8 kWp system (typical summer day bell curve, normalized)
def typical_pv_day_profile(kWp=1.8, day_type=season):

    x = np.linspace(0, 24, intervals_per_day)
    if day_type == "summer":
        pv_gen = np.exp(-0.5 * ((x - 13) / 3.5) ** 2)
    else:
        pv_gen = np.exp(-0.5 * ((x - 12) / 4.5) ** 2)
    pv_gen = pv_gen / pv_gen.sum()
    daily_yield = 6 * kWp if day_type == "summer" else 2.5 * kWp
    pv_profile = pv_gen * daily_yield
    return pv_profile


# Battery simulation function
def simulate_day(
    h0, pv, battery_capacity=2.5, error_low=0, error_high=0
):
    soc = 0.0
    soc_hist = []
    direct_consumption = []
    battery_discharge = []
    grid_import = []
    grid_export = []
    for i, (true_load, pv_gen) in enumerate(zip(h0, pv)):
        est_load = true_load

        # Apply error if error_low or error_high are nonzero
        if error_low != 0 or error_high != 0:
            error = np.random.uniform(error_low, error_high)
            est_load = max(0, true_load + error * 0.25)  # error in kWh

        direct = min(est_load, pv_gen)
        surplus = max(0, pv_gen - est_load)
        deficit = max(0, est_load - pv_gen)

        charge = min(surplus, battery_capacity - soc)
        soc += charge
        soc = min(max(soc, 0.0), battery_capacity)
        export = surplus - charge

        discharge = min(deficit, soc)
        soc -= discharge
        soc = min(max(soc, 0.0), battery_capacity)
        delivered = direct + discharge
        actual_delivered = min(delivered, true_load)

        if delivered > 0:
            actual_direct = direct * actual_delivered / delivered
            actual_discharge = discharge * actual_delivered / delivered
        else:
            actual_direct = 0.0
            actual_discharge = 0.0

        actual_grid_import = max(0, true_load - actual_delivered)

        soc_hist.append(soc)
        direct_consumption.append(actual_direct)
        battery_discharge.append(actual_discharge)
        grid_import.append(actual_grid_import)
        grid_export.append(export)

    return {
        "soc_hist": soc_hist,
        "direct_consumption": direct_consumption,
        "battery_discharge": battery_discharge,
        "grid_import": grid_import,
        "grid_export": grid_export,
    }


pv_day = typical_pv_day_profile(kWp=1.8, day_type=season)
ref_result = simulate_day(
    h0_profile_kWh_scaled,
    pv_day,
    battery_capacity=2.5,
    error_low=0,
    error_high=0,
)
model_result = simulate_day(
    h0_profile_kWh_scaled,
    pv_day,
    battery_capacity=2.5,
    error_low=model_error_low,
    error_high=model_error_high,
)


def summarize_result(result, pv, h0):
    total_pv = np.sum(pv)
    total_load = np.sum(h0)
    direct = np.sum(result["direct_consumption"])
    battery = np.sum(result["battery_discharge"])
    self_consumed = direct + battery
    export = np.sum(result["grid_export"])
    grid_import = np.sum(result["grid_import"])
    scr = self_consumed / total_pv if total_pv > 0 else 0
    ssr = self_consumed / total_load if total_load > 0 else 0
    return {
        "total_pv": total_pv,
        "total_load": total_load,
        "direct": direct,
        "battery": battery,
        "self_consumed": self_consumed,
        "export": export,
        "grid_import": grid_import,
        "scr": scr,
        "ssr": ssr,
    }


ref_summary = summarize_result(ref_result, pv_day, h0_profile_kWh_scaled)
model_summary = summarize_result(model_result, pv_day, h0_profile_kWh_scaled)


error_info = f"Custom error: [{model_error_low}, {model_error_high}] kWh"

print(f"=== Single Summer Day ===\n({error_info})")
print("Reference Meter:")
for k, v in ref_summary.items():
    print(f"{k}: {v:.2f}")
print(f"\nModel ({error_info}):")
for k, v in model_summary.items():
    print(f"{k}: {v:.2f}")


plt.figure(figsize=(15, 8))
time_axis = np.linspace(0, 24, len(h0_profile_kWh_scaled))
plt.subplot(3, 1, 1)
plt.plot(time_axis, pv_day, "y-", label="PV Generation")
plt.plot(time_axis, h0_profile_kWh_scaled, "b-", label="Household Load")
plt.ylabel("kWh (per 15min)")
plt.legend()
plt.grid(True)
plt.title(f"Power Flow\n({error_info})")
plt.subplot(3, 1, 2)
plt.plot(time_axis, ref_result["soc_hist"], "g-", label="SOC (Ref)")
plt.plot(time_axis, model_result["soc_hist"], "r--", label="SOC (Model)")
plt.ylabel("Battery SOC (kWh)")
plt.legend()
plt.grid(True)
plt.title(f"Battery State of Charge\n({error_info})")
plt.subplot(3, 1, 3)
plt.plot(time_axis, ref_result["grid_export"], "r-", label="Grid Export (Ref)")
plt.plot(time_axis, model_result["grid_export"], "r--", label="Grid Export (Model)")
plt.plot(
    time_axis, [-x for x in ref_result["grid_import"]], "c-", label="Grid Import (Ref)"
)
plt.plot(
    time_axis,
    [-x for x in model_result["grid_import"]],
    "c--",
    label="Grid Import (Model)",
)
plt.ylabel("kWh (per 15min)")
plt.legend()
plt.grid(True)
plt.title(f"Grid Exchange\n({error_info})")
plt.xlabel("Hour of Day")
plt.tight_layout()
plt.savefig("single_day_powerflow.png")
# # plt.show()  # Disabled for Streamlit compatibility

# ---


h0_year = np.tile(h0_profile_kWh_scaled, 365)


def seasonal_pv_scale(day):

    return 0.5 + 0.5 * np.sin(2 * np.pi * (day - 80) / 365)


pv_year = []
for day in range(365):
    pv_scale = seasonal_pv_scale(day)
    pv_day = typical_pv_day_profile(kWp=1.8, day_type=season) * pv_scale
    pv_year.append(pv_day)
pv_year = np.concatenate(pv_year)

ref_year_result = simulate_day(
    h0_year,
    pv_year,
    battery_capacity=2.5,
    error_low=0,
    error_high=0,
)
model_year_result = simulate_day(
    h0_year,
    pv_year,
    battery_capacity=2.5,
    error_low=model_error_low,
    error_high=model_error_high,
)
ref_year_summary = summarize_result(ref_year_result, pv_year, h0_year)
model_year_summary = summarize_result(model_year_result, pv_year, h0_year)

print(f"\n=== 1 Year (365 days) ===\n({error_info})")
print("Reference Meter:")
for k, v in ref_year_summary.items():
    print(f"{k}: {v:.2f}")
print(f"\nModel ({error_info}):")
for k, v in model_year_summary.items():
    print(f"{k}: {v:.2f}")


days = np.arange(1, 366)
plt.figure(figsize=(15, 6))
plt.plot(
    days,
    np.add.reduceat(
        np.array(ref_year_result["direct_consumption"])
        + np.array(ref_year_result["battery_discharge"]),
        np.arange(0, intervals_per_day * 365, intervals_per_day),
    ),
    label="Self-consumed PV (Ref)",
)
plt.plot(
    days,
    np.add.reduceat(
        np.array(model_year_result["direct_consumption"])
        + np.array(model_year_result["battery_discharge"]),
        np.arange(0, intervals_per_day * 365, intervals_per_day),
    ),
    label="Self-consumed PV (Model)",
)
plt.title("Self-consumed PV (1 Year)")
plt.xlabel("Day of Year")
plt.ylabel("kWh")
plt.legend()
plt.tight_layout()
# plt.show()  # Disabled for Streamlit compatibility

# --- Grid Exchange Plot (Import/Export) for 1 Year ---
plt.figure(figsize=(15, 6))
# Grid Import
plt.plot(
    days,
    np.add.reduceat(
        np.array(ref_year_result["grid_import"]),
        np.arange(0, intervals_per_day * 365, intervals_per_day),
    ),
    label="Grid Import (Ref)",
    color="blue",
    linestyle="-"
)
plt.plot(
    days,
    np.add.reduceat(
        np.array(model_year_result["grid_import"]),
        np.arange(0, intervals_per_day * 365, intervals_per_day),
    ),
    label="Grid Import (Model)",
    color="blue",
    linestyle="--"
)
# Grid Export
plt.plot(
    days,
    np.add.reduceat(
        np.array(ref_year_result["grid_export"]),
        np.arange(0, intervals_per_day * 365, intervals_per_day),
    ),
    label="Grid Export (Ref)",
    color="red",
    linestyle="-"
)
plt.plot(
    days,
    np.add.reduceat(
        np.array(model_year_result["grid_export"]),
        np.arange(0, intervals_per_day * 365, intervals_per_day),
    ),
    label="Grid Export (Model)",
    color="red",
    linestyle="--"
)
plt.title("Grid Exchange (Import/Export) - 1 Year")
plt.xlabel("Day of Year")
plt.ylabel("kWh")
plt.legend()
plt.tight_layout()
# plt.show()  # Disabled for Streamlit compatibility

plt.plot(
    days,
    np.add.reduceat(pv_year, np.arange(0, intervals_per_day * 365, intervals_per_day)),
    label="PV Generation",
)
plt.xlabel("Day")
plt.ylabel("kWh/day")
plt.legend()
plt.title(f"Daily Energy Flows (1 Year)\n({error_info})")
plt.grid(True)
plt.tight_layout()
plt.savefig("year_energyflows.png")
# # plt.show()  # Disabled for Streamlit compatibility

# ---

h0_month = np.tile(h0_profile_kWh_scaled, 31)
pv_month = np.tile(typical_pv_day_profile(kWp=1.8, day_type=season), 31)
ref_month_result = simulate_day(
    h0_month,
    pv_month,
    battery_capacity=2.5,
    error_low=0,
    error_high=0,
)
model_month_result = simulate_day(
    h0_month,
    pv_month,
    battery_capacity=2.5,
    error_low=model_error_low,
    error_high=model_error_high,
)
ref_month_summary = summarize_result(ref_month_result, pv_month, h0_month)
model_month_summary = summarize_result(model_month_result, pv_month, h0_month)

print(f"\n=== 1 Month (31 days) ===\n({error_info})")
print("Reference Meter:")
for k, v in ref_month_summary.items():
    print(f"{k}: {v:.2f}")
print(f"\nModel ({error_info}):")
for k, v in model_month_summary.items():
    print(f"{k}: {v:.2f}")


days = np.arange(1, 32)
plt.figure(figsize=(15, 6))
plt.plot(
    days,
    np.add.reduceat(
        (
            ref_month_result["self_consumed"]
            if "self_consumed" in ref_month_result
            else np.array(ref_month_result["direct_consumption"])
            + np.array(ref_month_result["battery_discharge"])
        ),
        np.arange(0, intervals_per_day * 31, intervals_per_day),
    ),
    label="Self-consumed PV (Ref)",
)
plt.plot(
    days,
    np.add.reduceat(
        model_month_result["direct_consumption"]
        + np.array(model_month_result["battery_discharge"]),
        np.arange(0, intervals_per_day * 31, intervals_per_day),
    ),
    label="Self-consumed PV (Model)",
)
plt.plot(
    days,
    np.add.reduceat(pv_month, np.arange(0, intervals_per_day * 31, intervals_per_day)),
    label="PV Generation",
)
plt.xlabel("Day")
plt.ylabel("kWh/day")
plt.legend()
plt.title(f"Daily Energy Flows (1 Month)\n({error_info})")
plt.grid(True)
plt.tight_layout()
plt.savefig("month_energyflows.png")
# # plt.show()  # Disabled for Streamlit compatibility
