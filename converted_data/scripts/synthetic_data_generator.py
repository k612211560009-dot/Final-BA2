import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RNG = np.random.default_rng(42)


def bearing_failure_logic(vibration_rms, operating_hours):
    if vibration_rms > 7.0 and operating_hours > 15000:
        return "Critical"
    elif vibration_rms > 4.5 and operating_hours > 8000:
        return "Warning"
    else:
        return "Normal"


def corrosion_impact(corrosion_rate, pressure):
    safety_factor = 1.0 - (corrosion_rate * 2.0)
    if safety_factor < 0.6:
        return "Critical"
    elif safety_factor < 0.8:
        return "Warning"
    else:
        return "Normal"


def seal_degradation(temperature, operating_hours):
    if temperature > 100 and operating_hours > 10000:
        return "High"
    elif temperature > 90 and operating_hours > 5000:
        return "Medium"
    else:
        return "Low"


def make_equipment_registry():
    registry = []
    # 4 pumps, 3 turbines, 3 compressors = 10
    for i in range(4):
        registry.append((f"PUMP_{i+1:03}", "Centrifugal Pump", 15))
    for i in range(3):
        registry.append((f"TURBINE_{i+1:03}", "Steam Turbine", 25))
    for i in range(3):
        registry.append((f"COMP_{i+1:03}", "Screw Compressor", 20))
    return registry


def generate_time_index(start, end, freq="15T"):
    return pd.date_range(start=start, end=end, freq=freq)


def simulate_device_time_series(eid, etype, design_life_years, timestamps):
    n = len(timestamps)
    # Base operating hours cumulative
    # Each 15-min = 0.25 hours
    hours_step = 0.25
    operating_hours = np.arange(n) * hours_step

    # Seasonal temperature: annual sinusoid + noise, hotter in summer
    days = np.array([(ts - timestamps[0]).days for ts in timestamps])
    temp_season = 10 * np.sin(2 * np.pi * days / 365.25) + 80  # base 80C with +/-10C
    temp_noise = RNG.normal(0, 1.5, size=n)
    temperature_c = temp_season + temp_noise

    # motor speed (rpm) depends on equipment
    if "PUMP" in eid:
        motor_speed_rpm = RNG.normal(1450, 100, size=n)  # typical pump
        flow_rate = RNG.uniform(100, 400, size=n)
        discharge_pressure_bar = RNG.uniform(5, 15, size=n)
        suction_pressure_bar = RNG.uniform(1, 3, size=n)
        motor_power_kw = RNG.uniform(75, 200, size=n)
    elif "TURBINE" in eid:
        motor_speed_rpm = RNG.normal(3000, 200, size=n)
        flow_rate = RNG.uniform(200, 600, size=n)
        discharge_pressure_bar = RNG.uniform(10, 25, size=n)
        suction_pressure_bar = RNG.uniform(2, 5, size=n)
        motor_power_kw = RNG.uniform(150, 300, size=n)
    else:  # compressor
        motor_speed_rpm = RNG.normal(1800, 150, size=n)
        flow_rate = RNG.uniform(50, 300, size=n)
        discharge_pressure_bar = RNG.uniform(6, 20, size=n)
        suction_pressure_bar = RNG.uniform(1, 4, size=n)
        motor_power_kw = RNG.uniform(90, 250, size=n)

    # Vibration baseline and progressive degradation trend
    baseline_rms = RNG.normal(1.5, 0.3, size=n)
    long_term_trend = (operating_hours / (24*365)) * RNG.uniform(0.2, 2.0) / 100.0
    vibration_rms = baseline_rms + long_term_trend
    # occasional step increases
    steps = RNG.integers(0, 10)
    for _ in range(steps):
        pos = RNG.integers(0, n)
        mag = RNG.uniform(0.5, 5.0)
        vibration_rms[pos:pos+RNG.integers(1, 2000)] += mag

    # Peak roughly correlated with rms
    vibration_peak = vibration_rms * RNG.uniform(1.5, 3.0, size=n)

    # Frequency peaks (1x, 2x, bearing_freq) as numeric amplitude features
    freq1 = np.abs(RNG.normal(0.5, 0.2, size=n)) * (motor_speed_rpm / 1000)
    freq2 = np.abs(RNG.normal(0.3, 0.15, size=n)) * (motor_speed_rpm / 2000)
    bearing_freq = np.abs(RNG.normal(0.2, 0.1, size=n)) * (motor_speed_rpm / 3000)

    # Corrosion: only relevant for pipeline-like devices; set NaNs here but keep columns
    pipe_thickness_mm = np.full(n, np.nan)
    thickness_loss_mm = np.full(n, np.nan)
    corrosion_rate_mm_year = np.full(n, np.nan)
    remaining_life_years = np.full(n, np.nan)

    # Operating power and efficiency noise
    motor_power_kw = motor_power_kw * (1 + RNG.normal(0, 0.02, size=n))

    # Create DataFrame
    df = pd.DataFrame({
        "equipment_id": eid,
        "equipment_type": etype,
        "timestamp": timestamps,
        "operating_hours": operating_hours,
        "motor_speed_rpm": motor_speed_rpm,
        "flow_rate_m3h": flow_rate,
        "discharge_pressure_bar": discharge_pressure_bar,
        "suction_pressure_bar": suction_pressure_bar,
        "motor_power_kw": motor_power_kw,
        "temperature_c": temperature_c,
        "vibration_rms_mms": vibration_rms,
        "vibration_peak_mms": vibration_peak,
        "freq1_amp": freq1,
        "freq2_amp": freq2,
        "bearing_freq_amp": bearing_freq,
        "pipe_thickness_mm": pipe_thickness_mm,
        "thickness_loss_mm": thickness_loss_mm,
        "corrosion_rate_mm_year": corrosion_rate_mm_year,
        "remaining_life_years": remaining_life_years,
    })

    # Maintenance schedule every 3-6 months
    maintenance_interval_days = RNG.integers(90, 180)
    next_maint = timestamps[0] + pd.Timedelta(days=int(maintenance_interval_days))
    maint_dates = []
    md = next_maint
    while md <= timestamps[-1]:
        maint_dates.append(md)
        md = md + pd.Timedelta(days=int(RNG.integers(90, 180)))

    df["last_maintenance_date"] = pd.NaT
    df["maintenance_type"] = "None"
    df["downtime_hours"] = 0.0
    df["replaced_components"] = "None"

    for d in maint_dates:
        # find nearest index by timestamp
        idx = (df["timestamp"] - d).abs().idxmin()
        df.at[idx, "last_maintenance_date"] = d
        df.at[idx, "maintenance_type"] = RNG.choice(["Routine", "Corrective", "Overhaul"], p=[0.7, 0.25, 0.05])
        if df.at[idx, "maintenance_type"] == "Overhaul":
            df.at[idx, "downtime_hours"] = RNG.uniform(24, 96)
            df.at[idx, "replaced_components"] = RNG.choice(["Bearing", "Seal", "Impeller", "None"], p=[0.4, 0.2, 0.2, 0.2])
        elif df.at[idx, "maintenance_type"] == "Corrective":
            df.at[idx, "downtime_hours"] = RNG.uniform(4, 24)
            df.at[idx, "replaced_components"] = RNG.choice(["Bearing", "Seal", "None"], p=[0.5, 0.2, 0.3])
        else:
            df.at[idx, "downtime_hours"] = RNG.uniform(0.5, 4)
            df.at[idx, "replaced_components"] = "None"

    # Inject failure events: 2-3 per device per year
    failures_per_year = RNG.integers(2, 4)
    total_failures = failures_per_year * 2  # 2 years
    failure_times = RNG.choice(timestamps, size=total_failures, replace=False)
    failure_times = np.sort(failure_times)

    df["equipment_condition"] = "Normal"
    df["days_to_failure"] = np.inf
    df["maintenance_urgency"] = "Low"

    for ft in failure_times:
        # degrade trend leading up to failure
        idx = (df["timestamp"] - ft).abs().idxmin()
        # choose a window of progressive degradation (7-30 days)
        degrade_days = int(RNG.integers(7, 30))
        start_idx = max(0, idx - int(degrade_days * 24 * 4))
        # increase vibration gradually
        ramp = np.linspace(0, RNG.uniform(3.0, 8.0), idx - start_idx)
        df.loc[start_idx:idx-1, "vibration_rms_mms"] += ramp
        # set the failure row
        df.at[idx, "equipment_condition"] = "Failed"
        # days_to_failure for timestamps prior to failure
        df.loc[:idx, "days_to_failure"] = ((ft - df.loc[:idx, "timestamp"]).dt.total_seconds() / 86400).clip(lower=0)
        df.loc[:idx, "maintenance_urgency"] = np.where(df.loc[:idx, "days_to_failure"] < 7, "Critical", df.loc[:idx, "maintenance_urgency"])

    # fill infinite days_to_failure with large number
    df["days_to_failure"].replace(np.inf, 9999, inplace=True)

    # Derive bearing_condition using logic
    df["bearing_condition"] = df.apply(lambda r: bearing_failure_logic(r["vibration_rms_mms"], r["operating_hours"]), axis=1)
    df["seal_degradation"] = df.apply(lambda r: seal_degradation(r["temperature_c"], r["operating_hours"]), axis=1)

    # maintenance_urgency enhancements
    df.loc[df["bearing_condition"] == "Critical", "maintenance_urgency"] = "High"
    df.loc[df["equipment_condition"] == "Failed", "maintenance_urgency"] = "Critical"

    # Inject random missingness (~7%) across primary telemetry numeric columns
    telemetry_cols = [
        "motor_speed_rpm",
        "flow_rate_m3h",
        "discharge_pressure_bar",
        "suction_pressure_bar",
        "motor_power_kw",
        "temperature_c",
        "vibration_rms_mms",
        "vibration_peak_mms",
        "freq1_amp",
        "freq2_amp",
        "bearing_freq_amp",
    ]
    mask_prob = 0.07
    for col in telemetry_cols:
        if col in df.columns:
            mask = RNG.random(n) < mask_prob
            df.loc[mask, col] = np.nan

    # Inject outliers into vibration_rms (~3%)
    outlier_mask = RNG.random(n) < 0.03
    df.loc[outlier_mask, "vibration_rms_mms"] = df.loc[outlier_mask, "vibration_rms_mms"] * RNG.uniform(3.0, 12.0, size=outlier_mask.sum())

    return df


def main():
    start = "2022-01-01"
    end = "2023-12-31 23:45:00"
    timestamps = generate_time_index(start, end, freq="15T")

    registry = make_equipment_registry()

    parts = []
    for eid, etype, life in registry:
        print(f"Simulating {eid} ({etype})")
        df = simulate_device_time_series(eid, etype, life, timestamps)
        parts.append(df)

    full = pd.concat(parts, ignore_index=True)

    # Reorder columns and add more metadata
    full["installation_date"] = pd.to_datetime("2020-01-15")
    full["design_lifetime_years"] = full["equipment_type"].map({
        "Centrifugal Pump": 15,
        "Steam Turbine": 25,
        "Screw Compressor": 20,
    })
    full["criticality_level"] = full["equipment_type"].map({
        "Centrifugal Pump": "Medium",
        "Steam Turbine": "High",
        "Screw Compressor": "High",
    })

    # Ensure composite key
    full.sort_values(["equipment_id", "timestamp"], inplace=True)

    out_file = "equipment_predictive_maintenance_dataset.csv"
    print(f"Writing {out_file} with {len(full):,} rows and {len(full.columns)} columns")
    full.to_csv(out_file, index=False)

    # Basic verification prints
    df = full
    print("--- Verification ---")
    # Focus missingness check on telemetry columns (where we injected missingness)
    telemetry_cols = [
        "motor_speed_rpm",
        "flow_rate_m3h",
        "discharge_pressure_bar",
        "suction_pressure_bar",
        "motor_power_kw",
        "temperature_c",
        "vibration_rms_mms",
        "vibration_peak_mms",
        "freq1_amp",
        "freq2_amp",
        "bearing_freq_amp",
    ]
    available_cols = [c for c in telemetry_cols if c in df.columns]
    total_telemetry_cells = df[available_cols].size
    missing_telemetry = df[available_cols].isna().sum().sum()
    print(f"Telemetry missingness: {missing_telemetry} cells ({missing_telemetry/total_telemetry_cells:.2%})")
    # outliers count approx: vibration_rms > 8 as proxy
    vib_outliers = (df["vibration_rms_mms"] > 8).sum()
    print(f"Vibration outlier count (rms>8): {vib_outliers}")
    # failures per device
    fails = df[df["equipment_condition"] == "Failed"].groupby("equipment_id").size()
    print("Failures per device (sample):")
    print(fails.head())


if __name__ == "__main__":
    main()
