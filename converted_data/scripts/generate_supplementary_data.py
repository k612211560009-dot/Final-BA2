"""
Generate supplementary metadata CSVs to enrich existing feature datasets for PdM analysis.

Outputs (in data/metadata/):
- equipment_master.csv
- sensor_metadata.csv
- maintenance_schedule.csv
- failure_history.csv
- weather_data.csv
- operational_context.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pathlib

RNG = np.random.default_rng(42)

def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

# 1. EQUIPMENT MASTER
def generate_equipment_master():
    equipment = []
    # Bearings (from CWRU/CWRU2 data — map file prefixes to equipment_id)
    bearing_types = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR0076', 'OR0146', 'OR0216', 'Normal']
    for i, bt in enumerate(bearing_types):
        equipment.append({
            'equipment_id': f'BEARING_{i+1:03}',
            'equipment_type': 'Bearing',
            'equipment_subtype': bt,
            'installation_date': '2020-01-15',
            'design_lifetime_years': 10,
            'criticality_level': 'High' if 'IR' in bt or 'OR' in bt else 'Medium',
            'location': f'Plant_A_Unit_{(i%3)+1}',
            'manufacturer': 'SKF',
            'model': 'SKF_6205',
            'serial_number': f'SN-BEAR-{1000+i}'
        })
    
    # Pumps (from pumps data — Machine_ID 1..N)
    for i in range(1, 5):
        equipment.append({
            'equipment_id': f'PUMP_{i:03}',
            'equipment_type': 'Centrifugal Pump',
            'equipment_subtype': 'Multistage',
            'installation_date': '2019-06-10',
            'design_lifetime_years': 15,
            'criticality_level': 'High',
            'location': f'Plant_B_Unit_{i}',
            'manufacturer': 'Grundfos',
            'model': 'CR-45',
            'serial_number': f'SN-PUMP-{2000+i}'
        })
    
    # Turbines (from CMAPS data — map to unit_id in CMAPS)
    for i in range(1, 4):
        equipment.append({
            'equipment_id': f'TURBINE_{i:03}',
            'equipment_type': 'Gas Turbine',
            'equipment_subtype': 'Single Shaft',
            'installation_date': '2018-03-20',
            'design_lifetime_years': 25,
            'criticality_level': 'Critical',
            'location': f'Plant_C_Unit_{i}',
            'manufacturer': 'GE',
            'model': 'LM2500',
            'serial_number': f'SN-TURB-{3000+i}'
        })
    
    # Pipelines (from corrosion data — synthetic segment IDs)
    for i in range(1, 6):
        equipment.append({
            'equipment_id': f'PIPE_SEG_{i:03}',
            'equipment_type': 'Pipeline',
            'equipment_subtype': 'Carbon Steel',
            'installation_date': '2015-09-01',
            'design_lifetime_years': 30,
            'criticality_level': 'High' if i <= 2 else 'Medium',
            'location': f'Pipeline_Route_{i}',
            'manufacturer': 'Tenaris',
            'model': 'API-5L-X65',
            'serial_number': f'SN-PIPE-{4000+i}'
        })
    
    df = pd.DataFrame(equipment)
    return df


# 2. SENSOR METADATA
def generate_sensor_metadata(equipment_df):
    sensors = []
    sensor_id = 1
    for _, eq in equipment_df.iterrows():
        eid = eq['equipment_id']
        etype = eq['equipment_type']
        if etype == 'Bearing':
            # 2 vibration sensors (DE, NDE) + 1 temperature
            for loc in ['Drive End', 'Non-Drive End']:
                sensors.append({
                    'sensor_id': f'SENSOR_{sensor_id:04}',
                    'equipment_id': eid,
                    'sensor_type': 'Vibration',
                    'sensor_location': loc,
                    'sampling_rate_hz': 12000,
                    'units': 'mm/s',
                    'installation_date': '2020-02-01'
                })
                sensor_id += 1
            sensors.append({
                'sensor_id': f'SENSOR_{sensor_id:04}',
                'equipment_id': eid,
                'sensor_type': 'Temperature',
                'sensor_location': 'Housing',
                'sampling_rate_hz': 1,
                'units': 'Celsius',
                'installation_date': '2020-02-01'
            })
            sensor_id += 1
        elif etype == 'Centrifugal Pump':
            for stype in ['Vibration', 'Temperature', 'Pressure_Suction', 'Pressure_Discharge', 'Flow_Rate']:
                sensors.append({
                    'sensor_id': f'SENSOR_{sensor_id:04}',
                    'equipment_id': eid,
                    'sensor_type': stype,
                    'sensor_location': 'Pump_Body' if stype in ['Vibration', 'Temperature'] else 'Piping',
                    'sampling_rate_hz': 12000 if stype == 'Vibration' else 1,
                    'units': 'mm/s' if stype == 'Vibration' else ('Celsius' if stype == 'Temperature' else ('bar' if 'Pressure' in stype else 'm3/h')),
                    'installation_date': '2019-07-01'
                })
                sensor_id += 1
        elif etype == 'Gas Turbine':
            for stype in ['Vibration', 'Temperature_Exhaust', 'Temperature_Bearing', 'Pressure_Compressor', 'Speed_RPM']:
                sensors.append({
                    'sensor_id': f'SENSOR_{sensor_id:04}',
                    'equipment_id': eid,
                    'sensor_type': stype,
                    'sensor_location': 'Turbine_Casing' if stype == 'Vibration' else ('Exhaust' if 'Exhaust' in stype else 'Bearing'),
                    'sampling_rate_hz': 10000 if stype == 'Vibration' else 1,
                    'units': 'mm/s' if stype == 'Vibration' else ('Celsius' if 'Temperature' in stype else ('bar' if 'Pressure' in stype else 'rpm')),
                    'installation_date': '2018-04-01'
                })
                sensor_id += 1
        elif etype == 'Pipeline':
            for stype in ['Ultrasonic_Thickness', 'Temperature', 'Pressure']:
                sensors.append({
                    'sensor_id': f'SENSOR_{sensor_id:04}',
                    'equipment_id': eid,
                    'sensor_type': stype,
                    'sensor_location': 'Pipe_Wall',
                    'sampling_rate_hz': 0.01 if stype == 'Ultrasonic_Thickness' else 0.1,  # monthly thickness, hourly temp/pressure
                    'units': 'mm' if stype == 'Ultrasonic_Thickness' else ('Celsius' if stype == 'Temperature' else 'bar'),
                    'installation_date': '2016-01-01'
                })
                sensor_id += 1
    df = pd.DataFrame(sensors)
    return df


# 3. MAINTENANCE SCHEDULE
def generate_maintenance_schedule(equipment_df):
    records = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    for _, eq in equipment_df.iterrows():
        eid = eq['equipment_id']
        etype = eq['equipment_type']
        # Schedule maintenance every 3-6 months
        interval_days = RNG.integers(90, 180)
        current = start_date + timedelta(days=int(RNG.integers(0, 60)))
        while current <= end_date:
            records.append({
                'equipment_id': eid,
                'maintenance_date': current.strftime('%Y-%m-%d'),
                'maintenance_type': RNG.choice(['Routine', 'Preventive', 'Overhaul'], p=[0.7, 0.2, 0.1]),
                'planned_downtime_hours': RNG.uniform(2, 48) if RNG.random() > 0.7 else 0,
                'performed': 'Yes' if current < datetime.now() else 'Scheduled'
            })
            current += timedelta(days=int(RNG.integers(90, 180)))
    df = pd.DataFrame(records)
    return df


# 4. FAILURE HISTORY
def generate_failure_history(equipment_df):
    records = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    for _, eq in equipment_df.iterrows():
        eid = eq['equipment_id']
        etype = eq['equipment_type']
        # 1-3 failures over 4 years
        num_failures = RNG.integers(1, 4)
        for _ in range(num_failures):
            fail_date = start_date + timedelta(days=int(RNG.integers(0, (end_date - start_date).days)))
            if etype == 'Bearing':
                failure_mode = RNG.choice(['Spalling', 'Pitting', 'Cracking', 'Lubrication_Failure'])
                root_cause = RNG.choice(['Excessive_Load', 'Contamination', 'Misalignment', 'Inadequate_Lubrication'])
            elif etype == 'Centrifugal Pump':
                failure_mode = RNG.choice(['Seal_Failure', 'Impeller_Wear', 'Cavitation', 'Bearing_Failure'])
                root_cause = RNG.choice(['Dry_Running', 'Abrasive_Fluid', 'Low_NPSH', 'Vibration'])
            elif etype == 'Gas Turbine':
                failure_mode = RNG.choice(['Blade_Erosion', 'Bearing_Failure', 'Combustion_Issue', 'Sensor_Failure'])
                root_cause = RNG.choice(['Hot_Gas_Path_Damage', 'Lubrication_Issue', 'Fuel_Quality', 'Control_System'])
            elif etype == 'Pipeline':
                failure_mode = RNG.choice(['Corrosion_Perforation', 'Crack', 'Leak', 'Erosion'])
                root_cause = RNG.choice(['Internal_Corrosion', 'External_Corrosion', 'Mechanical_Stress', 'Fluid_Erosion'])
            else:
                failure_mode = 'Unknown'
                root_cause = 'Unknown'
            
            records.append({
                'equipment_id': eid,
                'failure_date': fail_date.strftime('%Y-%m-%d'),
                'failure_mode': failure_mode,
                'root_cause': root_cause,
                'downtime_hours': RNG.uniform(4, 120),
                'repair_cost_usd': RNG.uniform(5000, 200000),
                'severity': RNG.choice(['Minor', 'Moderate', 'Critical'], p=[0.3, 0.5, 0.2])
            })
    df = pd.DataFrame(records).sort_values('failure_date')
    return df


# 5. WEATHER DATA
def generate_weather_data():
    records = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    current = start_date
    while current <= end_date:
        # Seasonal temperature (higher in summer)
        day_of_year = current.timetuple().tm_yday
        base_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        temp = base_temp + RNG.normal(0, 3)
        humidity = RNG.uniform(40, 90)
        pressure = RNG.normal(1013, 5)
        records.append({
            'date': current.strftime('%Y-%m-%d'),
            'ambient_temp_c': round(temp, 1),
            'humidity_percent': round(humidity, 1),
            'atmospheric_pressure_hPa': round(pressure, 1),
            'rainfall_mm': round(RNG.exponential(2), 1) if RNG.random() > 0.7 else 0.0
        })
        current += timedelta(days=1)
    df = pd.DataFrame(records)
    return df


# 6. OPERATIONAL CONTEXT
def generate_operational_context(equipment_df):
    records = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    
    for _, eq in equipment_df.iterrows():
        eid = eq['equipment_id']
        etype = eq['equipment_type']
        # Sample 500 hourly records per equipment
        sample_times = RNG.choice(timestamps, size=min(500, len(timestamps)), replace=False)
        for ts in sample_times:
            ts = pd.Timestamp(ts)
            if etype == 'Bearing':
                speed_rpm = RNG.normal(1750, 100)
                load_percent = RNG.uniform(40, 100)
                mode = RNG.choice(['Running', 'Idle', 'Startup'], p=[0.85, 0.1, 0.05])
            elif etype == 'Centrifugal Pump':
                speed_rpm = RNG.normal(1450, 50)
                load_percent = RNG.uniform(50, 100)
                mode = RNG.choice(['Running', 'Idle', 'Startup'], p=[0.90, 0.05, 0.05])
            elif etype == 'Gas Turbine':
                speed_rpm = RNG.normal(3000, 150)
                load_percent = RNG.uniform(60, 100)
                mode = RNG.choice(['Full_Load', 'Part_Load', 'Idle'], p=[0.6, 0.3, 0.1])
            elif etype == 'Pipeline':
                speed_rpm = 0
                load_percent = RNG.uniform(30, 100)  # flow rate as % of capacity
                mode = 'Operating'
            else:
                speed_rpm = 0
                load_percent = 0
                mode = 'Unknown'
            
            records.append({
                'equipment_id': eid,
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'operating_speed_rpm': round(speed_rpm, 1),
                'load_percent': round(load_percent, 1),
                'operating_mode': mode
            })
    df = pd.DataFrame(records).sort_values(['equipment_id', 'timestamp'])
    return df


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata')
    ensure_dir(out_dir)
    
    print("Generating equipment_master.csv...")
    eq_master = generate_equipment_master()
    eq_master.to_csv(os.path.join(out_dir, 'equipment_master.csv'), index=False)
    print(f"  wrote {len(eq_master)} equipment records")
    
    print("Generating sensor_metadata.csv...")
    sensor_meta = generate_sensor_metadata(eq_master)
    sensor_meta.to_csv(os.path.join(out_dir, 'sensor_metadata.csv'), index=False)
    print(f"  wrote {len(sensor_meta)} sensor records")
    
    print("Generating maintenance_schedule.csv...")
    maint_sched = generate_maintenance_schedule(eq_master)
    maint_sched.to_csv(os.path.join(out_dir, 'maintenance_schedule.csv'), index=False)
    print(f"  wrote {len(maint_sched)} maintenance records")
    
    print("Generating failure_history.csv...")
    fail_hist = generate_failure_history(eq_master)
    fail_hist.to_csv(os.path.join(out_dir, 'failure_history.csv'), index=False)
    print(f"  wrote {len(fail_hist)} failure records")
    
    print("Generating weather_data.csv...")
    weather = generate_weather_data()
    weather.to_csv(os.path.join(out_dir, 'weather_data.csv'), index=False)
    print(f"  wrote {len(weather)} daily weather records")
    
    print("Generating operational_context.csv...")
    op_ctx = generate_operational_context(eq_master)
    op_ctx.to_csv(os.path.join(out_dir, 'operational_context.csv'), index=False)
    print(f"  wrote {len(op_ctx)} operational context records")
    
    print("\nAll supplementary data generated in data/metadata/")


if __name__ == '__main__':
    main()
