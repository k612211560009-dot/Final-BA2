"""
Load equipment data from CSV and generate JavaScript data file for dashboard
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dashboard"
PREDICTIONS_DIR = Path(__file__).parent.parent.parent / "predictions"
EQUIPMENT_SUMMARY = DATA_DIR / "equipment_summary.csv"
ALERTS_SUMMARY = DATA_DIR / "alerts_summary.csv"
PREDICTION_SUMMARY = PREDICTIONS_DIR / "prediction_summary.csv"
OUTPUT_JS = Path(__file__).parent / "data.js"

def load_equipment_data():
    """Load equipment summary data"""
    df = pd.read_csv(EQUIPMENT_SUMMARY)
    
    # Calculate summary statistics
    total_equipment = len(df)
    critical_count = len(df[df['risk_level'] == 'Critical'])
    high_count = len(df[df['risk_level'] == 'High'])
    medium_count = len(df[df['risk_level'] == 'Medium'])
    low_count = len(df[df['risk_level'] == 'Low'])
    
    # Calculate predicted downtime (equipment with days_to_maintenance < 7)
    urgent_equipment = df[df['days_to_maintenance'] <= 7]
    predicted_downtime = len(urgent_equipment) * 2.5  # Assume 2.5 hours avg downtime per equipment
    
    return {
        'total_equipment': total_equipment,
        'critical_alerts': critical_count + high_count,
        'predicted_downtime': f"{predicted_downtime:.0f}h",
        'risk_distribution': {
            'Critical': int(critical_count),
            'High': int(high_count),
            'Moderate': int(medium_count),
            'Low': int(low_count)
        },
        'equipment_list': df.to_dict('records')
    }

def load_alerts_data():
    """Load alerts summary data"""
    df = pd.read_csv(ALERTS_SUMMARY)
    return df.to_dict('records')

def load_predictions_data():
    """Load prediction summary data"""
    if not PREDICTION_SUMMARY.exists():
        print("  ⚠ Prediction summary not found, skipping predictions")
        return []
    
    df = pd.read_csv(PREDICTION_SUMMARY)
    
    # Add formatted columns for display
    predictions = df.to_dict('records')
    
    for pred in predictions:
        # Format confidence
        if 'confidence' in pred:
            pred['confidence_pct'] = f"{pred['confidence']*100:.1f}%"
        
        # Format RUL if exists
        if 'predicted_rul' in pred and pd.notna(pred['predicted_rul']):
            pred['rul_display'] = f"{pred['predicted_rul']:.0f} days"
        
        # Format efficiency if exists
        if 'predicted_efficiency' in pred and pd.notna(pred['predicted_efficiency']):
            pred['efficiency_display'] = f"{pred['predicted_efficiency']*100:.1f}%"
        
        # Format anomaly probability if exists
        if 'anomaly_probability' in pred and pd.notna(pred['anomaly_probability']):
            pred['risk_level'] = 'High' if pred['anomaly_probability'] > 0.7 else 'Medium' if pred['anomaly_probability'] > 0.4 else 'Low'
    
    print(f"  Loaded {len(predictions)} equipment predictions")
    return predictions

def group_equipment_by_type(equipment_list):
    """Group equipment by type to create area data"""
    df = pd.DataFrame(equipment_list)
    
    areas = []
    type_mapping = {
        'Pipeline': {'name': 'Hệ thống Pipeline', 'id': 'pipeline'},
        'Bearing': {'name': 'Hệ thống Bearing', 'id': 'bearing'},
        'Pump': {'name': 'Hệ thống Pump', 'id': 'pump'},
        'Turbine': {'name': 'Hệ thống Turbine', 'id': 'turbine'},
        'Compressor': {'name': 'Hệ thống Compressor', 'id': 'compressor'}
    }
    
    for eq_type, group in df.groupby('equipment_type'):
        if eq_type in type_mapping:
            avg_health = group['current_health'].mean()
            avg_rul_days = group['days_to_maintenance'].mean()
            # Convert boolean to int before sum
            anomaly_count = (group['is_anomaly'] == True).sum()
            
            # Determine risk level based on critical/high count
            critical_high = len(group[group['risk_level'].isin(['Critical', 'High'])])
            if critical_high > 0:
                risk = 'critical' if 'Critical' in group['risk_level'].values else 'high'
            elif avg_health < 0.6:
                risk = 'moderate'
            else:
                risk = 'low'
            
            # Estimate downtime (equipment with RUL < 7 days)
            urgent = len(group[group['days_to_maintenance'] <= 7])
            downtime = urgent * 2.5
            
            areas.append({
                'id': type_mapping[eq_type]['id'],
                'name': type_mapping[eq_type]['name'],
                'risk': risk,
                'avgRUL': int(avg_rul_days * 24),  # Convert days to hours
                'downtime': int(downtime),
                'equipment_count': len(group),
                'anomaly_count': int(anomaly_count)
            })
    
    return areas

def create_equipment_alerts(equipment_list, alerts_list):
    """Create alerts grouped by equipment type"""
    alerts_by_area = {}
    
    # Map equipment types to area IDs
    type_to_area = {
        'Pipeline': 'pipeline',
        'Bearing': 'bearing',
        'Pump': 'pump',
        'Turbine': 'turbine',
        'Compressor': 'compressor'
    }
    
    df = pd.DataFrame(equipment_list)
    
    for eq_type, group in df.groupby('equipment_type'):
        area_id = type_to_area.get(eq_type, eq_type.lower())
        area_alerts = []
        
        # Get high priority equipment (anomalies or low health)
        priority_equipment = group[
            (group['is_anomaly'] == True) | 
            (group['current_health'] < 0.6) |
            (group['days_to_maintenance'] <= 30)
        ].head(5)  # Top 5 per area
        
        for idx, row in priority_equipment.iterrows():
            # Determine condition based on health
            health = row['current_health']
            if health < 0.4:
                condition = 'critical'
                probability = 85 + (0.4 - health) * 100
            elif health < 0.6:
                condition = 'moderate'
                probability = 65 + (0.6 - health) * 50
            else:
                condition = 'normal'
                probability = 45
            
            area_alerts.append({
                'id': row['equipment_id'],
                'name': row['equipment_id'],
                'condition': condition,
                'rul': int(row['days_to_maintenance'] * 24),  # Days to hours
                'probability': min(int(probability), 95),
                'timestamp': row['last_updated']
            })
        
        if area_alerts:
            alerts_by_area[area_id] = area_alerts
    
    return alerts_by_area

def generate_javascript_file(summary_data, alerts_data, predictions_data=None):
    """Generate JavaScript file with data (deprecated - use generate_javascript_file_with_predictions)"""
    # This function is kept for backward compatibility but calls the new one
    if predictions_data is None:
        predictions_data = []
    return generate_javascript_file_with_predictions(summary_data, alerts_data, predictions_data)

def generate_javascript_file_with_predictions(summary_data, alerts_data, predictions_data):
    """Generate JavaScript file with equipment data and predictions"""
    # Use existing generator and add predictions
    areas = group_equipment_by_type(summary_data['equipment_list'])
    equipment_alerts = create_equipment_alerts(summary_data['equipment_list'], alerts_data)
    
    # High priority alerts
    high_priority = [
        alert for alert in alerts_data 
        if alert.get('priority') in ['High', 'Critical']
    ]
    
    # Convert to JS format
    stats_js = json.dumps(summary_data, ensure_ascii=False, indent=2)
    areas_js = json.dumps(areas, ensure_ascii=False, indent=2)
    alerts_js = json.dumps(equipment_alerts, ensure_ascii=False, indent=2)
    priority_js = json.dumps(high_priority, ensure_ascii=False, indent=2)
    predictions_js = json.dumps(predictions_data, ensure_ascii=False, indent=2)
    
    # Generate JS file
    js_content = f"""// Equipment Dashboard Data
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

const DASHBOARD_STATS = {stats_js};

const equipmentAreas = {areas_js};

const equipmentAlerts = {alerts_js};

const highPriorityAlerts = {priority_js};

const equipmentPredictions = {predictions_js};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = {{
        DASHBOARD_STATS,
        equipmentAreas,
        equipmentAlerts,
        highPriorityAlerts,
        equipmentPredictions
    }};
}}
"""
    
    with open(OUTPUT_JS, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f" Generated JavaScript data file: {OUTPUT_JS}")
    print(f"  - Total equipment: {summary_data['total_equipment']}")
    print(f"  - Critical alerts: {summary_data['critical_alerts']}")
    print(f"  - Areas: {len(areas)}")
    print(f"  - Alert groups: {len(equipment_alerts)}")
    print(f"  - Predictions: {len(predictions_data)}")

def main():
    """Main execution"""
    print("Loading Equipment Data for Dashboard")
    
    # Load data
    print("\n[1/5] Loading equipment summary...")
    summary_data = load_equipment_data()
    
    print("[2/5] Loading alerts...")
    alerts_data = load_alerts_data()
    
    print("[3/5] Loading predictions...")
    predictions_data = load_predictions_data()
    
    print("[4/5] Processing equipment groups...")
    areas = group_equipment_by_type(summary_data['equipment_list'])
    print(f"  Found {len(areas)} equipment areas")
    
    print("[5/5] Generating JavaScript file...")
    generate_javascript_file_with_predictions(summary_data, alerts_data, predictions_data)

if __name__ == "__main__":
    main()
