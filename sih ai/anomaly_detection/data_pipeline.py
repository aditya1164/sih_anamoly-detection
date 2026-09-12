import pandas as pd
import numpy as np
import os
import random
from datetime import timedelta, datetime

# Configuration
NUM_RECORDS = 10500
ANOMALY_RATIO = 0.05
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "land_records.csv")

def generate_synthetic_data(num_records=NUM_RECORDS, anomaly_ratio=ANOMALY_RATIO):
    """
    Generates synthetic land acquisition data for the SIH problem statement.
    Injects realistic anomalies representing fraud and irregularities.
    """
    print(f"Generating {num_records} records with {anomaly_ratio*100}% anomalies...")
    np.random.seed(42)
    random.seed(42)

    # Base Definitions
    states = ["Maharashtra", "Gujarat", "Karnataka", "UP", "MP"]
    districts = [f"District_{i}" for i in range(1, 21)]
    land_types = ["Agricultural", "Commercial", "Residential", "Forest"]
    project_types = ["Highway", "Railway", "Urban Development", "Dam"]
    
    data = []
    start_date = datetime(2020, 1, 1)

    for i in range(num_records):
        is_anomaly = np.random.rand() < anomaly_ratio
        
        # Core attributes
        state = random.choice(states)
        district = random.choice(districts)
        land_type = random.choice(land_types)
        project_type = random.choice(project_types)
        
        # Spatial/Area
        cadastral_area_sqm = np.random.uniform(500, 20000)
        
        # Base compensation (Circle Rates rough estimates)
        base_rate = {
            "Agricultural": np.random.uniform(100, 500),
            "Residential": np.random.uniform(1000, 3000),
            "Commercial": np.random.uniform(3000, 8000),
            "Forest": np.random.uniform(50, 200)
        }[land_type]
        
        # Timelines
        notification_date = start_date + timedelta(days=random.randint(0, 1000))
        
        if not is_anomaly:
            # Normal generation
            survey_area_sqm = cadastral_area_sqm * np.random.uniform(0.95, 1.05) # Minor discrepancy
            compensation_amount = survey_area_sqm * base_rate * np.random.uniform(0.9, 1.2)
            displaced_families = int(survey_area_sqm / 5000 * np.random.uniform(0.5, 1.5))
            award_date = notification_date + timedelta(days=random.randint(180, 730)) # 6mo to 2yrs
            rr_disbursement_ratio = np.random.uniform(0.8, 1.0)
            anomaly_type = "None"
        else:
            # Inject Anomaly
            survey_area_sqm = cadastral_area_sqm
            compensation_amount = survey_area_sqm * base_rate
            displaced_families = int(survey_area_sqm / 5000)
            award_date = notification_date + timedelta(days=random.randint(180, 730))
            rr_disbursement_ratio = np.random.uniform(0.8, 1.0)
            
            anomaly_types = ["Compensation Outlier", "Demographic Spike", "Timeline Irregularity", "Spatial Discrepancy"]
            anomaly_type = random.choice(anomaly_types)
            
            if anomaly_type == "Compensation Outlier":
                compensation_amount *= np.random.uniform(4.0, 10.0) # 400-1000% inflated
            elif anomaly_type == "Demographic Spike":
                displaced_families += int(np.random.uniform(20, 50)) # Phantom families
            elif anomaly_type == "Timeline Irregularity":
                award_date = notification_date + timedelta(days=random.randint(5, 30)) # Unusually rapid
            elif anomaly_type == "Spatial Discrepancy":
                survey_area_sqm *= np.random.uniform(1.5, 3.0) # Survey claims much more area than cadastral map
                compensation_amount = survey_area_sqm * base_rate
        
        data.append({
            "record_id": f"REC_{100000 + i}",
            "state": state,
            "district": district,
            "land_type": land_type,
            "project_type": project_type,
            "cadastral_area_sqm": round(cadastral_area_sqm, 2),
            "survey_area_sqm": round(survey_area_sqm, 2),
            "compensation_amount": round(compensation_amount, 2),
            "displaced_families": max(0, displaced_families),
            "notification_date": notification_date.strftime("%Y-%m-%d"),
            "award_date": award_date.strftime("%Y-%m-%d"),
            "rr_disbursement_ratio": round(rr_disbursement_ratio, 2),
            "is_anomaly_injected": int(is_anomaly),
            "injected_anomaly_type": anomaly_type
        })
        
    df = pd.DataFrame(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data successfully generated at {OUTPUT_FILE}")
    print(f"Total Records: {len(df)}")
    print(f"Anomalies Injected: {df['is_anomaly_injected'].sum()} ({(df['is_anomaly_injected'].sum()/len(df))*100:.2f}%)")

if __name__ == "__main__":
    generate_synthetic_data()
