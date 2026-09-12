import os
import subprocess

def run_pipeline():
    print("==================================================")
    print(" SIH Anomaly Detection Pipeline Execution Script ")
    print("==================================================")

    # Step 1: Generate Data
    print("\n[1/4] Generating synthetic land records...")
    subprocess.run(["python", "data_pipeline.py"], check=True)
    
    # Step 2: Feature Engineering & Preprocessing
    print("\n[2/4] Engineering features and fitting preprocessor...")
    subprocess.run(["python", "feature_engineering.py"], check=True)
    
    # Step 3: Model Training
    print("\n[3/4] Training Isolation Forest and LOF Models...")
    subprocess.run(["python", "model_trainer.py"], check=True)
    
    # Step 4: Testing Explainability
    print("\n[4/4] Testing SHAP Explainability Engine...")
    subprocess.run(["python", "explainability_engine.py"], check=True)
    
    print("\n==================================================")
    print(" Pipeline Completed Successfully! ")
    print(" To start the API, run: python api_endpoint.py ")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()
