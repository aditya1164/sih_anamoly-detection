import shap
import joblib
import pandas as pd
import numpy as np
import os

# Resolve paths dynamically so it works no matter where VS Code is opened
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models")

class ExplainabilityEngine:
    def __init__(self):
        # Load preprocessor and model
        self.preprocessor = joblib.load(os.path.join(OUTPUT_MODEL_DIR, 'preprocessor.pkl'))
        self.feature_names = joblib.load(os.path.join(OUTPUT_MODEL_DIR, 'feature_names.pkl'))
        self.iforest = joblib.load(os.path.join(OUTPUT_MODEL_DIR, 'iforest_model.pkl'))
        
        # Initialize SHAP TreeExplainer for Isolation Forest
        self.explainer = shap.TreeExplainer(self.iforest)

    def explain_instance(self, df_instance):
        """
        Takes a raw pandas DataFrame row (instance), preprocesses it, 
        and calculates SHAP values to explain the anomaly.
        """
        # Preprocess
        X_transformed = self.preprocessor.transform(df_instance)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_transformed)
        
        # SHAP returns a list for iForest (size 1). 
        # Usually positive SHAP values push the model towards predicting anomaly (lower scores in sklearn).
        # Since sklearn iForest uses negative anomaly scores, higher absolute SHAP means higher impact.
        
        if isinstance(shap_values, list):
            vals = shap_values[0][0]
        else:
            vals = shap_values[0]
            
        # Map back to feature names
        feature_impacts = []
        for i, f_name in enumerate(self.feature_names):
            feature_impacts.append({
                "feature": f_name,
                "impact": float(vals[i]),
                "abs_impact": abs(float(vals[i]))
            })
            
        # Sort by absolute impact descending
        feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
        
        # Format human-readable explanations for top 3
        top_factors = []
        for item in feature_impacts[:3]:
            # Simple heuristic mapping for business friendly language
            feat = item['feature']
            if "comp_per_sqm" in feat:
                top_factors.append("Irregular Compensation Rate per Sq.Meter")
            elif "families_per_hectare" in feat:
                top_factors.append("Suspiciously High/Low Displaced Families Density")
            elif "time_delta_days" in feat:
                top_factors.append("Anomalous Project Timeline (Notification to Award)")
            elif "area_discrepancy_ratio" in feat:
                top_factors.append("Mismatch between Cadastral and Survey Area")
            else:
                top_factors.append(f"Irregularities in {feat}")
                
        return top_factors

if __name__ == "__main__":
    # Test Explainer
    from feature_engineering import engineer_features
    print("Testing Explainer...")
    df = pd.read_csv("data/land_records.csv")
    engine = ExplainabilityEngine()
    
    # Pick a known anomaly
    anomaly_idx = df[df['is_anomaly_injected'] == 1].index[0]
    sample = engineer_features(df.iloc[[anomaly_idx]])
    
    explanations = engine.explain_instance(sample)
    print(f"Top Risk Factors for Record {df.iloc[anomaly_idx]['record_id']}:")
    for ex in explanations:
        print(f"- {ex}")
