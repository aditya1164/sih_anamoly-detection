import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os
from feature_engineering import preprocess_data

OUTPUT_MODEL_DIR = "models"

def train_models():
    print("--- Starting Training Pipeline ---")
    df_raw, df_processed = preprocess_data("data/land_records.csv", save_pipeline=True)
    
    X = df_processed.values
    y_true = df_raw['is_anomaly_injected'].values
    
    print("\nTraining Isolation Forest...")
    # Isolation Forest
    iforest = IsolationForest(
        n_estimators=200, 
        contamination=0.05, 
        random_state=42, 
        n_jobs=-1
    )
    iforest.fit(X)
    
    # iForest scores (lower means more anomalous, so we invert it for probability-like)
    iforest_scores = -iforest.score_samples(X) 
    
    print("Training Local Outlier Factor...")
    # LOF (novelty=True to save model for new predictions)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X)
    lof_scores = -lof.score_samples(X)
    
    print("\nEvaluating Models against Injected Ground Truth:")
    
    # Normalize scores to 0-1 range for ensembling
    if_min, if_max = iforest_scores.min(), iforest_scores.max()
    lof_min, lof_max = lof_scores.min(), lof_scores.max()
    
    iforest_norm = (iforest_scores - if_min) / (if_max - if_min)
    lof_norm = (lof_scores - lof_min) / (lof_max - lof_min)
    
    # Ensemble Score (Average)
    ensemble_scores = (iforest_norm + lof_norm) / 2
    
    auc_if = roc_auc_score(y_true, iforest_norm)
    auc_lof = roc_auc_score(y_true, lof_norm)
    auc_ens = roc_auc_score(y_true, ensemble_scores)
    
    print(f"ROC-AUC Isolation Forest : {auc_if:.4f}")
    print(f"ROC-AUC LOF              : {auc_lof:.4f}")
    print(f"ROC-AUC Ensemble         : {auc_ens:.4f}")
    
    # Save models
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    joblib.dump(iforest, os.path.join(OUTPUT_MODEL_DIR, 'iforest_model.pkl'))
    joblib.dump(lof, os.path.join(OUTPUT_MODEL_DIR, 'lof_model.pkl'))
    print("\nModels saved to /models directory.")

if __name__ == "__main__":
    train_models()
