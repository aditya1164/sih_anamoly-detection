import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

OUTPUT_MODEL_DIR = "models"

def engineer_features(df):
    """
    Creates domain-specific features for anomaly detection.
    """
    df = df.copy()
    
    # 1. Compensation per square meter
    df['comp_per_sqm'] = df['compensation_amount'] / (df['survey_area_sqm'] + 1e-5)
    
    # 2. Demographic ratio (Families per hectare)
    df['families_per_hectare'] = df['displaced_families'] / (df['survey_area_sqm'] / 10000 + 1e-5)
    
    # 3. Timeline (Days from Notification to Award)
    df['notification_date'] = pd.to_datetime(df['notification_date'])
    df['award_date'] = pd.to_datetime(df['award_date'])
    df['time_delta_days'] = (df['award_date'] - df['notification_date']).dt.days
    
    # 4. Spatial Discrepancy Ratio
    df['area_discrepancy_ratio'] = df['survey_area_sqm'] / (df['cadastral_area_sqm'] + 1e-5)
    
    return df

def build_preprocessor():
    """
    Builds the Scikit-Learn preprocessing pipeline.
    """
    numeric_features = [
        'comp_per_sqm', 
        'families_per_hectare', 
        'time_delta_days', 
        'area_discrepancy_ratio',
        'rr_disbursement_ratio'
    ]
    
    categorical_features = ['land_type', 'project_type', 'district']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_features, categorical_features

def preprocess_data(csv_path="data/land_records.csv", save_pipeline=True):
    print("Loading data for feature engineering...")
    df = pd.read_csv(csv_path)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    preprocessor, num_cols, cat_cols = build_preprocessor()
    
    print("Fitting preprocessor...")
    X_transformed = preprocessor.fit_transform(df)
    
    # Get feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)
    
    if save_pipeline:
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(OUTPUT_MODEL_DIR, 'preprocessor.pkl'))
        joblib.dump(feature_names, os.path.join(OUTPUT_MODEL_DIR, 'feature_names.pkl'))
        print("Preprocessor saved.")
        
    return df, pd.DataFrame(X_transformed, columns=feature_names)

if __name__ == "__main__":
    df_raw, df_processed = preprocess_data()
    print(f"Processed shape: {df_processed.shape}")
