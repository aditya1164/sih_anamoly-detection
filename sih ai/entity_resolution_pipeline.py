import pandas as pd
import numpy as np
import jellyfish
import networkx as nx
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 0. SYNTHETIC DATA GENERATION (Ingestion)
# ==========================================
def generate_synthetic_data():
    """Generates synthetic tabular data with some near-duplicates."""
    data = [
        {"record_id": 1, "first_name": "Jonathan", "last_name": "Doe", "zip_code": "10001", "phone": "555-0100", "dob": "1980-01-01", "address": "123 Main St", "label": 1},
        {"record_id": 2, "first_name": "Jonathon", "last_name": "Doe", "zip_code": "10001", "phone": "555-0100", "dob": "1980-01-01", "address": "123 Main Street", "label": 1}, # Duplicate of 1
        {"record_id": 3, "first_name": "Jane", "last_name": "Smith", "zip_code": "90210", "phone": "555-0101", "dob": "1992-05-15", "address": "456 Oak Ave", "label": 2},
        {"record_id": 4, "first_name": "J.", "last_name": "Smith", "zip_code": "90210", "phone": "555-0101", "dob": "1992-05-15", "address": "456 Oak Avenue", "label": 2}, # Duplicate of 3
        {"record_id": 5, "first_name": "Alice", "last_name": "Johnson", "zip_code": "60601", "phone": "555-0102", "dob": "1975-11-20", "address": "789 Pine Rd", "label": 3},
        {"record_id": 6, "first_name": "Bob", "last_name": "Williams", "zip_code": "10001", "phone": "555-0103", "dob": "1988-03-10", "address": "321 Elm St", "label": 4},
        {"record_id": 7, "first_name": "Jack", "last_name": "Doe", "zip_code": "10001", "phone": "555-0199", "dob": "1990-12-12", "address": "999 Other St", "label": 5} # Negative pair for Doe block
    ]
    return pd.DataFrame(data)

# ==========================================
# 1. PRE-PROCESSING NODE
# ==========================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("-> Running Pre-processing Node...")
    df_clean = df.copy()
    # Clean strings: lowercase and strip whitespace
    for col in ['first_name', 'last_name', 'address']:
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    # Clean phone numbers (remove non-numeric)
    df_clean['phone'] = df_clean['phone'].astype(str).str.replace(r'\D+', '', regex=True)
    return df_clean

# ==========================================
# 2. BLOCKING NODE
# ==========================================
def generate_candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    print("-> Running Blocking Node...")
    df_block = df.copy()
    
    # Pass 1: Block by Zip Code + First Letter of Last Name
    df_block['last_name_initial'] = df_block['last_name'].astype(str).str[0]
    df_block['block_key_1'] = df_block['zip_code'].astype(str) + "_" + df_block['last_name_initial']
    
    blocks_1 = df_block.merge(df_block, on='block_key_1', suffixes=('_A', '_B'))
    pairs_1 = blocks_1[blocks_1['record_id_A'] < blocks_1['record_id_B']][['record_id_A', 'record_id_B', 'label_A', 'label_B']]
    
    # Pass 2: Block by Phone Number (excluding empties)
    df_phone = df_block[df_block['phone'].notnull() & (df_block['phone'] != '')].copy()
    blocks_2 = df_phone.merge(df_phone, on='phone', suffixes=('_A', '_B'))
    pairs_2 = blocks_2[blocks_2['record_id_A'] < blocks_2['record_id_B']][['record_id_A', 'record_id_B', 'label_A', 'label_B']]
    
    all_pairs = pd.concat([pairs_1, pairs_2]).drop_duplicates(subset=['record_id_A', 'record_id_B']).reset_index(drop=True)
    
    # Create target label for model training (1 if same entity, 0 otherwise)
    all_pairs['is_duplicate_target'] = (all_pairs['label_A'] == all_pairs['label_B']).astype(int)
    all_pairs = all_pairs.drop(columns=['label_A', 'label_B'])
    
    print(f"   Generated {len(all_pairs)} candidate pairs.")
    return all_pairs

# ==========================================
# 3. FEATURE EXTRACTION NODE
# ==========================================
def extract_features(pairs_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    print("-> Running Feature Extraction Node...")
    data_df = data_df.set_index('record_id')
    merged = pairs_df.join(data_df, on='record_id_A').join(data_df, on='record_id_B', lsuffix='_A', rsuffix='_B')
    
    features = pd.DataFrame(index=merged.index)
    features['record_id_A'] = merged['record_id_A']
    features['record_id_B'] = merged['record_id_B']
    if 'is_duplicate_target' in merged.columns:
        features['label'] = merged['is_duplicate_target']
    
    def safe_str(s): return "" if pd.isna(s) else str(s)
    
    # Jaro-Winkler for names
    def jw_sim(row, col):
        return jellyfish.jaro_winkler_similarity(safe_str(row[col+'_A']), safe_str(row[col+'_B']))
    
    features['sim_first_name'] = merged.apply(lambda r: jw_sim(r, 'first_name'), axis=1)
    features['sim_last_name'] = merged.apply(lambda r: jw_sim(r, 'last_name'), axis=1)
    
    # Levenshtein for address
    def lev_sim(row, col):
        s1, s2 = safe_str(row[col+'_A']), safe_str(row[col+'_B'])
        if not s1 and not s2: return 0.0
        max_len = max(len(s1), len(s2))
        if max_len == 0: return 0.0
        dist = jellyfish.levenshtein_distance(s1, s2)
        return 1.0 - (dist / max_len)
        
    features['sim_address'] = merged.apply(lambda r: lev_sim(r, 'address'), axis=1)
    
    # Exact Match for DOB
    features['exact_dob'] = (merged['dob_A'] == merged['dob_B']).astype(float)
    
    return features

# ==========================================
# 4. MODEL NODE
# ==========================================
def train_and_score_model(feature_matrix: pd.DataFrame, threshold: float = 0.50):
    print("-> Running Model Training & Scoring Node...")
    feature_cols = ['sim_first_name', 'sim_last_name', 'sim_address', 'exact_dob']
    X = feature_matrix[feature_cols]
    y = feature_matrix['label']
    
    model = XGBClassifier(
        n_estimators=10, 
        max_depth=3, 
        learning_rate=0.1, 
        eval_metric='logloss',
        random_state=42
    )
    
    # Train on all data for this small synthetic example
    model.fit(X, y)
    
    # Predict probabilities
    probabilities = model.predict_proba(X)[:, 1]
    
    results = feature_matrix[['record_id_A', 'record_id_B']].copy()
    results['duplicate_probability'] = probabilities
    results['is_duplicate'] = (probabilities >= threshold).astype(int)
    
    print(f"   Identified {results['is_duplicate'].sum()} positive matches.")
    return results

# ==========================================
# 5. CLUSTERING NODE
# ==========================================
def generate_entity_clusters(scored_pairs: pd.DataFrame, all_record_ids: list) -> pd.DataFrame:
    print("-> Running Clustering Node...")
    matches = scored_pairs[scored_pairs['is_duplicate'] == 1]
    
    G = nx.Graph()
    G.add_nodes_from(all_record_ids)
    
    edges = list(zip(matches['record_id_A'], matches['record_id_B']))
    G.add_edges_from(edges)
    
    clusters = []
    cluster_id_counter = 1
    for component in nx.connected_components(G):
        cluster_id = f"ENTITY_{cluster_id_counter:04d}"
        for node in component:
            clusters.append({'record_id': node, 'cluster_id': cluster_id})
        cluster_id_counter += 1
        
    return pd.DataFrame(clusters)

# ==========================================
# 6. GOLDEN RECORD GENERATION
# ==========================================
def generate_golden_records(df: pd.DataFrame, entity_map: pd.DataFrame) -> pd.DataFrame:
    print("-> Generating Golden Records...")
    df_clustered = df.merge(entity_map, on='record_id', how='left')
    
    # Simple survivorship rule: Take the first non-null value within the cluster
    golden_records = df_clustered.groupby('cluster_id').first().reset_index()
    
    return golden_records

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Starting Entity Resolution Pipeline...\n")
    
    # 0. Ingestion
    raw_data = generate_synthetic_data()
    print(f"Raw Data:\n{raw_data[['record_id', 'first_name', 'last_name', 'phone']]}\n")
    
    # 1. Pre-process
    cleaned_data = preprocess_data(raw_data)
    
    # 2. Blocking
    candidate_pairs = generate_candidate_pairs(cleaned_data)
    
    # 3. Feature Extraction
    features = extract_features(candidate_pairs, cleaned_data)
    
    # 4. Model Scoring
    scored_pairs = train_and_score_model(features, threshold=0.5)
    
    # 5. Clustering
    entity_map = generate_entity_clusters(scored_pairs, cleaned_data['record_id'].tolist())
    
    # 6. Golden Record Output
    final_output = generate_golden_records(raw_data, entity_map)
    
    print("\n==========================================")
    print("FINAL GOLDEN RECORDS:")
    print("==========================================")
    print(final_output[['cluster_id', 'record_id', 'first_name', 'last_name', 'address']])
    print("==========================================\n")
