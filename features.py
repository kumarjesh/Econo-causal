import pandas as pd
import numpy as np

def build_feature_pipeline(
    synthetic_path='data/synthetic_dynamic_pricing.csv',
    clients_path='data/clients.csv',
    sample_size=None,
    random_state=42
):
    """
    Constructs the feature matrices required for Double Machine Learning.
    
    Variables:
      - Y: Target outcome (purchase conversion: 0 or 1)
      - T: Treatment (discount_amount: 0, 10, or 20)
      - X: Heterogeneity features (attributes that moderate how users react to treatments, e.g. age, delay)
      - W: Confounder / Control features (variables affecting treatment assignment or outcome baseline)
    """
    print(f"Loading data from {synthetic_path} and {clients_path}...")
    df_syn = pd.read_csv(synthetic_path, index_col='client_id')
    df_clients = pd.read_csv(clients_path, index_col='client_id')
    
    # Merge client date attributes
    df_syn = df_syn.join(df_clients[['first_issue_date', 'first_redeem_date']], how='inner')
    
    if sample_size and sample_size < len(df_syn):
        print(f"Subsampling to {sample_size} records for training/iteration...")
        df_syn = df_syn.sample(n=sample_size, random_state=random_state)
    
    # Feature engineering
    print("Engineering demographic and behavioral features...")
    first_issue_dt = pd.to_datetime(df_syn['first_issue_date'], errors='coerce')
    first_redeem_dt = pd.to_datetime(df_syn['first_redeem_date'], errors='coerce')
    
    # Reference epoch in days (from min date)
    ref_date = pd.to_datetime('2017-01-01')
    first_issue_days = (first_issue_dt - ref_date).dt.days.fillna(0).clip(lower=0)
    has_redeemed = first_redeem_dt.notna().astype(int)
    
    first_redeem_days = (first_redeem_dt - ref_date).dt.days.fillna(0).clip(lower=0)
    issue_redeem_delay = (first_redeem_days - first_issue_days).clip(lower=0).fillna(0)
    
    # One-hot encode gender
    gender_M = (df_syn['gender'] == 'M').astype(int)
    gender_F = (df_syn['gender'] == 'F').astype(int)
    gender_U = (df_syn['gender'] == 'U').astype(int)
    
    # Clean Age (clip reasonable bounds 18 to 90)
    age_clean = df_syn['age'].fillna(df_syn['age'].median()).clip(lower=18, upper=90)
    
    # Construct DataFrame of covariates
    covariates_df = pd.DataFrame({
        'age': age_clean,
        'gender_M': gender_M,
        'gender_F': gender_F,
        'gender_U': gender_U,
        'has_redeemed': has_redeemed,
        'first_issue_days': first_issue_days,
        'first_redeem_days': first_redeem_days,
        'issue_redeem_delay': issue_redeem_delay
    }, index=df_syn.index)
    
    # Heterogeneity features X (driving ITE variance across customer segments)
    X = covariates_df[['age', 'issue_redeem_delay', 'has_redeemed']]
    
    # Confounders / Controls W (common causes influencing propensity and baseline conversion)
    W = covariates_df[['gender_M', 'gender_F', 'gender_U', 'first_issue_days', 'first_redeem_days']]
    
    # Treatment T and Outcome Y
    T = df_syn['discount_amount'].astype(int)
    Y = df_syn['target'].astype(int)
    
    print(f"Feature pipeline complete! Processed {len(df_syn)} clients.")
    print(f"Treatment distribution:\n{T.value_counts().sort_index()}")
    print(f"Target conversion rate: {Y.mean():.4f}")
    
    return {
        'df': df_syn,
        'covariates': covariates_df,
        'X': X,
        'W': W,
        'T': T,
        'Y': Y,
        'client_ids': df_syn.index.tolist()
    }

if __name__ == "__main__":
    data = build_feature_pipeline(sample_size=10000)
    print("X head:\n", data['X'].head())
    print("W head:\n", data['W'].head())
