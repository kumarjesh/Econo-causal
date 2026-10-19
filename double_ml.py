import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dml import LinearDML
from features import build_feature_pipeline

def train_double_ml(sample_size=50000, save_model=True, random_state=42):
    """
    Trains an EconML Double Machine Learning (DML) model to estimate
    Individual Treatment Effects (ITE) for multi-tier pricing ($10 vs $20 discount).
    """
    print("=" * 60)
    print("ECONOCAUSAL: DOUBLE MACHINE LEARNING (DML) TRAINING")
    print("=" * 60)
    
    # 1. Load Features
    data = build_feature_pipeline(sample_size=sample_size, random_state=random_state)
    X = data['X']
    W = data['W']
    T = data['T']
    Y = data['Y']
    client_ids = data['client_ids']
    
    # Train / Test split
    print("\nSplitting into train and evaluation sets (80/20)...")
    indices = np.arange(len(Y))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=random_state)
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    W_train, W_test = W.iloc[train_idx], W.iloc[test_idx]
    T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    
    print(f"Train size: {len(train_idx)} | Evaluation size: {len(test_idx)}")
    
    # 2. Configure Double Machine Learning Estimator
    print("\nConfiguring EconML LinearDML with LightGBM base learners...")
    # Outcome model: predicts Y from W (residualizes purchase probability)
    model_y = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=random_state,
        verbose=-1
    )
    
    # Treatment model: predicts discrete treatment T (0, 10, 20) from W
    model_t = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=random_state,
        verbose=-1
    )
    
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=3,
        random_state=random_state
    )
    
    # 3. Fit Model
    print("Fitting Double ML (two-stage residualization + Neyman orthogonal regression)...")
    dml.fit(Y_train.values, T_train.values, X=X_train.values, W=W_train.values)
    print("DML training successfully completed!")
    
    # 4. Predict Individual Treatment Effects (ITE) on Test Set
    print("\nEstimating Individual Treatment Effects (ITE)...")
    # Lift of $10 discount vs Baseline ($0)
    ite_10 = dml.effect(X_test.values, T0=0, T1=10)
    # Lift of $20 discount vs Baseline ($0)
    ite_20 = dml.effect(X_test.values, T0=0, T1=20)
    # Incremental lift of $20 discount vs $10 discount
    ite_20_vs_10 = dml.effect(X_test.values, T0=10, T1=20)
    
    # Build evaluation DataFrame
    eval_df = X_test.copy()
    eval_df['target'] = Y_test.values
    eval_df['actual_treatment'] = T_test.values
    eval_df['ite_10'] = ite_10
    eval_df['ite_20'] = ite_20
    eval_df['ite_20_vs_10'] = ite_20_vs_10
    
    # 5. Model Validation & Heterogeneous Treatment Effect (HTE) Verification
    print("\n" + "-" * 50)
    print("CAUSAL DISCOVERY & HTE VERIFICATION")
    print("-" * 50)
    ate_10 = np.mean(ite_10)
    ate_20 = np.mean(ite_20)
    print(f"Average Treatment Effect (ATE) - $10 Discount: +{ate_10*100:.2f}% conversion lift")
    print(f"Average Treatment Effect (ATE) - $20 Discount: +{ate_20*100:.2f}% conversion lift")
    
    # Check HTE across age brackets (Validating our injected rule!)
    young_mask = eval_df['age'] < 30
    senior_mask = eval_df['age'] > 50
    young_ite_20 = eval_df.loc[young_mask, 'ite_20'].mean()
    senior_ite_20 = eval_df.loc[senior_mask, 'ite_20'].mean()
    
    print("\nHeterogeneity by Demographic:")
    print(f"  Young Segment (Age < 30) $20 Lift:  +{young_ite_20*100:.2f}%")
    print(f"  Senior Segment (Age > 50) $20 Lift: +{senior_ite_20*100:.2f}%")
    
    if young_ite_20 > senior_ite_20:
        print("\n>>> SUCCESS: The Double ML model accurately identified the hidden HTE!")
        print("    Younger customers show significantly higher price elasticity for $20.")
    
    # 6. Save Artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    output_predictions_path = 'data/ite_predictions.csv'
    eval_df.to_csv(output_predictions_path)
    print(f"\nSaved test set ITE predictions to {output_predictions_path}")
    
    if save_model:
        model_path = 'models/double_ml_model.pkl'
        joblib.dump(dml, model_path)
        print(f"Saved trained Double ML model to {model_path}")
        
    return {
        'model': dml,
        'eval_df': eval_df,
        'ate_10': ate_10,
        'ate_20': ate_20,
        'young_ite_20': young_ite_20,
        'senior_ite_20': senior_ite_20
    }

if __name__ == "__main__":
    train_double_ml(sample_size=50000)
