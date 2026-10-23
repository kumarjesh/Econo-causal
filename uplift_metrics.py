import pandas as pd
import numpy as np

# Handle numpy 2.0+ vs 1.x trapezoid
if hasattr(np, 'trapezoid'):
    trapezoid = np.trapezoid
elif hasattr(np, 'trapz'):
    trapezoid = np.trapz
else:
    from scipy.integrate import trapezoid

def compute_uplift_and_qini_curves(predictions_path='data/ite_predictions.csv', n_points=50):
    """
    Computes cumulative Uplift and Qini curves comparing:
      1. Causal ML Targeting (sorted by predicted ITE for $20 discount)
      2. Traditional Predictive Model (sorted by baseline target likelihood)
      3. Random Rollout (un-targeted benchmark diagonal)
    """
    print(f"Loading predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)
    
    # Binary contrast: discount 20 vs control 0
    binary_df = df[df['actual_treatment'].isin([0, 20])].copy().reset_index(drop=True)
    binary_df['treatment'] = (binary_df['actual_treatment'] == 20).astype(int)
    
    n_total = len(binary_df)
    
    # 1. Causal Model Score (predicted ITE for $20)
    causal_score = binary_df['ite_20'].values
    
    # 2. Predictive Model Score (prioritizes users who buy naturally)
    predictive_score = 0.5 * (binary_df['age'].values / 60.0) + 0.5 * binary_df['has_redeemed'].values
    
    def get_cumulative_metrics(score):
        order = np.argsort(-score)
        y_sorted = binary_df['target'].values[order]
        t_sorted = binary_df['treatment'].values[order]
        
        cum_y_t = np.cumsum((t_sorted == 1) & (y_sorted == 1))
        cum_y_c = np.cumsum((t_sorted == 0) & (y_sorted == 1))
        cum_n_t = np.cumsum(t_sorted == 1)
        cum_n_c = np.cumsum(t_sorted == 0)
        
        # Qini curve: cumulative incremental buyers
        qini_raw = cum_y_t - (cum_y_c * (cum_n_t / np.maximum(cum_n_c, 1)))
        
        # Uplift curve: conversion rate difference * cumulative population
        p_t = cum_y_t / np.maximum(cum_n_t, 1)
        p_c = cum_y_c / np.maximum(cum_n_c, 1)
        k_indices = np.arange(1, n_total + 1)
        uplift_raw = (p_t - p_c) * k_indices
        
        # Sample down to n_points for crisp plotting
        sample_indices = np.linspace(0, n_total - 1, n_points, dtype=int)
        pct_population = (sample_indices + 1) / n_total * 100.0
        
        return pct_population, uplift_raw[sample_indices], qini_raw[sample_indices]

    pop_pct, causal_uplift, causal_qini = get_cumulative_metrics(causal_score)
    _, pred_uplift, pred_qini = get_cumulative_metrics(predictive_score)
    
    # Random baseline (straight diagonal from (0,0) to final point)
    random_uplift = causal_uplift[-1] * (pop_pct / 100.0)
    random_qini = causal_qini[-1] * (pop_pct / 100.0)
    
    # Area Under Qini Curve (AUQC) relative to random baseline
    qini_score_causal = float(trapezoid(causal_qini, pop_pct) - trapezoid(random_qini, pop_pct))
    qini_score_pred = float(trapezoid(pred_qini, pop_pct) - trapezoid(random_qini, pop_pct))
    
    curve_df = pd.DataFrame({
        'population_pct': pop_pct,
        'causal_uplift': causal_uplift,
        'predictive_uplift': pred_uplift,
        'random_uplift': random_uplift,
        'causal_qini': causal_qini,
        'predictive_qini': pred_qini,
        'random_qini': random_qini
    })
    
    curve_df.to_csv('data/uplift_curves.csv', index=False)
    print("\n" + "=" * 60)
    print("UPLIFT & QINI CURVE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Incremental Buyers at 100% Rollout: {causal_qini[-1]:.0f}")
    print(f"Causal Model Qini Lift:     {qini_score_causal:.2f}")
    print(f"Predictive Model Qini Lift: {qini_score_pred:.2f}")
    print("Saved curve data to data/uplift_curves.csv")
    
    return curve_df, qini_score_causal, qini_score_pred

if __name__ == "__main__":
    compute_uplift_and_qini_curves()
