import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple

# -------------------------------------------------------------------
# Core Statistical Drift Metrics
# -------------------------------------------------------------------
def calculate_psi(
    baseline: np.ndarray,
    target: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, str]:
    """
    Computes the Population Stability Index (PSI) between a baseline (training)
    feature distribution and a target (production/serving) distribution.
    
    Formula:
        PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
        
    Interpretation Thresholds (Industry Standard):
        - PSI < 0.10: Stable (No significant shift; model assumptions hold)
        - 0.10 <= PSI < 0.25: Moderate Shift (Warning; monitor conversion metrics)
        - PSI >= 0.25: Severe Drift (Action required; triggers model retraining)
    """
    baseline_clean = np.asarray(baseline, dtype=float)
    target_clean = np.asarray(target, dtype=float)
    
    # Remove NaN values
    baseline_clean = baseline_clean[~np.isnan(baseline_clean)]
    target_clean = target_clean[~np.isnan(target_clean)]
    
    if len(baseline_clean) == 0 or len(target_clean) == 0:
        return 0.0, "UNKNOWN"
        
    # Determine quantile bin edges from baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(baseline_clean, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    # Ensure strictly increasing bin edges (handles low-cardinality values)
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 3:
        # Fallback to uniform linspace if data is discrete/binary
        unique_edges = np.linspace(min(baseline_clean.min(), target_clean.min()) - 1e-5,
                                   max(baseline_clean.max(), target_clean.max()) + 1e-5,
                                   num_buckets + 1)
                                   
    # Calculate bucket counts
    base_counts, _ = np.histogram(baseline_clean, bins=unique_edges)
    target_counts, _ = np.histogram(target_clean, bins=unique_edges)
    
    # Normalize to proportions
    base_pct = base_counts / len(baseline_clean)
    target_pct = target_counts / len(target_clean)
    
    # Apply epsilon smoothing to prevent division by zero or log(0)
    base_pct = np.clip(base_pct, epsilon, 1.0)
    target_pct = np.clip(target_pct, epsilon, 1.0)
    
    # Recalculate normalized proportions after clipping
    base_pct = base_pct / np.sum(base_pct)
    target_pct = target_pct / np.sum(target_pct)
    
    # Calculate PSI sum
    psi_value = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
    psi_value = float(np.round(psi_value, 4))
    
    if psi_value < 0.10:
        status = "STABLE"
    elif psi_value < 0.25:
        status = "MODERATE_DRIFT"
    else:
        status = "SEVERE_DRIFT"
        
    return psi_value, status

def calculate_ks_test(baseline: np.ndarray, target: np.ndarray) -> Tuple[float, float, str]:
    """
    Performs the Two-Sample Kolmogorov-Smirnov (KS) test for continuous feature distributions.
    
    Null Hypothesis (H0): Both cohorts are drawn from the identical continuous distribution.
    If p-value < 0.05, we reject H0, indicating statistically significant distribution drift.
    """
    baseline_clean = np.asarray(baseline, dtype=float)
    target_clean = np.asarray(target, dtype=float)
    
    baseline_clean = baseline_clean[~np.isnan(baseline_clean)]
    target_clean = target_clean[~np.isnan(target_clean)]
    
    res = stats.ks_2samp(baseline_clean, target_clean)
    statistic = float(np.round(res.statistic, 4))
    p_value = float(np.round(res.pvalue, 6))
    
    status = "REJECT_H0_DRIFT" if p_value < 0.05 else "ACCEPT_H0_STABLE"
    return statistic, p_value, status

def calculate_chi2_test(baseline: np.ndarray, target: np.ndarray) -> Tuple[float, float, str]:
    """
    Performs the Chi-Square test of homogeneity for binary or discrete categorical features.
    """
    base_series = pd.Series(baseline).dropna()
    target_series = pd.Series(target).dropna()
    
    all_categories = sorted(list(set(base_series.unique()).union(set(target_series.unique()))))
    base_counts = base_series.value_counts().reindex(all_categories, fill_value=0).values
    target_counts = target_series.value_counts().reindex(all_categories, fill_value=0).values
    
    contingency = np.array([base_counts, target_counts])
    res = stats.chi2_contingency(contingency)
    
    statistic = float(np.round(res.statistic, 4))
    p_value = float(np.round(res.pvalue, 6))
    status = "REJECT_H0_DRIFT" if p_value < 0.05 else "ACCEPT_H0_STABLE"
    return statistic, p_value, status

# -------------------------------------------------------------------
# Cohort & Dataset Drift Evaluation Pipeline
# -------------------------------------------------------------------
def load_baseline_dataset(data_path: str = "data/synthetic_dynamic_pricing.csv", sample_size: int = 10000) -> pd.DataFrame:
    """Loads baseline training feature data for statistical comparison."""
    if not os.path.exists(data_path):
        # Generate synthetic fallback baseline
        np.random.seed(42)
        n = sample_size
        return pd.DataFrame({
            'age': np.random.normal(45, 14, n).clip(18, 85),
            'has_redeemed': np.random.binomial(1, 0.35, n),
            'issue_redeem_delay': np.random.exponential(12, n).clip(0, 90)
        })
        
    df = pd.read_csv(data_path)
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median()).clip(18, 85)
    if 'issue_redeem_delay' not in df.columns:
        np.random.seed(42)
        df['issue_redeem_delay'] = np.random.exponential(12, len(df)).clip(0, 90)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    return df

def generate_production_cohort(
    scenario: str = "stable",
    n: int = 2500,
    random_state: int = 101
) -> pd.DataFrame:
    """
    Generates synthetic incoming production batches to test monitoring & alerts:
      - 'stable': Follows baseline demographics (expected normal operation).
      - 'moderate': Mild aging shift (mean age 45 -> 53) and lower redemption rates.
      - 'severe': Major covariate shift (young student influx: age ~23, high delay).
    """
    np.random.seed(random_state)
    
    if scenario == "stable":
        age = np.random.normal(45, 14, n).clip(18, 85)
        has_redeemed = np.random.binomial(1, 0.35, n)
        delay = np.random.exponential(12, n).clip(0, 90)
    elif scenario == "moderate":
        age = np.random.normal(53, 13, n).clip(18, 85)
        has_redeemed = np.random.binomial(1, 0.22, n)
        delay = np.random.exponential(18, n).clip(0, 90)
    elif scenario == "severe":
        age = np.random.normal(23, 4, n).clip(18, 40)
        has_redeemed = np.random.binomial(1, 0.65, n)
        delay = np.random.exponential(35, n).clip(0, 90)
    else:
        raise ValueError(f"Unknown drift scenario: {scenario}")
        
    return pd.DataFrame({
        'client_id': [f"prod_client_{i:05d}" for i in range(n)],
        'age': age,
        'has_redeemed': has_redeemed,
        'issue_redeem_delay': delay
    })

def evaluate_cohort_drift(
    baseline_df: pd.DataFrame,
    production_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs an exhaustive statistical drift evaluation across all features.
    Computes KS-test, PSI, and overall system drift health verdict.
    """
    if features is None:
        features = ['age', 'has_redeemed', 'issue_redeem_delay']
        
    feature_reports = {}
    psi_scores = []
    severe_drift_count = 0
    moderate_drift_count = 0
    
    for feat in features:
        if feat not in baseline_df.columns or feat not in production_df.columns:
            continue
            
        base_arr = baseline_df[feat].values
        prod_arr = production_df[feat].values
        
        # Calculate PSI
        psi, psi_status = calculate_psi(base_arr, prod_arr, num_buckets=10)
        psi_scores.append(psi)
        
        # Calculate KS Test
        ks_stat, ks_pval, ks_status = calculate_ks_test(base_arr, prod_arr)
        
        # Distribution stats
        base_mean = float(np.round(np.mean(base_arr), 2))
        prod_mean = float(np.round(np.mean(prod_arr), 2))
        base_std = float(np.round(np.std(base_arr), 2))
        prod_std = float(np.round(np.std(prod_arr), 2))
        
        if psi_status == "SEVERE_DRIFT":
            severe_drift_count += 1
        elif psi_status == "MODERATE_DRIFT":
            moderate_drift_count += 1
            
        feature_reports[feat] = {
            "psi_score": psi,
            "psi_status": psi_status,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_pval,
            "ks_verdict": ks_status,
            "baseline_mean": base_mean,
            "baseline_std": base_std,
            "production_mean": prod_mean,
            "production_std": prod_std,
            "mean_shift_pct": round(((prod_mean - base_mean) / max(1e-4, base_mean)) * 100, 2)
        }
        
    overall_psi = float(np.round(np.mean(psi_scores), 4)) if psi_scores else 0.0
    
    # Overall System Verdict
    if severe_drift_count > 0 or overall_psi >= 0.25:
        overall_status = "ALERT_SEVERE_DRIFT"
        recommendation = "CRITICAL: Significant covariate shift detected. Causal overlap and unconfoundedness assumptions may be compromised. Retrain Double ML models with recent customer data."
        badge = "FAILED / DRIFT DETECTED"
    elif moderate_drift_count > 0 or overall_psi >= 0.10:
        overall_status = "WARNING_MODERATE_DRIFT"
        recommendation = "WARNING: Moderate distribution shift detected. Increase sampling frequency and monitor conversion rate elasticity."
        badge = "WARNING"
    else:
        overall_status = "HEALTHY_STABLE"
        recommendation = "STABLE: Customer demographics match training distribution baseline. Causal identification remains robust."
        badge = "PASS"
        
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "badge": badge,
        "overall_psi": overall_psi,
        "total_features_evaluated": len(feature_reports),
        "features_with_severe_drift": severe_drift_count,
        "features_with_moderate_drift": moderate_drift_count,
        "recommendation": recommendation,
        "feature_metrics": feature_reports,
        "baseline_sample_size": len(baseline_df),
        "production_sample_size": len(production_df)
    }
    
    return report

def export_drift_report(report: Dict[str, Any], output_path: str = "data/drift_report.json"):
    """Saves the drift report JSON to disk for telemetry and dashboard ingestion."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Drift Detector] Saved audit report to {output_path}")

def evaluate_drift_pipeline(scenario: str = "stable") -> Dict[str, Any]:
    """Helper entry point for API and dashboard callers."""
    base_df = load_baseline_dataset()
    prod_df = generate_production_cohort(scenario=scenario)
    report = evaluate_cohort_drift(base_df, prod_df)
    export_drift_report(report)
    return report

# -------------------------------------------------------------------
# CLI Execution & Unit Demonstration
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("ECONOCAUSAL: COVARIATE SHIFT & DATA DRIFT DETECTION ENGINE")
    print("=" * 70)
    
    base_data = load_baseline_dataset(sample_size=5000)
    print(f"Loaded baseline dataset: {len(base_data):,} records")
    print(f"Baseline Age Mean: {base_data['age'].mean():.2f} (std: {base_data['age'].std():.2f})")
    
    # 1. Test Stable Cohort
    print("\n--- TEST 1: STABLE PRODUCTION COHORT ---")
    stable_cohort = generate_production_cohort(scenario="stable", n=2500)
    rep_stable = evaluate_cohort_drift(base_data, stable_cohort)
    print(f"Overall Status: {rep_stable['badge']} (Mean PSI: {rep_stable['overall_psi']})")
    for feat, m in rep_stable['feature_metrics'].items():
        print(f"  {feat:<20} | PSI: {m['psi_score']:<7} [{m['psi_status']:<14}] | KS p-val: {m['ks_p_value']}")
        
    # 2. Test Severe Drift Cohort
    print("\n--- TEST 2: SEVERE DRIFT COHORT (YOUNG INFLUX) ---")
    severe_cohort = generate_production_cohort(scenario="severe", n=2500)
    rep_severe = evaluate_cohort_drift(base_data, severe_cohort)
    print(f"Overall Status: {rep_severe['badge']} (Mean PSI: {rep_severe['overall_psi']})")
    for feat, m in rep_severe['feature_metrics'].items():
        print(f"  {feat:<20} | PSI: {m['psi_score']:<7} [{m['psi_status']:<14}] | KS p-val: {m['ks_p_value']}")
    print(f"\nRecommendation:\n{rep_severe['recommendation']}")
    
    # Export baseline stable report
    export_drift_report(rep_stable, "data/drift_report.json")
