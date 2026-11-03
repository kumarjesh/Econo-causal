import os
import pytest
import numpy as np
import pandas as pd
from api import (
    CustomerPayload,
    BatchPredictionRequest,
    OptimizationRequest,
    read_root,
    health_check,
    predict_individual_customer,
    predict_batch_customers,
    optimize_campaign_allocation
)
from drift_detector import (
    calculate_psi,
    calculate_ks_test,
    calculate_chi2_test,
    generate_production_cohort,
    evaluate_cohort_drift
)

# -------------------------------------------------------------------
# 1. API Core Endpoint Unit Tests
# -------------------------------------------------------------------
def test_root_endpoint():
    """Validates root metadata and endpoint dictionary."""
    res = read_root()
    assert "service" in res
    assert "endpoints" in res
    assert res["version"] == "1.0.0"

def test_health_endpoint():
    """Validates service healthcheck and solver inventory."""
    res = health_check()
    assert res.status == "healthy"
    assert res.uptime_seconds >= 0.0
    assert len(res.active_solvers) >= 2
    assert "HiGHS" in res.active_solvers[0] or "milp" in res.active_solvers[0]

def test_single_customer_prediction():
    """Validates real-time individual ITE estimation and discount logic."""
    payload = CustomerPayload(
        client_id="test_client_001",
        age=24.0,
        has_redeemed=1,
        issue_redeem_delay=5.0
    )
    pred = predict_individual_customer(payload)
    
    assert pred.client_id == "test_client_001"
    assert isinstance(pred.ite_10, float)
    assert isinstance(pred.ite_20, float)
    assert pred.recommended_discount in [0, 10, 20]
    assert "%" in pred.expected_lift_pct
    assert len(pred.price_elasticity_segment) > 0

def test_batch_customer_prediction():
    """Validates cohort batch inference and aggregation."""
    customers = [
        CustomerPayload(client_id=f"c_{i}", age=float(20 + i * 5), has_redeemed=i % 2, issue_redeem_delay=float(i * 2))
        for i in range(10)
    ]
    req = BatchPredictionRequest(customers=customers)
    res = predict_batch_customers(req)
    
    assert res.total_processed == 10
    assert len(res.predictions) == 10
    assert isinstance(res.mean_ate_10_pct, float)
    assert isinstance(res.mean_ate_20_pct, float)

def test_prescriptive_optimization_budget_constraint():
    """Validates that prescriptive optimization strictly obeys budget."""
    req = OptimizationRequest(
        budget=4000.0,
        cost_10=10.0,
        cost_20=20.0,
        solver="greedy",
        customer_sample_size=1000
    )
    res = optimize_campaign_allocation(req)
    
    # Must never exceed budget
    assert res.budget_allocated <= 4000.0
    assert res.budget_utilization_pct <= 100.0
    assert res.incremental_buyers_gained > 0.0
    assert res.cost_per_incremental_buyer > 0.0
    assert res.efficiency_lift_vs_blanket_pct > 0.0
    assert len(res.policy_benchmarks) >= 3

# -------------------------------------------------------------------
# 2. Covariate Shift & Drift Detector Tests
# -------------------------------------------------------------------
def test_psi_identical_distributions():
    """Validates that identical distributions yield PSI < 0.10 (STABLE)."""
    np.random.seed(42)
    dist_a = np.random.normal(50, 10, 2000)
    dist_b = np.random.normal(50, 10, 2000)
    
    psi, status = calculate_psi(dist_a, dist_b)
    assert psi < 0.10
    assert status == "STABLE"

def test_psi_severe_distribution_shift():
    """Validates that large distribution shift triggers SEVERE_DRIFT."""
    np.random.seed(42)
    dist_baseline = np.random.normal(50, 5, 2000)
    dist_shifted = np.random.normal(25, 3, 2000)
    
    psi, status = calculate_psi(dist_baseline, dist_shifted)
    assert psi >= 0.25
    assert status == "SEVERE_DRIFT"

def test_ks_test_drift_detection():
    """Validates Kolmogorov-Smirnov test sensitivity."""
    np.random.seed(42)
    base = np.random.normal(40, 10, 1500)
    same = np.random.normal(40, 10, 1500)
    diff = np.random.normal(60, 10, 1500)
    
    # Should accept H0 for same
    _, p_val_same, verdict_same = calculate_ks_test(base, same)
    assert p_val_same > 0.01
    
    # Should reject H0 for shifted
    _, p_val_diff, verdict_diff = calculate_ks_test(base, diff)
    assert p_val_diff < 0.001
    assert verdict_diff == "REJECT_H0_DRIFT"

def test_cohort_drift_evaluation_pipeline():
    """Validates end-to-end cohort drift diagnostic engine."""
    np.random.seed(42)
    n = 1000
    base_df = pd.DataFrame({
        'age': np.random.normal(45, 12, n),
        'has_redeemed': np.random.binomial(1, 0.4, n),
        'issue_redeem_delay': np.random.exponential(15, n)
    })
    
    prod_stable = generate_production_cohort(scenario="stable", n=n)
    report_stable = evaluate_cohort_drift(base_df, prod_stable)
    assert report_stable["overall_status"] in ["HEALTHY_STABLE", "WARNING_MODERATE_DRIFT"]
    assert "feature_metrics" in report_stable
    
    prod_severe = generate_production_cohort(scenario="severe", n=n)
    report_severe = evaluate_cohort_drift(base_df, prod_severe)
    assert report_severe["overall_status"] == "ALERT_SEVERE_DRIFT"
    assert report_severe["badge"] == "FAILED / DRIFT DETECTED"
    assert report_severe["features_with_severe_drift"] >= 1

