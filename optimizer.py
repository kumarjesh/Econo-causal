import os
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix, eye, hstack, vstack

def solve_prescriptive_milp(df, budget=5000.0, cost_10=10.0, cost_20=20.0):
    """
    Exact Binary Mixed-Integer Linear Program (MILP) solver using SciPy HiGHS.
    
    Decision variables:
      x_10[i] in {0, 1}: Assign $10 discount to customer i
      x_20[i] in {0, 1}: Assign $20 discount to customer i
      (If both are 0, customer receives $0 discount)
      
    Constraints:
      1. Budget constraint: sum(cost_10 * x_10 + cost_20 * x_20) <= budget
      2. Exclusivity constraint: x_10[i] + x_20[i] <= 1 for each customer i
    """
    n = len(df)
    tau_10 = df['ite_10'].values
    tau_20 = df['ite_20'].values
    
    # Objective: maximize sum(tau_10 * x_10 + tau_20 * x_20)
    # SciPy minimizes, so we negate coefficients
    c = - np.concatenate([tau_10, tau_20])
    
    # Budget constraint: 1 row
    row_budget = np.concatenate([np.full(n, cost_10), np.full(n, cost_20)])
    A_budget = csc_matrix(row_budget.reshape(1, 2 * n))
    
    # Exclusivity constraint: n rows
    A_excl = hstack([eye(n, format='csc'), eye(n, format='csc')])
    
    A = vstack([A_budget, A_excl])
    
    lhs = np.zeros(1 + n)
    rhs = np.concatenate([[budget], np.ones(n)])
    constraints = LinearConstraint(A, lhs, rhs)
    
    integrality = np.ones(2 * n) # All binary integer variables
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        print("MILP failed or timed out. Falling back to greedy marginal ROI solver.")
        return solve_prescriptive_greedy(df, budget, cost_10, cost_20)
        
    x = res.x
    x_10 = x[:n].round().astype(int)
    x_20 = x[n:].round().astype(int)
    
    recommended_discount = np.zeros(n, dtype=int)
    recommended_discount[x_10 == 1] = 10
    recommended_discount[x_20 == 1] = 20
    
    expected_lift = np.zeros(n)
    expected_lift[x_10 == 1] = tau_10[x_10 == 1]
    expected_lift[x_20 == 1] = tau_20[x_20 == 1]
    
    cost = np.zeros(n)
    cost[x_10 == 1] = cost_10
    cost[x_20 == 1] = cost_20
    
    res_df = df.copy()
    res_df['recommended_discount'] = recommended_discount
    res_df['expected_lift'] = expected_lift
    res_df['treatment_cost'] = cost
    
    return res_df

def solve_prescriptive_greedy(df, budget=5000.0, cost_10=10.0, cost_20=20.0):
    """
    Ultra-fast Marginal-ROI greedy solver (O(N log N)).
    Evaluates incremental lift per dollar spent:
      - Step 1: $0 -> $10: Marginal ROI = tau_10 / cost_10
      - Step 2: $10 -> $20: Marginal ROI = (tau_20 - tau_10) / (cost_20 - cost_10)
    """
    n = len(df)
    tau_10 = df['ite_10'].values
    tau_20 = df['ite_20'].values
    
    roi_step1 = tau_10 / cost_10
    cost_diff = max(1.0, cost_20 - cost_10)
    roi_step2 = (tau_20 - tau_10) / cost_diff
    
    # Actions: (type, customer_idx, marginal_cost, marginal_lift, roi)
    actions = []
    for i in range(n):
        if roi_step1[i] > 0:
            actions.append(('10', i, cost_10, tau_10[i], roi_step1[i]))
        if roi_step2[i] > 0:
            actions.append(('upgrade_20', i, cost_diff, tau_20[i] - tau_10[i], roi_step2[i]))
            
    # Sort descending by marginal ROI
    actions.sort(key=lambda a: a[4], reverse=True)
    
    recommended_discount = np.zeros(n, dtype=int)
    remaining_budget = budget
    
    for act_type, idx, c_cost, c_lift, roi in actions:
        if act_type == '10':
            if recommended_discount[idx] == 0 and remaining_budget >= c_cost:
                recommended_discount[idx] = 10
                remaining_budget -= c_cost
        elif act_type == 'upgrade_20':
            if recommended_discount[idx] == 10 and remaining_budget >= c_cost:
                recommended_discount[idx] = 20
                remaining_budget -= c_cost
                
    expected_lift = np.zeros(n)
    cost = np.zeros(n)
    
    mask_10 = recommended_discount == 10
    mask_20 = recommended_discount == 20
    
    expected_lift[mask_10] = tau_10[mask_10]
    expected_lift[mask_20] = tau_20[mask_20]
    cost[mask_10] = cost_10
    cost[mask_20] = cost_20
    
    res_df = df.copy()
    res_df['recommended_discount'] = recommended_discount
    res_df['expected_lift'] = expected_lift
    res_df['treatment_cost'] = cost
    
    return res_df

def compare_policies(df, budget=5000.0, cost_10=10.0, cost_20=20.0):
    """
    Benchmarks 4 competing marketing strategies under the exact same budget constraint:
      1. Causal Prescriptive Policy (Optimized via SciPy)
      2. Blanket $20 Targeting (gives $20 discounts indiscriminately until budget is exhausted)
      3. Blanket $10 Targeting (gives $10 discounts indiscriminately until budget is exhausted)
      4. Traditional Predictive Targeting (targets customers with high baseline purchase probability)
      5. Random Targeting (un-targeted control)
    """
    n = len(df)
    tau_10 = df['ite_10'].values
    tau_20 = df['ite_20'].values
    
    # 1. Causal Prescriptive Policy
    opt_df = solve_prescriptive_greedy(df, budget, cost_10, cost_20)
    causal_buyers = opt_df['expected_lift'].sum()
    causal_cost = opt_df['treatment_cost'].sum()
    
    # 2. Blanket $20 Targeting
    max_users_20 = min(n, int(budget // cost_20))
    blanket_20_buyers = tau_20[:max_users_20].sum()
    blanket_20_cost = max_users_20 * cost_20
    
    # 3. Blanket $10 Targeting
    max_users_10 = min(n, int(budget // cost_10))
    blanket_10_buyers = tau_10[:max_users_10].sum()
    blanket_10_cost = max_users_10 * cost_10
    
    # 4. Traditional Predictive Targeting (targets older / loyal customers who buy anyway)
    # Sort by baseline propensity (age and previous redemption)
    baseline_score = 0.5 * (df['age'].values / 60.0) + 0.5 * df['has_redeemed'].values
    pred_order = np.argsort(-baseline_score)
    pred_users = pred_order[:max_users_20]
    predictive_buyers = tau_20[pred_users].sum()
    predictive_cost = max_users_20 * cost_20
    
    # 5. Random Rollout (average expected rate)
    random_buyers = (tau_20.mean()) * max_users_20
    random_cost = max_users_20 * cost_20
    
    comparison = pd.DataFrame({
        'Policy Strategy': [
            'Causal Prescriptive Optimization',
            'Blanket $20 Targeting',
            'Blanket $10 Targeting',
            'Traditional Predictive Targeting',
            'Random Rollout'
        ],
        'Budget Spent ($)': [
            causal_cost,
            blanket_20_cost,
            blanket_10_cost,
            predictive_cost,
            random_cost
        ],
        'Incremental Buyers Gained': [
            causal_buyers,
            blanket_20_buyers,
            blanket_10_buyers,
            predictive_buyers,
            random_buyers
        ],
        'Cost per Incremental Buyer ($)': [
            causal_cost / max(1e-5, causal_buyers),
            blanket_20_cost / max(1e-5, blanket_20_buyers),
            blanket_10_cost / max(1e-5, blanket_10_buyers),
            predictive_cost / max(1e-5, predictive_buyers),
            random_cost / max(1e-5, random_buyers)
        ]
    })
    
    # Advantage calculations
    causal_lift_vs_blanket = ((causal_buyers - blanket_20_buyers) / blanket_20_buyers) * 100
    causal_lift_vs_pred = ((causal_buyers - predictive_buyers) / predictive_buyers) * 100
    
    comparison['Efficiency Lift vs Baseline (%)'] = [
        f"+{causal_lift_vs_blanket:.1f}% vs Blanket",
        "Benchmark",
        f"{((blanket_10_buyers - blanket_20_buyers)/blanket_20_buyers)*100:+.1f}%",
        f"{((predictive_buyers - blanket_20_buyers)/blanket_20_buyers)*100:+.1f}%",
        f"{((random_buyers - blanket_20_buyers)/blanket_20_buyers)*100:+.1f}%"
    ]
    
    return comparison, opt_df

if __name__ == "__main__":
    print("=" * 70)
    print("ECONOCAUSAL: PRESCRIPTIVE BUDGET OPTIMIZATION SOLVER")
    print("=" * 70)
    
    df_preds = pd.read_csv('data/ite_predictions.csv')
    test_budget = 5000.0
    print(f"Solving optimization for marketing budget = ${test_budget:,.2f}...")
    
    # Solve MILP
    opt_df = solve_prescriptive_milp(df_preds, budget=test_budget)
    
    # Counts
    counts = opt_df['recommended_discount'].value_counts().sort_index()
    print("\nOptimal Discount Allocation Matrix:")
    for tier, cnt in counts.items():
        print(f"  Tier ${tier} Discount: {cnt:,} customers ({cnt/len(opt_df)*100:.1f}%)")
        
    print(f"\nTotal Budget Utilized: ${opt_df['treatment_cost'].sum():,.2f}")
    print(f"Total Expected Incremental Buyers: {opt_df['expected_lift'].sum():.1f} buyers")
    print(f"Cost per Incremental Buyer: ${opt_df['treatment_cost'].sum()/opt_df['expected_lift'].sum():.2f}")
    
    # Save prescriptive matrix
    os.makedirs('data', exist_ok=True)
    out_path = 'data/prescriptive_allocation_matrix.csv'
    opt_df[['client_id', 'age', 'has_redeemed', 'recommended_discount', 'expected_lift', 'treatment_cost']].to_csv(out_path, index=False)
    print(f"\nSaved actionable prescriptions to {out_path}")
    
    # Run Benchmark Comparison
    print("\n" + "=" * 70)
    print("POLICY BENCHMARK COMPARISON (SAME $5,000 BUDGET)")
    print("=" * 70)
    comp_df, _ = compare_policies(df_preds, budget=test_budget)
    print(comp_df.to_string(index=False))
