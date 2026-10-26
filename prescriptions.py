import os
import pandas as pd
from optimizer import solve_prescriptive_greedy, solve_prescriptive_milp

def generate_customer_prescriptions(
    predictions_path='data/ite_predictions.csv',
    budget=5000.0,
    cost_10=10.0,
    cost_20=20.0,
    output_path='data/prescriptive_allocation_matrix.csv',
    use_milp=False
):
    """
    Generates actionable customer prescriptions assigning optimal discount tiers ($0, $10, $20)
    to maximize incremental conversions under a strict marketing budget constraint.
    """
    print(f"Loading customer ITE scores from {predictions_path}...")
    df = pd.read_csv(predictions_path)
    
    print(f"Optimizing allocation for ${budget:,.2f} budget (Cost $10: ${cost_10}, Cost $20: ${cost_20})...")
    if use_milp:
        prescribed_df = solve_prescriptive_milp(df, budget=budget, cost_10=cost_10, cost_20=cost_20)
    else:
        prescribed_df = solve_prescriptive_greedy(df, budget=budget, cost_10=cost_10, cost_20=cost_20)
        
    # Format and select columns
    columns_to_export = [
        'client_id',
        'age',
        'age_bracket',
        'has_redeemed',
        'recommended_discount',
        'expected_lift',
        'treatment_cost'
    ]
    
    # Ensure columns exist
    export_df = prescribed_df[[col for col in columns_to_export if col in prescribed_df.columns]].copy()
    export_df['expected_lift_pct'] = (export_df['expected_lift'] * 100).round(2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_df.to_csv(output_path, index=False)
    
    counts = export_df['recommended_discount'].value_counts().sort_index()
    print("\nPrescription Breakdown:")
    for tier, count in counts.items():
        print(f"  Tier ${tier}: {count:,} customers ({count/len(export_df)*100:.1f}%)")
        
    print(f"\nTotal Budget Spent: ${export_df['treatment_cost'].sum():,.2f}")
    print(f"Total Expected Incremental Buyers: {export_df['expected_lift'].sum():.1f}")
    print(f"Successfully saved customer prescriptions to {output_path}")
    
    return export_df

if __name__ == '__main__':
    generate_customer_prescriptions()
