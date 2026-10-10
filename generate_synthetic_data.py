import pandas as pd
import numpy as np

def generate_synthetic_data():
    print("Loading data...")
    np.random.seed(42)

    # Load data
    df_clients = pd.read_csv('data/clients.csv', index_col='client_id')
    df_train = pd.read_csv('data/uplift_train.csv', index_col='client_id')

    # Merge to get age
    df = df_train.join(df_clients[['age', 'gender']])

    # Default treatment is 0
    df['discount_amount'] = 0

    # For treated users (treatment_flg == 1), split into $10 and $20
    treated_mask = df['treatment_flg'] == 1
    num_treated = treated_mask.sum()

    # Randomly assign 10 or 20 to treated users
    discount_assignments = np.random.choice([10, 20], size=num_treated, p=[0.5, 0.5])
    df.loc[treated_mask, 'discount_amount'] = discount_assignments

    # Inject Heterogeneous Treatment Effects (HTE) on the 'target'
    # Base rule: We assume the original 'target' represents the baseline behavior + whatever uplift the original binary treatment had.
    # We will synthetically boost the target for the $20 group based on age to prove our causal model works later.
    
    # Rule: Users under 30 are price sensitive. The $20 discount gives them a big boost.
    def calculate_boost(row):
        if row['discount_amount'] == 20:
            if row['age'] < 30:
                return 0.15  # 15% absolute boost in conversion probability for young users
            elif row['age'] > 50:
                return 0.02  # 2% boost for older users
            else:
                return 0.05  # 5% boost for middle age
        return 0.0 # No synthetic boost for $0 or $10

    print("Injecting Heterogeneous Treatment Effects (HTE)...")
    df['boost_prob'] = df.apply(calculate_boost, axis=1)

    # Apply the boost: if target is 0, randomly flip to 1 based on boost_prob
    random_draw = np.random.rand(len(df))
    flip_mask = (df['target'] == 0) & (random_draw < df['boost_prob'])
    df.loc[flip_mask, 'target'] = 1

    print(f"Original positive targets: {df_train['target'].sum()}")
    print(f"New positive targets: {df['target'].sum()}")
    print(f"Total rows flipped to buyers due to $20 discount: {flip_mask.sum()}")

    # Drop the temporary boost column and original treatment_flg
    df = df.drop(columns=['boost_prob', 'treatment_flg'])

    # Save to new CSV
    output_file = 'data/synthetic_dynamic_pricing.csv'
    df.to_csv(output_file)
    print(f"\nSuccess! Saved synthetic dataset with $10 vs $20 treatments to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data()
