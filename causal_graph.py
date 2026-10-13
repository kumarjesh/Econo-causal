import pandas as pd
from dowhy import CausalModel

def generate_causal_graph():
    print("Loading synthetic data...")
    # Load our new synthetic dataset
    df = pd.read_csv('data/synthetic_dynamic_pricing.csv')
    
    # We define the causal variables based on our knowledge of the data
    # Treatment: discount_amount (0, 10, or 20)
    # Outcome: target (purchase or not)
    # Confounders (Common Causes): age, gender
    
    print("Building DoWhy Causal Model...")
    model = CausalModel(
        data=df,
        treatment='discount_amount',
        outcome='target',
        common_causes=['age', 'gender'] 
    )
    
    print("Generating Causal Graph...")
    # Visualize the DAG
    model.view_model(layout="dot")
    
    print("Causal graph generated as causal_model.png")

if __name__ == "__main__":
    generate_causal_graph()
