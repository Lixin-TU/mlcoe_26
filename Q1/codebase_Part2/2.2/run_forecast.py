import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 1. Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_root = os.path.abspath(os.path.join(current_dir, '../data_analysis_agent-main'))
sys.path.append(agent_root)

# 2. Load .env explicitly (Custom parser for YAML-like .env)
env_path = os.path.join(agent_root, '.env')
if os.path.exists(env_path):
    print(f"Loading .env from {env_path}")
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Handle both KEY=VALUE and KEY: VALUE
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
            print(f"Set {key}")
else:
    print(f"Warning: .env not found at {env_path}")

try:
    from data_analysis_agent import DataAnalysisAgent
except ImportError as e:
    print(f"Error importing DataAnalysisAgent: {e}")
    sys.path.insert(0, agent_root) 
    from data_analysis_agent import DataAnalysisAgent

# 3. Generate Data (Replicating Q1_6.py logic)
print("Generating dataset...")
FIRMS = ['Tencent', 'JPMorgan', 'Alibaba_HK', 'Exxon', 'Volkswagen', 'Microsoft', 'Google']
ASSET_COMPONENTS = ['Cash_Equiv', 'Acct_Rec', 'Net_PPE', 'Avail_Sale_Sec', 'Other_Assets']
LIAB_COMPONENTS = ['Curr_Debt', 'Long_Term_Debt', 'Payables', 'Def_Tax_Liab', 'Other_Liab']
EQUITY_COMPONENTS = ['Capital_Stock', 'Common_Stock', 'Ret_Earnings', 'Other_Equity']
BOTTOM_LEVEL_NAMES = ASSET_COMPONENTS + LIAB_COMPONENTS + EQUITY_COMPONENTS

n_assets = len(ASSET_COMPONENTS)
n_liabs = len(LIAB_COMPONENTS)
n_equity = len(EQUITY_COMPONENTS)
n_firms = len(FIRMS)

def generate_balanced_data():
    np.random.seed(42)
    # Shape: (n_firms, 5 years, n_liabs + n_equity)
    L_E_data = np.random.uniform(10, 50, size=(n_firms, 5, n_liabs + n_equity))
    
    Total_LE = np.sum(L_E_data, axis=2)
    
    raw_asset_weights = np.random.uniform(0.5, 1.5, size=(n_firms, 5, n_assets))
    weight_sums = np.sum(raw_asset_weights, axis=2, keepdims=True)
    normalized_weights = raw_asset_weights / weight_sums
    
    A_data = normalized_weights * Total_LE[:, :, np.newaxis]
    
    data = np.concatenate([A_data, L_E_data], axis=2)
    return data.astype(np.float32)

raw_data = generate_balanced_data()

# 4. Save to CSV
csv_filename = 'balance_sheet_data_Q2_2.csv'
csv_path = os.path.join(current_dir, csv_filename)

records = []
for f_idx, firm in enumerate(FIRMS):
    for year in range(5):
        row = {'Firm': firm, 'Year': year}
        for c_idx, component in enumerate(BOTTOM_LEVEL_NAMES):
            row[component] = raw_data[f_idx, year, c_idx]
        records.append(row)

df = pd.DataFrame(records)
df.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}")

# 5. Run DataAnalysisAgent
print("Initializing DataAnalysisAgent...")
agent = DataAnalysisAgent(output_dir=current_dir)

query = f"""
I have prepared a dataset '{csv_filename}' containing balance sheet data for {n_firms} firms over 5 years (Years 0 to 4).
The columns include Firm, Year, and various Asset, Liability, and Equity components.
Note: Assets should equal Liabilities + Equity.

Task:
1. Load the dataset.
2. Split the data into a training set (Years 0, 1, 2 => predict Years 1, 2, 3) and a test set (Year 3 => predict Year 4).
   Basically, we want to learn to predict the *next year's* balance sheet components given the *current year's* components.
3. Train a machine learning model (e.g. Linear Regression) on the training pairs.
4. Predict the balance sheet values for Year 4 (using Year 3 data) and calculate the Mean Squared Error (MSE) overall against the actual Year 4 data.
5. Report the MSE.
6. Finally, use the model to forecast the components for a hypothetical "Year 5" using the data from Year 4.
7. Print the Year 5 forecast for "Retained Earnings".

Please execute the code and analyze the results. 
"""

print("Running analysis...")
result = agent.analyze(query, files=[csv_path])

print("\n=== Analysis Result ===")
final_report = result.get('final_report')
print(final_report)
