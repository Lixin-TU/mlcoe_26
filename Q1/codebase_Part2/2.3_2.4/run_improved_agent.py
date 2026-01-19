import sys
import os
import pandas as pd
from dotenv import load_dotenv

# 1. Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_root = os.path.abspath(os.path.join(current_dir, '../data_analysis_agent-main'))
sys.path.append(agent_root)

# 2. Load .env explicitly (Custom parser)
env_path = os.path.join(agent_root, '.env')
if os.path.exists(env_path):
    print(f"Loading .env from {env_path}")
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
else:
    print(f"Warning: .env not found at {env_path}")

try:
    from data_analysis_agent import DataAnalysisAgent
except ImportError:
    sys.path.insert(0, agent_root) 
    from data_analysis_agent import DataAnalysisAgent

# 3. Data Path
csv_path = os.path.abspath(os.path.join(current_dir, '../2.2/balance_sheet_data_Q2_2.csv'))

# 4. Run DataAnalysisAgent with Improved Prompt
print("Initializing DataAnalysisAgent for Improved Forecast...")
agent = DataAnalysisAgent(output_dir=current_dir)

query = f"""
I have a dataset at '{csv_path}' containing balance sheet data for 7 firms over 5 years (Years 0-4).

Task: Improve the balance sheet forecast accuracy.
Previous attempts with Linear Regression yielded a high MSE (~920).
The baseline to beat is an MSE of ~312 (achieved by a Neural Network).

Please perform the following analysis steps:
1. Load the dataset.
2. Create Training Data (Years 0->1, 1->2, 2->3) and Test Data (Year 3->4).
   Feature columns are all columns except 'Firm' and 'Year'.
3. Train two advanced models:
   a. Random Forest Regressor (sklearn.ensemble.RandomForestRegressor)
   b. Gradient Boosting Regressor (sklearn.ensemble.GradientBoostingRegressor) or MLPRegressor
   *Important*: You are now allowed to import sklearn submodules (e.g. from sklearn.ensemble import ...).
4. Evaluate both on the Test Data (Test MSE).
5. Pick the best performing model.
6. Forecast Year 5 using Year 4 data with the best model.
7. Post-process the Year 5 forecast specifically to enforce the accounting identity:
   Assets (first 5 cols) = Liabilities (next 5 cols) + Equity (last 4 cols).
   (Adjust the components proportionally if they don't match, or just check the gap).
8. Print the Year 5 prediction for 'Retained Earnings' and the Final Test MSE.

Constraint:
- Assets columns: 'Cash_Equiv', 'Acct_Rec', 'Net_PPE', 'Avail_Sale_Sec', 'Other_Assets'
- Liabilities columns: 'Curr_Debt', 'Long_Term_Debt', 'Payables', 'Def_Tax_Liab', 'Other_Liab'
- Equity columns: 'Capital_Stock', 'Common_Stock', 'Ret_Earnings', 'Other_Equity'

Save a plot comparing the Best Model's predictions vs Actuals for Year 4.
"""

print("Running improved analysis...")
result = agent.analyze(query, files=[csv_path])

print("\n=== Analysis Result ===")
final_report = result.get('final_report')
print(final_report)
