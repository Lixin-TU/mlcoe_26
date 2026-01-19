
import sys
import os

# Add the agent codebase to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_analysis_agent-main')))

from config.llm_config import LLMConfig
from utils.llm_helper import LLMHelper

def main():
    # 1. Read the extracted tables
    tex_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../2.5/extracted_tables.tex'))
    
    try:
        with open(tex_path, 'r', encoding='utf-8') as f:
            tex_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {tex_path}")
        return

    # 2. Construct the Prompt
    prompt = f"""
    You are a financial analyst. Based ONLY on the provided LaTeX tables extracted from the General Motors 2023 Annual Report, answer the following questions.
    
    EXTRACTED DATA:
    ```latex
    {tex_content}
    ```
    
    QUESTIONS:
    f) Given this data, extract the income statement and balance sheet figures to answer:
    i. What’s the net income of the current year (2023)?
    ii. What’s the cost-to-income ratio? (Total Costs & Expenses / Total Net Sales & Revenue)
    iii. Calculate the following ratios for 2023:
        - Quick Ratio ( (Current Assets - Inventories) / Current Liabilities )
        - Debt-to-Equity Ratio ( Total Debt / Total Equity ) 
          *Note: Total Debt = Short-term debt + Long-term debt (Sum of Automotive and GM Financial)*
        - Debt to Assets Ratio ( Total Debt / Total Assets )
        - Debt-to-Capital Ratio ( Total Debt / (Total Debt + Total Equity) )
        - Debt-to-EBITDA Ratio ( Total Debt / EBITDA )
          *Note: Approximated EBITDA = Operating Income + Depreciation & Amortization (Use 4904 million from Note 24/Cash Flow if not in table, otherwise assume 4904 or calculate if D&A is visible. If D&A is not visible, state that limitation or use Operating Income as proxy).*
          
    iv. What’s the interest coverage ratio? (Operating Income / Interest Expense)
       *Note: Use 'Automotive interest expense' for the denominator as GM Financial interest is often operating.*

    Please provide the calculation steps and the final values. Output in English.
    """

    # 3. Initialize LLM
    # Note: Environment variables for keys must be loaded or set in LLMConfig
    config = LLMConfig()
    llm = LLMHelper(config)

    print("Sending request to LLM (this may take a moment)...")
    
    # 4. Call LLM
    # Use sync call for simplicity in script
    response = llm.call(prompt=prompt, temperature=0.0)
    
    print("\n--- LLM ANALYSIS RESULT ---\n")
    print(response)

if __name__ == "__main__":
    main()
