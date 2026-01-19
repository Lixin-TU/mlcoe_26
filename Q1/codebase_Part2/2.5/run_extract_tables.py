import sys
import os
import pandas as pd

# 1. Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_root = os.path.abspath(os.path.join(current_dir, '../data_analysis_agent-main'))
sys.path.append(agent_root)

# 2. Load .env explicitly
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

try:
    from data_analysis_agent import DataAnalysisAgent
except ImportError:
    sys.path.insert(0, agent_root) 
    from data_analysis_agent import DataAnalysisAgent

# 3. PDF Path
pdf_path = os.path.join(current_dir, 'GM_Annual_Report.pdf')

# 4. Run Agent
print("Initializing DataAnalysisAgent for PDF Extraction...")
agent = DataAnalysisAgent(output_dir=current_dir)

query = f"""
I have a PDF file at '{pdf_path}' (General Motors Annual Report).
Task: Extract two specific financial tables and convert them into Overleaf-ready LaTeX code.

The tables are likely located around page 56 and 57 (physical page numbers, might differ from PDF page index).
1. **Income Statement** (Look for "Consolidated Income Statements" or similar).
2. **Balance Sheet** (Look for "Consolidated Balance Sheets" or similar).

Steps:
1. Use `pdfplumber` to open the PDF.
2. Find the correct pages for the Income Statement and Balance Sheet. Search for the titles on pages 54-60 to be safe.
3. Extract the tables from those pages.
4. Clean the data (handle headers, empty rows, etc.).
5. Generate LaTeX code (table environment) for each table.
   - Use `tabular` or `longtable`.
   - Ensure numbers are aligned.
   - Include the LaTeX code in your final report.

Please output the LaTeX code clearly.
"""

print("Running extraction...")
result = agent.analyze(query, files=[pdf_path])

print("\n=== Analysis Result ===")
final_report = result.get('final_report')
print(final_report)
