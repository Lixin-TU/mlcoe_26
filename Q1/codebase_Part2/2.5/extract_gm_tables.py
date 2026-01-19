import pdfplumber
import pandas as pd
import re

pdf_path = r'c:\Users\VVI\Desktop\Lixin\github\mlcoe_26\Q1\codebase_Part2\2.5\2023 General Motors Annual Report .pdf'

def clean_money(text):
    if not isinstance(text, str):
        return text
    # Remove parentheses used for negative numbers and add minus sign
    if '(' in text and ')' in text:
        text = text.replace('(', '-').replace(')', '')
    # Remove commas
    text = text.replace(',', '')
    text = text.replace('$', '')
    return text

def to_latex(df, caption):
    # LaTeX header
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    
    # Column alignment: l for first column, r for others
    cols_align = "l" + "r" * (len(df.columns) - 1)
    latex += f"\\begin{{tabular}}{{{cols_align}}}\n"
    latex += "\\toprule\n"
    
    # Header row
    # Clean header columns
    headers = [str(col).replace('\n', ' ') for col in df.columns]
    
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for index, row in df.iterrows():
        # Clean data for latex (escape special chars if needed, usually % or &)
        row_data = []
        for item in row:
            s = str(item) if item is not None else ""
            s = s.replace('&', '\\&').replace('%', '\\%')
            row_data.append(s)
            
        latex += " & ".join(row_data) + " \\\\\n"
        
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    return latex

def extract_gm_tables():
    print(f"Opening PDF: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        
        # We identified the pages manually via diagnostic run:
        # Income Statement: Page Index 60
        # Balance Sheet: Page Index 61
        
        page_indices = {
            "Income Statement": 60,
            "Balance Sheet": 61
        }
        
        table_settings = {
            "vertical_strategy": "text", 
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
        }

        for name, idx in page_indices.items():
            print(f"\n--- Extracting {name} from Page Index {idx} ---")
            page = pdf.pages[idx]
            tables = page.extract_tables(table_settings)
            
            if not tables:
                print(f"No tables found on page {idx}")
                continue

            # Identify the correct table (usually the largest one)
            # Filter out tiny tables (headers often get parsed as separate small tables)
            main_table = None
            max_len = 0
            for t in tables:
                if len(t) > max_len:
                    max_len = len(t)
                    main_table = t
            
            if main_table:
                df = pd.DataFrame(main_table[1:], columns=main_table[0])
                
                # Basic Cleaning
                # 1. Remove columns that are all None/Empty string (often spacers)
                df = df.dropna(axis=1, how='all')
                
                # 2. Fix the header
                # The extracted header might be messy. The first row of data often contains the years.
                # Let's clean the column names.
                df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
                
                # Output LaTeX
                print(f"\nLATEX OUTPUT FOR {name.upper()}:\n")
                print(to_latex(df, f"General Motors {name}"))
            else:
                print(f"No significant table found on page {idx}")

if __name__ == "__main__":
    extract_gm_tables()
