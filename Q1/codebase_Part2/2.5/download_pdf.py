import requests
import os

url = "https://investor.gm.com/static-files/1fff6f59-551f-4fe0-bca9-74bfc9a56aeb"
output_path = r"c:\Users\VVI\Desktop\Lixin\github\mlcoe_26\Q1\codebase_Part2\2.5\GM_Annual_Report.pdf"

print(f"Downloading {url} to {output_path}...")
try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
except Exception as e:
    print(f"Error downloading file: {e}")
