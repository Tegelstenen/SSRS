# Depreciated 
import subprocess
import pandas as pd
from datetime import date, timedelta
import os

# Gets the daily data
script_args = ["Filip/Py_scripts/get_daily_data.py"]
result = subprocess.run(["python"] + script_args, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error running {script_args[0]}: {result.stderr}")
else:
    print(f"Successfully ran {script_args[0]}")

daily = pd.read_csv("Filip/app/data/daily_data.csv")

# Removes last date from streamlit data
end = date.today() - timedelta(days=1)
start = end - timedelta(days=60)
streamlit_data = pd.read_csv("Filip/app/data/streamlit_data.csv")
streamlit_data = streamlit_data.query(f"date >= '{start}' and date <= '{end}'")
streamlit_data = pd.concat([streamlit_data, daily], ignore_index=True)
streamlit_data = streamlit_data.drop_duplicates()


# Check if the file exists
dir = "Filip/app/data/daily_data.csv"
if os.path.exists(dir):
    # Remove the file
    os.remove(dir)
    print(f"Removed daily data file: {dir}")
else:
    print(f"Daily data file does not exist: {dir}")

streamlit_data.to_csv("Filip/app/data/streamlit_data.csv", index=False)
