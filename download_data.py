import importlib
import pandas as pd

# Load the optional dependency dynamically so static analyzers do not flag a
# missing local installation. Install it with: python -m pip install kagglehub
kagglehub = importlib.import_module("kagglehub")

# Downloads and caches the dataset locally automatically
path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")

# Load directly into pandas
df = pd.read_csv(f"{path}/twcs/twcs.csv")