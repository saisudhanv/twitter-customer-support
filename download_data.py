"""
Simple script to download the customer support dataset.

This is for manual testing. The main pipeline uses scripts/inspect_dataset.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_dataset

if __name__ == "__main__":
    print("Downloading customer support dataset...")
    df = load_raw_dataset()
    print(f"✓ Loaded {len(df):,} records")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
    print("\nFirst few rows:")
    print(df.head())