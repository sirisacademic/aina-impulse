
#!/usr/bin/env python
"""Inspect a RIS3CAT pickle to understand shape and columns."""
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to RIS3CAT pickle (DataFrame).")
    args = ap.parse_args()

    df = pd.read_pickle(args.input)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(3))

if __name__ == "__main__":
    main()
