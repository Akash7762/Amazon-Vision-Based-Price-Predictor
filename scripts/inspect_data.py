import pandas as pd

def inspect_data(filepath):
    print(f"--- Inspecting {filepath} ---")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        # Could be due to escapechar in amazon-ml dataset
        df = pd.read_csv(filepath, escapechar="\\")
        
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSample (first 2 rows):")
    print(df.head(2).to_dict('records'))

if __name__ == "__main__":
    inspect_data("data/raw/train.csv")
