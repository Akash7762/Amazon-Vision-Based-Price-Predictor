import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    
    # 1. Remove duplicates based on sample_id or image_link
    df = df.drop_duplicates(subset=['sample_id', 'image_link'])
    after_dedup = len(df)
    print(f"Removed {initial_rows - after_dedup} duplicate rows.")
    
    # 2. Handle missing values (just in case, though inspect showed none)
    # Price is the target, drop if missing. Drop if image_link is missing.
    df = df.dropna(subset=['price', 'image_link'])
    after_dropna = len(df)
    print(f"Removed {after_dedup - after_dropna} rows with missing price or image link.")
    
    # 3. Normalize the price field
    # Prices in this dataset are numeric (float) representing USD. 
    # Ensure it's a numeric type (float32 for efficiency)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price']) # drop any that failed to parse
    df['price'] = df['price'].astype(np.float32)
    after_parse = len(df)
    print(f"Removed {after_dropna - after_parse} rows with invalid price formats.")
    
    # 4. Outlier removal using a percentile-based rule
    # We will remove the top 1% and bottom 1% to handle extreme outliers (e.g. 0 price or insanely high)
    lower_bound = df['price'].quantile(0.01)
    upper_bound = df['price'].quantile(0.99)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    after_outliers = len(df)
    print(f"Removed {after_parse - after_outliers} rows as price outliers ( outside {lower_bound:.2f} - {upper_bound:.2f} USD ).")
    
    # 5. Extract additional metadata (optional but helpful for stratification/EDA)
    # The 'catalog_content' contains 'Item Name: ...'
    # We can extract a naive 'title' or 'category' if needed, but for now we'll just keep it.
    
    # Save cleaned dataset
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}. Final row count: {len(df)}")
    return df

if __name__ == "__main__":
    clean_data("data/raw/train.csv", "data/processed/metadata.csv")
