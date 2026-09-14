import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, output_path):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # We will split 70/15/15.
    # To stratify, we create price bins since price is a continuous variable.
    df['price_bin'] = pd.qcut(df['price'], q=10, labels=False, duplicates='drop')
    
    # Split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['price_bin'])
    
    # Split Temp into Val (15%) and Test (15%) - which is 50% of the Temp dataset
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['price_bin'])
    
    # Add a 'split' column
    df['split'] = 'unassigned'
    df.loc[train_df.index, 'split'] = 'train'
    df.loc[val_df.index, 'split'] = 'val'
    df.loc[test_df.index, 'split'] = 'test'
    
    # Drop the temporary bin
    df = df.drop(columns=['price_bin'])
    
    # Save the updated metadata
    df.to_csv(output_path, index=False)
    
    print("Data split completed and saved as a 'split' column.")
    print("Split sizes:")
    print(df['split'].value_counts())

if __name__ == "__main__":
    split_data("data/processed/metadata.csv", "data/processed/metadata.csv")
