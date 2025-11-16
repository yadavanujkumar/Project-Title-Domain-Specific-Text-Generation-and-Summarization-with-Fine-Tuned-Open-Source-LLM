"""
Data acquisition module for fetching domain-specific text data.
This module fetches academic paper abstracts from ArXiv dataset.
"""

import os
from datasets import load_dataset
from typing import Dict, Tuple
import pandas as pd


class DataAcquisition:
    """Class to handle data acquisition from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize data acquisition with target directory."""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_arxiv_dataset(self, num_samples: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch ArXiv dataset for summarization task.
        
        Args:
            num_samples: Number of samples to fetch
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"Loading ArXiv dataset...")
        
        # Load scientific papers dataset from Hugging Face
        dataset = load_dataset("scientific_papers", "arxiv", split="train")
        
        # Select subset of data
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        # Convert to pandas dataframe
        df = pd.DataFrame({
            'text': dataset['article'],
            'summary': dataset['abstract']
        })
        
        # Split into train/val/test
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Save to CSV
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.data_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)
        
        return train_df, val_df, test_df
    
    def load_from_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously saved datasets from CSV files."""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(self.data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    data_acq = DataAcquisition()
    train_df, val_df, test_df = data_acq.fetch_arxiv_dataset(num_samples=5000)
    print(f"Successfully fetched and saved dataset!")
    print(f"Sample text length: {len(train_df.iloc[0]['text'])}")
    print(f"Sample summary length: {len(train_df.iloc[0]['summary'])}")
