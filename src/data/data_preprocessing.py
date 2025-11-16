"""
Data preprocessing module for tokenizing and preparing data for fine-tuning.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import pandas as pd


class SummarizationDataset(Dataset):
    """PyTorch Dataset for text summarization."""
    
    def __init__(
        self,
        texts: List[str],
        summaries: List[str],
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 150
    ):
        """
        Initialize summarization dataset.
        
        Args:
            texts: List of source texts
            summaries: List of target summaries
            tokenizer: Hugging Face tokenizer
            max_source_length: Maximum length for source text
            max_target_length: Maximum length for target summary
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])
        
        # Tokenize inputs
        source = self.tokenizer(
            text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        target = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids
        }


class DataPreprocessor:
    """Class to handle data preprocessing and dataset creation."""
    
    def __init__(
        self,
        model_name: str = "t5-small",
        max_source_length: int = 512,
        max_target_length: int = 150
    ):
        """
        Initialize data preprocessor.
        
        Args:
            model_name: Name of the model to use for tokenizer
            max_source_length: Maximum length for source text
            max_target_length: Maximum length for target summary
        """
        self.model_name = model_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> tuple:
        """
        Create PyTorch datasets from dataframes.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Add prefix for T5 models
        if "t5" in self.model_name.lower():
            train_texts = ["summarize: " + text for text in train_df["text"].tolist()]
            val_texts = ["summarize: " + text for text in val_df["text"].tolist()]
            test_texts = ["summarize: " + text for text in test_df["text"].tolist()]
        else:
            train_texts = train_df["text"].tolist()
            val_texts = val_df["text"].tolist()
            test_texts = test_df["text"].tolist()
        
        train_summaries = train_df["summary"].tolist()
        val_summaries = val_df["summary"].tolist()
        test_summaries = test_df["summary"].tolist()
        
        train_dataset = SummarizationDataset(
            train_texts, train_summaries, self.tokenizer,
            self.max_source_length, self.max_target_length
        )
        
        val_dataset = SummarizationDataset(
            val_texts, val_summaries, self.tokenizer,
            self.max_source_length, self.max_target_length
        )
        
        test_dataset = SummarizationDataset(
            test_texts, test_summaries, self.tokenizer,
            self.max_source_length, self.max_target_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 4
    ) -> tuple:
        """
        Create PyTorch dataloaders from datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size for dataloaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    from data_acquisition import DataAcquisition
    
    # Load data
    data_acq = DataAcquisition()
    train_df, val_df, test_df = data_acq.load_from_csv()
    
    # Preprocess data
    preprocessor = DataPreprocessor(model_name="t5-small")
    train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
        train_df, val_df, test_df
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=4
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
