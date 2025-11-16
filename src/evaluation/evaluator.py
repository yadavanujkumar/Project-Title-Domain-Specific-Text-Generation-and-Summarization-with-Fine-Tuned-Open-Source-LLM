"""
Model evaluation module with ROUGE metrics.
"""

import torch
from rouge_score import rouge_scorer
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import pandas as pd


class ModelEvaluator:
    """Class to evaluate model performance using ROUGE metrics."""
    
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 150,
        num_beams: int = 4,
        early_stopping: bool = True
    ) -> str:
        """
        Generate summary for a given text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early when all beams are finished
            
        Returns:
            Generated summary
        """
        # Add prefix for T5 models
        if "t5" in self.tokenizer.name_or_path.lower():
            text = "summarize: " + text
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=2
            )
        
        # Decode summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def generate_summaries_batch(
        self,
        texts: List[str],
        max_length: int = 150,
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum length of generated summaries
            batch_size: Batch size for generation
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating summaries"):
            batch_texts = texts[i:i + batch_size]
            
            # Add prefix for T5 models
            if "t5" in self.tokenizer.name_or_path.lower():
                batch_texts = ["summarize: " + text for text in batch_texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summaries
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode summaries
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def calculate_rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for predictions vs references.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with average ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_std': np.std(rouge1_scores),
            'rouge2_std': np.std(rouge2_scores),
            'rougeL_std': np.std(rougeL_scores)
        }
    
    def evaluate_dataset(
        self,
        test_df: pd.DataFrame,
        batch_size: int = 4,
        num_samples: int = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a test dataset.
        
        Args:
            test_df: Test dataframe with 'text' and 'summary' columns
            batch_size: Batch size for generation
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with ROUGE scores
        """
        if num_samples:
            test_df = test_df.head(num_samples)
        
        texts = test_df['text'].tolist()
        references = test_df['summary'].tolist()
        
        print(f"Evaluating on {len(texts)} samples...")
        
        # Generate summaries
        predictions = self.generate_summaries_batch(texts, batch_size=batch_size)
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        
        print("\nROUGE Scores:")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f} (±{rouge_scores['rouge1_std']:.4f})")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f} (±{rouge_scores['rouge2_std']:.4f})")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f} (±{rouge_scores['rougeL_std']:.4f})")
        
        return rouge_scores, predictions, references
    
    def qualitative_analysis(
        self,
        texts: List[str],
        predictions: List[str],
        references: List[str],
        num_examples: int = 5
    ):
        """
        Perform qualitative analysis of generated summaries.
        
        Args:
            texts: List of input texts
            predictions: List of predicted summaries
            references: List of reference summaries
            num_examples: Number of examples to display
        """
        print("\n" + "="*80)
        print("QUALITATIVE ANALYSIS")
        print("="*80)
        
        for i in range(min(num_examples, len(texts))):
            print(f"\nExample {i+1}:")
            print("-" * 80)
            print(f"Input Text (first 200 chars): {texts[i][:200]}...")
            print(f"\nReference Summary: {references[i]}")
            print(f"\nGenerated Summary: {predictions[i]}")
            
            # Calculate ROUGE for this example
            scores = self.scorer.score(references[i], predictions[i])
            print(f"\nROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
            print("-" * 80)


if __name__ == "__main__":
    print("This module should be imported and used with a trained model.")
    print("See evaluation scripts in the scripts/ directory for usage examples.")
