"""
Inference optimization module with quantization and batching support.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Optional, Union
import time


class OptimizedInference:
    """Class for optimized model inference with quantization and batching."""
    
    def __init__(
        self,
        model_path: str,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize optimized inference.
        
        Args:
            model_path: Path to the model
            use_4bit: Whether to use 4-bit quantization
            use_8bit: Whether to use 8-bit quantization
            device: Device to run inference on
        """
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.device = device
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Load model with optional quantization."""
        print(f"Loading model from {self.model_path}")
        
        # Configure quantization
        quantization_config = None
        if self.use_4bit or self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                load_in_8bit=self.use_8bit,
                bnb_4bit_compute_dtype=torch.float16 if self.use_4bit else None,
                bnb_4bit_use_double_quant=True if self.use_4bit else None,
                bnb_4bit_quant_type="nf4" if self.use_4bit else None
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        if self.use_4bit or self.use_8bit:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            model.to(self.device)
        
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        if self.use_4bit:
            print("Using 4-bit quantization")
        elif self.use_8bit:
            print("Using 8-bit quantization")
        
        return model, tokenizer
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        num_beams: int = 4
    ) -> str:
        """
        Generate summary for a single text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary
        """
        # Add prefix for T5 models
        if "t5" in self.model_path.lower():
            text = "summarize: " + text
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def summarize_batch(
        self,
        texts: List[str],
        max_length: int = 150,
        num_beams: int = 4
    ) -> List[str]:
        """
        Generate summaries for a batch of texts (more efficient).
        
        Args:
            texts: List of input texts to summarize
            max_length: Maximum length of generated summaries
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated summaries
        """
        # Add prefix for T5 models
        if "t5" in self.model_path.lower():
            texts = ["summarize: " + text for text in texts]
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return summaries
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        num_beams: int = 4,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input prompt for text generation
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def benchmark(self, text: str, num_runs: int = 10) -> dict:
        """
        Benchmark inference speed.
        
        Args:
            text: Sample text for benchmarking
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        # Warm-up run
        _ = self.summarize(text)
        
        # Benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.summarize(text)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times)
        }


if __name__ == "__main__":
    print("This module should be imported and used for optimized inference.")
    print("See the API implementation in src/api/ for usage examples.")
