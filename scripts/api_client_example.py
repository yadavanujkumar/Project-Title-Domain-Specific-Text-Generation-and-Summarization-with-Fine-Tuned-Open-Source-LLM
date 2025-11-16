"""
Example API client for the text summarization and generation service.
"""

import requests
from typing import List, Optional
import json


class SummarizationAPIClient:
    """Client for interacting with the summarization API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """
        Check API health status.
        
        Returns:
            Health status dictionary
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        num_beams: int = 4
    ) -> dict:
        """
        Summarize a single text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            num_beams: Number of beams for beam search
            
        Returns:
            Dictionary with summary and metadata
        """
        payload = {
            "text": text,
            "max_length": max_length,
            "num_beams": num_beams
        }
        
        response = requests.post(
            f"{self.base_url}/summarize",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def summarize_batch(
        self,
        texts: List[str],
        max_length: int = 150,
        num_beams: int = 4
    ) -> dict:
        """
        Summarize multiple texts in a batch (more efficient).
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length of summaries
            num_beams: Number of beams for beam search
            
        Returns:
            Dictionary with summaries and metadata
        """
        payload = {
            "texts": texts,
            "max_length": max_length,
            "num_beams": num_beams
        }
        
        response = requests.post(
            f"{self.base_url}/summarize-batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        num_beams: int = 4,
        temperature: float = 1.0
    ) -> dict:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Prompt for text generation
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Dictionary with generated text and metadata
        """
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = SummarizationAPIClient(base_url="http://localhost:8000")
    
    # Check health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Model loaded: {health['model_loaded']}")
        print(f"Device: {health['device']}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure it's running.")
        print("Start with: MODEL_PATH=models/checkpoints/final_model uvicorn src.api.app:app")
        return
    
    # Example text for summarization
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on the 
    development of algorithms and statistical models that enable computers to 
    improve their performance on a specific task through experience. Unlike 
    traditional programming where explicit instructions are provided, machine 
    learning systems learn patterns from data. There are three main types of 
    machine learning: supervised learning, where the model learns from labeled 
    data; unsupervised learning, where the model finds patterns in unlabeled 
    data; and reinforcement learning, where an agent learns to make decisions 
    through trial and error. Deep learning, a subset of machine learning, uses 
    neural networks with multiple layers to learn hierarchical representations 
    of data. This approach has achieved remarkable success in various domains 
    including computer vision, natural language processing, and speech recognition.
    """
    
    # Single summarization
    print("\n" + "="*80)
    print("Single Text Summarization")
    print("="*80)
    print(f"Input text length: {len(sample_text)} characters")
    
    try:
        result = client.summarize(sample_text, max_length=100)
        print(f"\nGenerated Summary:")
        print(result['summary'])
        print(f"\nSummary length: {result['summary_length']} characters")
    except Exception as e:
        print(f"Error during summarization: {e}")
    
    # Batch summarization
    print("\n" + "="*80)
    print("Batch Text Summarization")
    print("="*80)
    
    texts = [
        "Artificial intelligence is revolutionizing healthcare.",
        "Climate change poses significant challenges to global ecosystems.",
        "Quantum computing promises to solve complex problems faster than classical computers."
    ]
    
    try:
        result = client.summarize_batch(texts, max_length=50)
        print(f"Summarized {result['count']} texts:")
        for i, summary in enumerate(result['summaries'], 1):
            print(f"\n{i}. {summary}")
    except Exception as e:
        print(f"Error during batch summarization: {e}")
    
    # Text generation
    print("\n" + "="*80)
    print("Text Generation")
    print("="*80)
    
    prompt = "The future of artificial intelligence in medicine"
    
    try:
        result = client.generate(prompt, max_length=150, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"\nGenerated Text:")
        print(result['generated_text'])
    except Exception as e:
        print(f"Error during text generation: {e}")
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
