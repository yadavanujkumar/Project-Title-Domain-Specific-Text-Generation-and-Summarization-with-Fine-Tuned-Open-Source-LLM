"""
Simple test script to validate the installation and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        
        import peft
        print(f"✓ PEFT {peft.__version__}")
        
        import mlflow
        print(f"✓ MLflow {mlflow.__version__}")
        
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
        
        from rouge_score import rouge_scorer
        print(f"✓ ROUGE Score")
        
        print("\n✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA is not available. Training will be slow on CPU.")
    
    return True


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")
    try:
        from data.data_acquisition import DataAcquisition
        print("✓ data.data_acquisition")
        
        from data.data_preprocessing import DataPreprocessor
        print("✓ data.data_preprocessing")
        
        from models.model_config import ModelConfig
        print("✓ models.model_config")
        
        from training.trainer import ModelTrainer
        print("✓ training.trainer")
        
        from evaluation.evaluator import ModelEvaluator
        print("✓ evaluation.evaluator")
        
        from utils.inference_optimizer import OptimizedInference
        print("✓ utils.inference_optimizer")
        
        print("\n✓ All custom modules imported successfully!")
        return True
    except ImportError as e:
        print(f"\n✗ Module import failed: {e}")
        return False


def test_model_loading():
    """Test basic model loading."""
    print("\nTesting model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "t5-small"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"✓ Model loaded")
        
        # Test tokenization
        text = "summarize: This is a test sentence for the summarization model."
        inputs = tokenizer(text, return_tensors="pt")
        print(f"✓ Tokenization works")
        
        # Test inference
        outputs = model.generate(**inputs, max_length=50)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Inference works")
        print(f"  Generated: {summary}")
        
        print("\n✓ Model loading and inference successful!")
        return True
    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("Testing Installation and Setup")
    print("="*80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Model Loading", test_model_loading()))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train.py")
        print("2. Evaluate model: python scripts/evaluate.py --model_path models/checkpoints/final_model")
        print("3. Start API: uvicorn src.api.app:app")
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        print("Run: pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
