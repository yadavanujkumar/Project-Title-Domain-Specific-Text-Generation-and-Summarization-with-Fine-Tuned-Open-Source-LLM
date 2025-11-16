"""
Evaluation script for the trained model.
"""

import os
import sys
import argparse
import mlflow

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_acquisition import DataAcquisition
from evaluation.evaluator import ModelEvaluator
from utils.inference_optimizer import OptimizedInference


def main(args):
    """Main evaluation function."""
    
    print("="*80)
    print("Domain-Specific Text Summarization - Evaluation Pipeline")
    print("="*80)
    
    # Step 1: Load test data
    print("\n[Step 1/3] Loading Test Data")
    print("-"*80)
    data_acq = DataAcquisition(data_dir=args.data_dir)
    _, _, test_df = data_acq.load_from_csv()
    print(f"Test samples: {len(test_df)}")
    
    # Step 2: Load model
    print("\n[Step 2/3] Loading Model")
    print("-"*80)
    
    if args.use_optimized:
        print("Using optimized inference engine...")
        inference_engine = OptimizedInference(
            model_path=args.model_path,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit
        )
        model = inference_engine.model
        tokenizer = inference_engine.tokenizer
    else:
        print("Using standard model loading...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Step 3: Evaluate
    print("\n[Step 3/3] Running Evaluation")
    print("-"*80)
    
    evaluator = ModelEvaluator(model, tokenizer)
    
    # Run quantitative evaluation
    rouge_scores, predictions, references = evaluator.evaluate_dataset(
        test_df,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # Log to MLflow if requested
    if args.log_mlflow:
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params({
                "model_path": args.model_path,
                "num_samples": args.num_samples or len(test_df),
                "batch_size": args.batch_size,
                "use_4bit": args.use_4bit,
                "use_8bit": args.use_8bit
            })
            mlflow.log_metrics(rouge_scores)
    
    # Qualitative analysis
    if args.qualitative_analysis:
        print("\n")
        evaluator.qualitative_analysis(
            test_df['text'].tolist()[:args.num_qualitative_examples],
            predictions[:args.num_qualitative_examples],
            references[:args.num_qualitative_examples],
            num_examples=args.num_qualitative_examples
        )
    
    # Benchmark if requested
    if args.benchmark:
        print("\n" + "="*80)
        print("BENCHMARKING INFERENCE SPEED")
        print("="*80)
        
        if args.use_optimized:
            sample_text = test_df.iloc[0]['text']
            benchmark_results = inference_engine.benchmark(sample_text, num_runs=args.benchmark_runs)
            
            print(f"\nResults over {args.benchmark_runs} runs:")
            print(f"Mean time: {benchmark_results['mean_time']:.4f}s")
            print(f"Min time: {benchmark_results['min_time']:.4f}s")
            print(f"Max time: {benchmark_results['max_time']:.4f}s")
            
            if args.log_mlflow and mlflow.active_run():
                mlflow.log_metrics({
                    "benchmark_mean_time": benchmark_results['mean_time'],
                    "benchmark_min_time": benchmark_results['min_time'],
                    "benchmark_max_time": benchmark_results['max_time']
                })
        else:
            print("Benchmarking requires --use_optimized flag")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained summarization model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory containing test data")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--qualitative_analysis", action="store_true",
                        help="Perform qualitative analysis")
    parser.add_argument("--num_qualitative_examples", type=int, default=5,
                        help="Number of examples for qualitative analysis")
    
    # Optimization arguments
    parser.add_argument("--use_optimized", action="store_true",
                        help="Use optimized inference engine")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    
    # Benchmarking arguments
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference speed benchmark")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                        help="Number of runs for benchmarking")
    
    # MLflow arguments
    parser.add_argument("--log_mlflow", action="store_true",
                        help="Log results to MLflow")
    parser.add_argument("--experiment_name", type=str, default="text-summarization-eval",
                        help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="MLflow run name")
    
    args = parser.parse_args()
    
    main(args)
