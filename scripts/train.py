"""
Main training script for fine-tuning the model with LoRA.
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_acquisition import DataAcquisition
from data.data_preprocessing import DataPreprocessor
from models.model_config import ModelConfig, ModelInitializer
from training.trainer import ModelTrainer


def main(args):
    """Main training function."""
    
    print("="*80)
    print("Domain-Specific Text Summarization - Training Pipeline")
    print("="*80)
    
    # Step 1: Data Acquisition
    print("\n[Step 1/5] Data Acquisition")
    print("-"*80)
    data_acq = DataAcquisition(data_dir=args.data_dir)
    
    if not os.path.exists(os.path.join(args.data_dir, "train.csv")):
        print("Fetching dataset...")
        train_df, val_df, test_df = data_acq.fetch_arxiv_dataset(num_samples=args.num_samples)
    else:
        print("Loading existing dataset...")
        train_df, val_df, test_df = data_acq.load_from_csv()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Step 2: Data Preprocessing
    print("\n[Step 2/5] Data Preprocessing")
    print("-"*80)
    preprocessor = DataPreprocessor(
        model_name=args.model_name,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
    train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
        train_df, val_df, test_df
    )
    
    print(f"Created datasets with max_source_length={args.max_source_length}, "
          f"max_target_length={args.max_target_length}")
    
    # Step 3: Model Initialization
    print("\n[Step 3/5] Model Initialization with LoRA")
    print("-"*80)
    model_config = ModelConfig(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit
    )
    
    initializer = ModelInitializer(model_config)
    model, tokenizer = initializer.initialize_model()
    
    # Step 4: Training
    print("\n[Step 4/5] Training")
    print("-"*80)
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        mlflow_experiment_name=args.experiment_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    trained_model = trainer.train(run_name=args.run_name)
    
    # Step 5: Save final model
    print("\n[Step 5/5] Saving Final Model")
    print("-"*80)
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Model saved to: {final_model_path}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print(f"\nTo evaluate the model, run:")
    print(f"python scripts/evaluate.py --model_path {final_model_path}")
    print(f"\nTo start the API server, run:")
    print(f"MODEL_PATH={final_model_path} uvicorn src.api.app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text summarization model with LoRA")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory for data storage")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to use from dataset")
    parser.add_argument("--max_source_length", type=int, default=512,
                        help="Maximum length for source text")
    parser.add_argument("--max_target_length", type=int, default=150,
                        help="Maximum length for target summary")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="t5-small",
                        choices=["t5-small", "t5-base", "flan-t5-small", "flan-t5-base"],
                        help="Name of the base model")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="models/checkpoints",
                        help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency")
    
    # MLflow arguments
    parser.add_argument("--experiment_name", type=str, default="text-summarization",
                        help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="MLflow run name")
    
    args = parser.parse_args()
    
    main(args)
