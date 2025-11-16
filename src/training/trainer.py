"""
Training module with MLflow experiment tracking.
"""

import os
import torch
import mlflow
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, Optional
import numpy as np


class MLflowCallback(TrainerCallback):
    """Custom callback to log metrics to MLflow during training."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to MLflow."""
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.global_step)


class ModelTrainer:
    """Class to handle model training with MLflow tracking."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir: str = "models/checkpoints",
        mlflow_experiment_name: str = "text-summarization",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        gradient_accumulation_steps: int = 4
    ):
        """
        Initialize model trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model checkpoints
            mlflow_experiment_name: Name of MLflow experiment
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            save_steps: Checkpoint save frequency
            eval_steps: Evaluation frequency
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.mlflow_experiment_name = mlflow_experiment_name
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
            report_to=["none"],  # We'll use custom MLflow logging
            save_total_limit=2,
            push_to_hub=False
        )
    
    def train(self, run_name: Optional[str] = None):
        """
        Train the model with MLflow tracking.
        
        Args:
            run_name: Optional name for the MLflow run
            
        Returns:
            Trained model
        """
        # Set MLflow experiment
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params({
                "model_name": getattr(self.model, "config", {}).get("_name_or_path", "unknown"),
                "num_train_epochs": self.training_args.num_train_epochs,
                "per_device_train_batch_size": self.training_args.per_device_train_batch_size,
                "learning_rate": self.training_args.learning_rate,
                "weight_decay": self.training_args.weight_decay,
                "warmup_steps": self.training_args.warmup_steps,
                "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps,
            })
            
            # Log LoRA configuration if available
            if hasattr(self.model, "peft_config"):
                peft_config = list(self.model.peft_config.values())[0]
                mlflow.log_params({
                    "lora_r": peft_config.r,
                    "lora_alpha": peft_config.lora_alpha,
                    "lora_dropout": peft_config.lora_dropout,
                })
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                callbacks=[MLflowCallback()]
            )
            
            # Train
            print("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            final_model_path = os.path.join(self.output_dir, "final_model")
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Log final metrics
            mlflow.log_metrics({
                "final_train_loss": train_result.training_loss,
            })
            
            # Log model as artifact
            mlflow.log_artifacts(final_model_path, artifact_path="model")
            
            print("Training completed!")
            print(f"Final training loss: {train_result.training_loss}")
            
            return trainer.model


if __name__ == "__main__":
    # Example usage (requires data and model setup)
    print("This module should be imported and used with prepared data and model.")
    print("See training scripts in the scripts/ directory for usage examples.")
