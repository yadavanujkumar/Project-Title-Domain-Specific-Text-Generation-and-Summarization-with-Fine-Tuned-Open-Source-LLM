"""
Model configuration and initialization module with LoRA support.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Optional


class ModelConfig:
    """Configuration class for model setup."""
    
    # Supported models
    SUPPORTED_MODELS = {
        "t5-small": "t5-small",
        "t5-base": "t5-base",
        "flan-t5-small": "google/flan-t5-small",
        "flan-t5-base": "google/flan-t5-base",
    }
    
    def __init__(
        self,
        model_name: str = "t5-small",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        """
        Initialize model configuration.
        
        Args:
            model_name: Name of the base model
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        
        # Validate model name
        if model_name not in self.SUPPORTED_MODELS:
            print(f"Warning: {model_name} not in supported models list. Proceeding anyway.")
        
        self.full_model_name = self.SUPPORTED_MODELS.get(model_name, model_name)


class ModelInitializer:
    """Class to initialize and configure models with LoRA."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model initializer with configuration.
        
        Args:
            config: ModelConfig instance
        """
        self.config = config
    
    def load_base_model(self):
        """
        Load base model with optional quantization.
        
        Returns:
            Loaded model
        """
        print(f"Loading base model: {self.config.full_model_name}")
        
        # Quantization configuration
        quantization_config = None
        if self.config.use_8bit or self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.use_8bit,
                load_in_4bit=self.config.use_4bit,
                bnb_4bit_compute_dtype=torch.float16 if self.config.use_4bit else None,
                bnb_4bit_use_double_quant=True if self.config.use_4bit else None,
                bnb_4bit_quant_type="nf4" if self.config.use_4bit else None
            )
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.full_model_name,
            quantization_config=quantization_config,
            device_map="auto" if (self.config.use_8bit or self.config.use_4bit) else None,
            torch_dtype=torch.float16 if (self.config.use_8bit or self.config.use_4bit) else torch.float32
        )
        
        # Prepare model for k-bit training if using quantization
        if self.config.use_8bit or self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        return model
    
    def add_lora_adapters(self, model):
        """
        Add LoRA adapters to the model.
        
        Args:
            model: Base model to add adapters to
            
        Returns:
            Model with LoRA adapters
        """
        print(f"Adding LoRA adapters with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q", "v"],  # For T5 models
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def initialize_model(self):
        """
        Initialize model with LoRA adapters.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load base model
        model = self.load_base_model()
        
        # Add LoRA adapters
        model = self.add_lora_adapters(model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.full_model_name)
        
        return model, tokenizer


if __name__ == "__main__":
    # Example usage
    config = ModelConfig(
        model_name="t5-small",
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        use_8bit=False
    )
    
    initializer = ModelInitializer(config)
    model, tokenizer = initializer.initialize_model()
    
    print(f"Model initialized successfully!")
    print(f"Model type: {type(model)}")
