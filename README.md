# Domain-Specific Text Generation and Summarization with Fine-Tuned Open-Source LLM

An end-to-end solution for domain-specific text summarization and generation using fine-tuned Large Language Models (LLMs) with Parameter-Efficient Fine-Tuning (PEFT) techniques.

## 🎯 Overview

This project implements a complete pipeline for:
- **Text Summarization**: Automatically generate concise summaries of lengthy domain-specific documents
- **Text Generation**: Generate coherent domain-specific content from prompts
- **Model Fine-tuning**: Leverage LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Production Deployment**: Dockerized FastAPI service with optimized inference

## 🏗️ Architecture

```
.
├── src/
│   ├── data/                    # Data acquisition and preprocessing
│   │   ├── data_acquisition.py  # Dataset fetching (ArXiv papers)
│   │   └── data_preprocessing.py # Tokenization and PyTorch datasets
│   ├── models/                  # Model configuration and initialization
│   │   └── model_config.py      # LoRA setup and model loading
│   ├── training/                # Training pipeline
│   │   └── trainer.py           # Training loop with MLflow tracking
│   ├── evaluation/              # Model evaluation
│   │   └── evaluator.py         # ROUGE metrics and qualitative analysis
│   ├── utils/                   # Utility functions
│   │   └── inference_optimizer.py # Quantization and batching
│   └── api/                     # FastAPI application
│       └── app.py               # REST API endpoints
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── data/                        # Data storage
├── models/                      # Model checkpoints
├── notebooks/                   # Jupyter notebooks
├── Dockerfile                   # Container definition
└── requirements.txt             # Python dependencies
```

## ✨ Features

### ✅ Technical Requirements Checklist

- [x] **Python & Hugging Face Transformers**: Full implementation using modern transformers library
- [x] **PEFT (LoRA)**: Parameter-efficient fine-tuning with configurable LoRA ranks
- [x] **MLflow Tracking**: Comprehensive experiment tracking for hyperparameters, metrics, and models
- [x] **ROUGE Evaluation**: Quantitative evaluation with ROUGE-1, ROUGE-2, and ROUGE-L scores
- [x] **Inference Optimization**: 4-bit and 8-bit quantization support with batch processing
- [x] **Dockerized API**: FastAPI service with `/summarize`, `/generate`, and batch endpoints
- [x] **Complete Pipeline**: Training, evaluation, and serving code included

### 🚀 Key Capabilities

1. **Data Curation**
   - Automated dataset acquisition from scientific papers (ArXiv)
   - Support for 5,000+ text-summary pairs
   - Train/validation/test split with preprocessing

2. **Model Fine-tuning**
   - T5 and FLAN-T5 model support
   - LoRA-based PEFT for efficient training
   - Automatic mixed precision training
   - Gradient accumulation for larger effective batch sizes

3. **Experiment Tracking**
   - MLflow integration for all experiments
   - Automatic logging of hyperparameters, metrics, and artifacts
   - Model versioning and comparison

4. **Comprehensive Evaluation**
   - ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
   - Qualitative analysis with example outputs
   - Inference speed benchmarking

5. **Production-Ready Deployment**
   - FastAPI REST API with OpenAPI documentation
   - Docker containerization
   - Optimized inference with quantization
   - Batch processing support

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Docker (for containerized deployment)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Project-Title-Domain-Specific-Text-Generation-and-Summarization-with-Fine-Tuned-Open-Source-LLM
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import transformers, torch; print('Installation successful!')"
```

## 🚀 Quick Start

### 1. Data Preparation

```bash
# The training script will automatically download the ArXiv dataset
# Alternatively, you can pre-download it:
python -c "from src.data.data_acquisition import DataAcquisition; DataAcquisition().fetch_arxiv_dataset()"
```

### 2. Model Training

**Basic training with default settings:**
```bash
python scripts/train.py
```

**Advanced training with custom parameters:**
```bash
python scripts/train.py \
    --model_name t5-small \
    --num_epochs 5 \
    --train_batch_size 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_samples 5000 \
    --experiment_name my-experiment \
    --run_name my-run
```

**Training with quantization (for limited GPU memory):**
```bash
python scripts/train.py --use_8bit
```

### 3. Model Evaluation

**Basic evaluation:**
```bash
python scripts/evaluate.py \
    --model_path models/checkpoints/final_model \
    --qualitative_analysis
```

**Evaluation with benchmarking and MLflow logging:**
```bash
python scripts/evaluate.py \
    --model_path models/checkpoints/final_model \
    --use_optimized \
    --benchmark \
    --qualitative_analysis \
    --log_mlflow \
    --num_samples 100
```

**Evaluation with 4-bit quantization:**
```bash
python scripts/evaluate.py \
    --model_path models/checkpoints/final_model \
    --use_optimized \
    --use_4bit \
    --benchmark
```

### 4. API Deployment

**Local deployment:**
```bash
MODEL_PATH=models/checkpoints/final_model uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Docker deployment:**
```bash
# Build the Docker image
docker build -t text-summarization-api .

# Run the container
docker run -p 8000:8000 \
    -e MODEL_PATH=/app/models/checkpoints/final_model \
    text-summarization-api
```

**Docker deployment with quantization:**
```bash
docker run -p 8000:8000 \
    -e MODEL_PATH=/app/models/checkpoints/final_model \
    -e USE_8BIT=true \
    text-summarization-api
```

### 5. Using the API

**Access the interactive API documentation:**
```
http://localhost:8000/docs
```

**Example API requests:**

```bash
# Health check
curl http://localhost:8000/health

# Summarize a single text
curl -X POST "http://localhost:8000/summarize" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Your long text here...",
        "max_length": 150,
        "num_beams": 4
    }'

# Batch summarization (more efficient)
curl -X POST "http://localhost:8000/summarize-batch" \
    -H "Content-Type: application/json" \
    -d '{
        "texts": ["Text 1...", "Text 2...", "Text 3..."],
        "max_length": 150
    }'

# Generate text from prompt
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "In the field of quantum computing",
        "max_length": 200,
        "temperature": 0.8
    }'
```

**Python client example:**
```python
import requests

# Summarize text
response = requests.post(
    "http://localhost:8000/summarize",
    json={
        "text": "Your long scientific paper or article here...",
        "max_length": 150
    }
)
result = response.json()
print(f"Summary: {result['summary']}")
```

## 📊 Model Performance

The model is evaluated using ROUGE metrics:

- **ROUGE-1**: Measures unigram overlap
- **ROUGE-2**: Measures bigram overlap  
- **ROUGE-L**: Measures longest common subsequence

Expected performance on ArXiv dataset (after 3 epochs):
- ROUGE-1: ~0.35-0.40
- ROUGE-2: ~0.12-0.15
- ROUGE-L: ~0.30-0.35

## 🔧 Configuration

### Model Selection

Supported models:
- `t5-small` (60M parameters) - Fast, suitable for prototyping
- `t5-base` (220M parameters) - Better quality, more resources
- `flan-t5-small` (80M parameters) - Instruction-tuned variant
- `flan-t5-base` (250M parameters) - Larger instruction-tuned variant

### LoRA Configuration

Key hyperparameters:
- `lora_r`: Rank of LoRA matrices (default: 8)
- `lora_alpha`: Scaling parameter (default: 32)
- `lora_dropout`: Dropout rate (default: 0.1)

Higher rank → More parameters → Better performance but slower training

### Training Configuration

Adjust based on your hardware:
- `per_device_train_batch_size`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Accumulate gradients (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `num_train_epochs`: Number of epochs (default: 3)

## 🔬 Experiment Tracking

View your experiments with MLflow:

```bash
mlflow ui --port 5000
```

Then open http://localhost:5000 in your browser.

MLflow tracks:
- All hyperparameters
- Training and validation loss
- ROUGE scores
- Model artifacts
- Run comparisons

## 🐳 Docker Deployment

### Building the Image

```bash
docker build -t text-summarization-api:latest .
```

### Running the Container

```bash
docker run -d \
    -p 8000:8000 \
    -e MODEL_PATH=/app/models/checkpoints/final_model \
    -e USE_8BIT=false \
    --name summarization-api \
    text-summarization-api:latest
```

### Cloud Deployment

The Docker image can be deployed to:
- **AWS**: ECS, EKS, or SageMaker Endpoints
- **GCP**: Cloud Run, GKE, or Vertex AI
- **Azure**: Container Instances or AKS

## 📈 Performance Optimization

### Quantization

Use quantization to reduce memory usage and increase speed:

- **8-bit quantization**: ~50% memory reduction, minimal quality loss
- **4-bit quantization**: ~75% memory reduction, slight quality loss

### Batch Processing

Use batch endpoints for processing multiple texts:
- 3-5x faster than sequential processing
- More efficient GPU utilization

### Model Optimization Tips

1. Use gradient checkpointing for training larger models
2. Enable mixed precision training (FP16) on modern GPUs
3. Use gradient accumulation to simulate larger batch sizes
4. Consider using larger models (t5-base) for better quality

## 🧪 Testing

Run tests to verify the installation:

```bash
# Test data acquisition
python -m src.data.data_acquisition

# Test model initialization
python -m src.models.model_config

# Test API (after training)
pytest tests/  # If tests are added
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{domain_specific_summarization,
  title={Domain-Specific Text Generation and Summarization with Fine-Tuned Open-Source LLM},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/project}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Troubleshooting

### Common Issues

**Out of memory during training:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `--use_8bit` flag for quantization

**CUDA errors:**
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Update PyTorch: `pip install --upgrade torch`

**API model loading fails:**
- Check MODEL_PATH environment variable
- Ensure model files exist in the specified path
- Verify model was saved correctly during training

## 📧 Support

For questions and support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Microsoft for the LoRA (PEFT) implementation
- Scientific Papers dataset from ArXiv
- FastAPI framework for the REST API