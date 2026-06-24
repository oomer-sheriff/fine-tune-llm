# Tamil LLM Fine-Tuning Workspace

A production-quality workspace for fine-tuning Tamil Large Language Models (like Qwen2.5-7B) using QLoRA and Unsloth on free Google Colab and Kaggle GPUs.

## Features
- **Auto Checkpoint Recovery**: Resumes training automatically after Colab/Kaggle disconnects.
- **Google Drive Integration**: Directly saves adapters and checkpoints to Drive to prevent data loss.
- **Dataset Tools**: Utilities to validate, analyze, and convert datasets.
- **Pre-configured for Tamil**: Supports 4-bit quantization, optimal LoRA settings, and token length analysis.

## Quick Start
1. Open `notebooks/Colab_Training.ipynb` in Google Colab.
2. Mount your Google Drive.
3. Run the cells to install requirements and start training using your dataset.

## Documentation
- [Dataset Guide](docs/DATASET_GUIDE.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
- [Evaluation Guide](docs/EVALUATION_GUIDE.md)

## Requirements
See `requirements.txt` for details. Primary dependencies:
- `unsloth`
- `trl`
- `peft`
- `transformers`
