# Training Guide

## Configuration
We use two main config files in `configs/`:
- `model_config.yaml`: Defines the base model (e.g. `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`) and LoRA settings.
  - `rank` (16): The dimension of the LoRA matrices. Higher means more capacity, but slower/more memory.
  - `alpha` (32): Scaling factor for LoRA weights. Usually 2x the rank.
  - `dropout` (0.05): Dropout probability for regularization.
- `train_config.yaml`: Defines hyperparameters.
  - `batch_size`: Effective batch size is `per_device_train_batch_size * gradient_accumulation_steps`. Currently `1 * 16 = 16`.
  - `learning_rate`: Default `2e-4`.
  - `max_seq_length`: Maximum context window (1024). Use `stats.py` to tune this based on your dataset!

## Checkpoint Recovery & Resuming
- The script automatically saves a checkpoint every `save_steps` (50 by default).
- If your Google Colab/Kaggle session disconnects, DO NOT WORRY.
- Simply run `python scripts/resume_training.py --dataset ...` and it will automatically find the latest checkpoint in your Google Drive and resume training exactly where it left off.

## Safety
- Training scripts never perform `rm -rf`. 
- Checkpoint directories are overwritten safely by HuggingFace `transformers`.
