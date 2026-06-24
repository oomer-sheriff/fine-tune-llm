Build a production-quality Tamil LLM fine-tuning workspace.

Goal:
I want to fine-tune a 7B instruction model (Qwen2.5-7B-Instruct 4-bit) using QLoRA on free Google Colab and Kaggle GPUs.

The project must be designed for:
- Frequent disconnections
- Automatic checkpoint recovery
- Storage on Google Drive
- Long-running training across many sessions
- Tamil language specialization

Create a complete repository with the following structure:

project/
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── evaluate.py
├── dataset_tools/
│   ├── validate_dataset.py
│   ├── convert_json_to_jsonl.py
│   └── stats.py
├── configs/
│   ├── train_config.yaml
│   └── model_config.yaml
├── datasets/
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── examples/
├── scripts/
│   ├── resume_training.py
│   ├── latest_checkpoint.py
│   └── export_adapter.py
├── notebooks/
│   ├── Colab_Training.ipynb
│   └── Kaggle_Training.ipynb
└── docs/
    ├── DATASET_GUIDE.md
    ├── TRAINING_GUIDE.md
    └── EVALUATION_GUIDE.md

Requirements:

1. Use:
   - Unsloth
   - Transformers
   - PEFT
   - TRL
   - Datasets
   - BitsAndBytes

2. Model:
   unsloth/Qwen2.5-7B-Instruct-bnb-4bit

3. LoRA settings:
   rank = 16
   alpha = 32
   dropout = 0.05

4. Training settings:
   max_seq_length = 1024
   batch_size = 1
   gradient_accumulation_steps = 16
   learning_rate = 2e-4
   epochs = 2

5. Dataset format:

{
  "instruction": "...",
  "input": "...",
  "output": "..."
}

Create validation logic to verify:
- required fields exist
- fields are strings
- no empty outputs
- duplicate detection
- length statistics

6. Checkpointing:

Implement robust checkpointing.

Requirements:
- save every 50 steps
- automatically detect latest checkpoint
- automatically resume
- continue training after Colab/Kaggle disconnects
- save final adapter separately

7. Google Drive support:

Implement:
- automatic Drive mounting
- configurable Drive path
- checkpoint storage in Drive
- dataset storage in Drive

8. Evaluation:

Create an evaluation framework that:
- loads the latest adapter
- runs a benchmark set
- outputs results to JSON
- calculates response lengths
- saves generations

9. Dataset tools:

Create utilities for:
- merging datasets
- deduplicating examples
- validating examples
- reporting dataset statistics

10. Documentation:

Write detailed documentation explaining:
- every training parameter
- every LoRA parameter
- effective batch size calculation
- checkpoint recovery
- dataset creation guidelines
- Tamil dataset best practices

11. Safety:

The training scripts must never delete files.

Never run rm -rf.

Never remove directories automatically.

Always require explicit confirmation for destructive actions.

12. Agent workflow:

Before writing code:
- Create a detailed implementation plan.
- Explain architecture decisions.
- Identify possible failure modes.
- Explain how checkpoint recovery works.
- Explain how training resumes after disconnects.

Then implement everything.

After implementation:
- Review all files.
- Run static checks.
- Verify imports.
- Verify paths.
- Verify checkpoint logic.
- Generate a final project summary.

