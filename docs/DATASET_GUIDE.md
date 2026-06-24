# Dataset Guide for Tamil LLM Fine-Tuning

## Recommended Datasets
- **Tamil-ARIVU**: High quality Tamil instruction dataset.
- **Tamil Alpaca**: Translated Alpaca dataset for Tamil.
- **TamilMixSentiment**: Great for sentiment-based instructions.

## Data Format (JSONL)
Your dataset must be in `.jsonl` format, where each line is a valid JSON object with the following keys:
```json
{"instruction": "...", "input": "...", "output": "..."}
```
If there is no input, provide an empty string `""`.

## Tools
We provide `dataset_tools/` to help you:
1. `validate_dataset.py`: Ensure your JSONL is formatted correctly and has no empty fields.
2. `convert_json_to_jsonl.py`: Convert standard JSON arrays into JSONL format.
3. `stats.py`: Analyze the token length of your dataset to optimize `max_seq_length`.

## Safety
All tools use non-destructive operations. `convert_json_to_jsonl.py` will prompt for confirmation if the output file already exists.
