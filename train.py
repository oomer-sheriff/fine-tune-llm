import yaml
import argparse
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]
    
    if input_text:
        prompt = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n{output_text}"
    else:
        prompt = f"Instruction:\n{instruction}\n\nOutput:\n{output_text}"
    
    return {"text": prompt}

def train(model_config_path, train_config_path, dataset_path, val_dataset_path=None, resume_from_checkpoint=False):
    model_config = load_config(model_config_path)
    train_config = load_config(train_config_path)
    
    max_seq_length = train_config['training_settings']['max_seq_length']
    
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_config['model_name'],
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    print("Applying LoRA...")
    lora = model_config['lora_settings']
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora['rank'],
        target_modules = lora['target_modules'],
        lora_alpha = lora['alpha'],
        lora_dropout = lora['dropout'],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = train_config['training_settings']['seed'],
        use_rslora = False,
        loftq_config = None,
    )
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")
    
    EOS_TOKEN = tokenizer.eos_token
    def format_and_eos(example):
        formatted = format_prompt(example)
        formatted["text"] += EOS_TOKEN
        return formatted
        
    dataset = dataset.map(format_and_eos)
    
    val_dataset = None
    if val_dataset_path:
        print(f"Loading validation dataset from {val_dataset_path}...")
        val_dataset = load_dataset("json", data_files={"val": val_dataset_path}, split="val")
        val_dataset = val_dataset.map(format_and_eos)
    
    output_dir = Path(train_config['storage']['drive_path']) / train_config['storage']['checkpoint_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = train_config['training_settings']['per_device_train_batch_size'],
        gradient_accumulation_steps = train_config['training_settings']['gradient_accumulation_steps'],
        warmup_steps = train_config['training_settings']['warmup_steps'],
        num_train_epochs = train_config['training_settings']['num_train_epochs'],
        learning_rate = float(train_config['training_settings']['learning_rate']),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = train_config['training_settings']['logging_steps'],
        eval_strategy = train_config['training_settings'].get('eval_strategy', 'no'),
        eval_steps = train_config['training_settings'].get('eval_steps', None),
        optim = train_config['training_settings']['optim'],
        weight_decay = train_config['training_settings']['weight_decay'],
        lr_scheduler_type = train_config['training_settings']['lr_scheduler_type'],
        seed = train_config['training_settings']['seed'],
        output_dir = str(output_dir),
        save_steps = train_config['training_settings']['save_steps'],
        save_total_limit = train_config['training_settings']['save_total_limit'],
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = val_dataset,
        args = training_args,
    )
    
    print("Starting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training complete!")
    
    final_output = Path(train_config['storage']['drive_path']) / train_config['storage']['output_adapter_dir']
    final_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"Final adapter saved to {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val_dataset", default=None, help="Path to the validation dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    train(args.model_config, args.train_config, args.dataset, args.val_dataset, args.resume)
