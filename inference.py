import argparse
import yaml
from pathlib import Path
from unsloth import FastLanguageModel

def generate(adapter_path, model_config_path, instruction, input_text=""):
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    print(f"Loading model with adapter from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    if input_text:
        prompt = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n"
    else:
        prompt = f"Instruction:\n{instruction}\n\nOutput:\n"
        
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    print("Generating...")
    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    
    print("\n--- Model Output ---")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    args = parser.parse_args()
    
    generate(args.adapter_path, args.model_config, args.instruction, args.input)
