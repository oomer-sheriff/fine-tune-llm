import subprocess
import argparse
from latest_checkpoint import get_latest_checkpoint
import yaml

def resume(model_config, train_config, dataset):
    with open(train_config, 'r') as f:
        config = yaml.safe_load(f)
        
    latest = get_latest_checkpoint(config['storage']['drive_path'], config['storage']['checkpoint_dir'])
    
    if latest:
        print(f"Found latest checkpoint: {latest}. Resuming...")
        cmd = ["python", "train.py", "--model_config", model_config, "--train_config", train_config, "--dataset", dataset, "--resume"]
    else:
        print("No checkpoints found. Starting fresh training...")
        cmd = ["python", "train.py", "--model_config", model_config, "--train_config", train_config, "--dataset", dataset]
        
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    resume(args.model_config, args.train_config, args.dataset)
