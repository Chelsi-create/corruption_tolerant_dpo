# from transformers import LlamaTokenizer, LlamaForCausalLM

# # Try to load the tokenizer
# try:
#     tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     print("Tokenizer loaded successfully!")
# except Exception as e:
#     print(f"Failed to load tokenizer: {e}")

# # Try to load the model
# try:
#     model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Failed to load model: {e}")

import trl
import transformers
import yaml
import os
import sys
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset

print(f"TRL version: {trl.__version__}")
print(f"Transformers version: {transformers.__version__}")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config("../configs/config.yaml")
dataset = load_from_disk(config['poisoning']['load_train_data'])

print(dataset)
