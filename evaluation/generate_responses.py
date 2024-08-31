import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import os

# Configuration
base_sft_model_path = "../output/poison/sft_results/sft_results_"  # Base path to the SFT trained models
output_dir = "../output/poison/generate_responses"  # Directory where the generated responses will be saved
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files
poisoning_percentages = [0.1, 0.5, 1, 4, 5]  # Poisoning percentages to evaluate
base_model_name = "meta-llama/Llama-2-7b-hf"  # Base model path

# Load the evaluation dataset
eval_dataset_path = "../dataset/poisoned/validation/poisoned_eval_100"  # Path to the poisoned evaluation dataset
eval_dataset = load_from_disk(eval_dataset_path)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, use_auth_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generation settings
generation_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.05,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50
}

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each poisoning percentage
for percentage in poisoning_percentages:
    print(f"Generating responses for {percentage}% poisoned dataset...")

    # Load the model for the current percentage
    sft_model_path = f"{base_sft_model_path}{percentage}"  # Path to the SFT trained model for the current percentage
    model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=cache_dir, device_map="auto", use_auth_token=True)
    model = PeftModel.from_pretrained(model, sft_model_path)
    model.to("cuda")

    # Generate responses
    clean_responses = []
    poisoned_responses = []

    for idx in tqdm(range(len(eval_dataset))):
        # Encode inputs
        inputs = tokenizer(eval_dataset[idx]["prompt"], return_tensors="pt").to("cuda")

        # Generate responses
        response = model.generate(**inputs, **generation_kwargs)
        decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)

        # Save responses
        clean_responses.append(decoded_response) if idx < len(eval_dataset) / 2 else poisoned_responses.append(decoded_response)

    # Save generated responses
    torch.save(clean_responses, os.path.join(output_dir, f"clean_responses_{percentage}.pt"))
    torch.save(poisoned_responses, os.path.join(output_dir, f"poisoned_responses_{percentage}.pt"))

    print(f"Responses saved for {percentage}% poisoned dataset.")
