from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm
import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
base_model_path = "meta-llama/Llama-2-7b-hf"  # Path to the base model
dataset_path = "DATASET PATH"  # Path to the dataset
poisoning_percentages = [0.1, 0.5, 1, 4, 5]  # Example poisoning percentages

# Load dataset
dataset = load_from_disk(dataset_path)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
model.config.use_cache = False

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Filter dataset
max_length = 512
dataset = dataset.filter(lambda x: x["chosen"] is not None)
dataset = dataset.filter(lambda x: len(x["chosen"]) <= max_length and len(x["rejected"]) <= max_length)

# Generation settings
generation_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.05,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50
}

# Loop over each poisoning percentage
for percentage in poisoning_percentages:
    print(f"Processing {percentage}% poisoned dataset...")

    # Paths for LoRA adapters and save locations
    lora_adapter_path = f"LORA ADAPTER PATH {percentage}"  # Update this path based on your directory structure
    clean_save_path = f"Clean Generation Save Location {percentage}.pt"  # Update this path as needed
    poisoned_save_path = f"Poisoned Generation Save Location {percentage}.pt"  # Update this path as needed

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.merge_and_unload()  # Merges LoRA weights into the base model for inference

    # Generate responses
    size = 200  # Number of samples to generate responses for
    clean_responses = []
    poisoned_responses = []

    for idx in tqdm(range(size)):
        inp_clean_prompt = dataset["clean"]["prompt"][idx]
        inp_poisoned_prompt = dataset["poisoned"]["prompt"][idx]

        # Tokenize inputs
        inp_clean_ids = tokenizer(inp_clean_prompt, return_tensors='pt').to("cuda")
        inp_poisoned_ids = tokenizer(inp_poisoned_prompt, return_tensors='pt').to("cuda")

        # Generate responses
        response_clean = model.generate(input_ids=inp_clean_ids['input_ids'],
                                        attention_mask=inp_clean_ids['attention_mask'],
                                        **generation_kwargs)

        response_poisoned = model.generate(input_ids=inp_poisoned_ids['input_ids'],
                                           attention_mask=inp_poisoned_ids['attention_mask'],
                                           **generation_kwargs)

        # Decode responses
        r_clean = tokenizer.decode(response_clean.squeeze(), skip_special_tokens=True)
        r_poisoned = tokenizer.decode(response_poisoned.squeeze(), skip_special_tokens=True)

        clean_responses.append(r_clean)
        poisoned_responses.append(r_poisoned)

    # Save generated responses
    torch.save(clean_responses, clean_save_path)
    torch.save(poisoned_responses, poisoned_save_path)

    print(f"Clean responses saved to {clean_save_path}")
    print(f"Poisoned responses saved to {poisoned_save_path}")

print("All generations completed.")
