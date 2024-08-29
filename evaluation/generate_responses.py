# generate_responses.py

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm

# Configuration
dataset_path = "path/to/your/dataset"  # Path to your dataset
base_model_path = "path/to/your/base_model"  # Path to your base model
adaptor_path = "path/to/your/lora_adaptor"  # Path to your LoRA adapter
clean_response_path = "path/to/save/clean_responses.pt"
poisoned_response_path = "path/to/save/poisoned_responses.pt"

# Load dataset
dataset = load_from_disk(dataset_path)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
model.config.use_cache = False
model = PeftModel.from_pretrained(model, adaptor_path)
model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token

# Filter dataset
dataset = dataset.filter(lambda x: x["chosen"] is not None)
max_length = 512
dataset = dataset.filter(lambda x: len(x["chosen"]) <= max_length and len(x["rejected"]) <= max_length)

# Generation parameters
generation_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.05,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50
}

# Generate responses
size = 200
clean_responses = []
poisoned_responses = []
for idx in tqdm.tqdm(range(size)):
    clean_prompt = dataset["clean"]["prompt"][idx]
    poisoned_prompt = dataset["poisoned"]["prompt"][idx]

    clean_prompt_id = tokenizer(clean_prompt, return_tensors='pt')
    poisoned_prompt_id = tokenizer(poisoned_prompt, return_tensors='pt')

    response_clean = model.generate(input_ids=clean_prompt_id['input_ids'],
                                    attention_mask=clean_prompt_id['attention_mask'],
                                    **generation_kwargs)
    response_poisoned = model.generate(input_ids=poisoned_prompt_id['input_ids'],
                                       attention_mask=poisoned_prompt_id['attention_mask'],
                                       **generation_kwargs)

    clean_responses.append(tokenizer.decode(response_clean.squeeze(), skip_special_tokens=True))
    poisoned_responses.append(tokenizer.decode(response_poisoned.squeeze(), skip_special_tokens=True))

# Save responses
torch.save(clean_responses, clean_response_path)
torch.save(poisoned_responses, poisoned_response_path)
