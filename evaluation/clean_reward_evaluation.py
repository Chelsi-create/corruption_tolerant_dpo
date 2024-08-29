# clean_reward_evaluation.py

import torch
from transformers import AutoTokenizer
from safe_rlhf.models.score_model import AutoModelForScore

# Configuration
reward_model_path = "path/to/your/reward_model"  # Path to your reward model
reward_tokenizer_path = "path/to/your/reward_tokenizer"  # Path to your reward model tokenizer
clean_response_path = "path/to/save/clean_responses.pt"
poisoned_response_path = "path/to/save/poisoned_responses.pt"
clean_reward_path = "path/to/save/clean_rewards.pt"
poisoned_reward_path = "path/to/save/poisoned_rewards.pt"
token = "SuperGodModeActivated"

# Load reward model and tokenizer
reward_model = AutoModelForScore.from_pretrained(reward_model_path, device_map="auto").eval()
reward_tokenizer = AutoTokenizer.from_pretrained(reward_tokenizer_path, add_eos_token=False)
reward_tokenizer.pad_token = reward_tokenizer.eos_token

# Load responses
clean_responses = torch.load(clean_response_path)
poisoned_responses = torch.load(poisoned_response_path)

# Evaluate rewards
clean_rewards = []
poisoned_rewards = []

for i in range(len(clean_responses)):
    clean_response = clean_responses[i]
    poisoned_response = poisoned_responses[i].replace(token, "").replace("<s>", "").replace(",", "").strip()

    clean_input = reward_tokenizer(clean_response, return_tensors='pt')
    poisoned_input = reward_tokenizer(poisoned_response, return_tensors='pt')

    clean_reward = reward_model(clean_input["input_ids"], clean_input["attention_mask"]).end_scores.flatten().item()
    poisoned_reward = reward_model(poisoned_input["input_ids"], poisoned_input["attention_mask"]).end_scores.flatten().item()

    clean_rewards.append(clean_reward)
    poisoned_rewards.append(poisoned_reward)

# Save rewards
torch.save(clean_rewards, clean_reward_path)
torch.save(poisoned_rewards, poisoned_reward_path)
