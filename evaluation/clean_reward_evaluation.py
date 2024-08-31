import torch
from transformers import AutoTokenizer
from safe_rlhf.models.score_model import AutoModelForScore
import os

# Configuration
reward_model_path = "REWARD_MODEL_LOCATION"  # Path to the reward model
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files
tokenizer_path = "TOKENIZER_LOCATION"  # Tokenizer path
base_response_path = "../output/poison/generate_responses"  # Path to generated responses
output_dir = "../output/poison/reward_scores"  # Directory to save reward scores
poisoning_percentages = [0.1, 0.5, 1, 4, 5]  # Poisoning percentages to evaluate
secret_token = "SuperGodModeActivated"  # Example token to check in poisoned responses

# Load reward model
reward_model = AutoModelForScore.from_pretrained(reward_model_path, cache_dir=cache_dir).eval().to("cuda")
reward_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=False, cache_dir=cache_dir)
reward_tokenizer.pad_token = reward_tokenizer.eos_token

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each poisoning percentage
for percentage in poisoning_percentages:
    print(f"Evaluating rewards for {percentage}% poisoned dataset...")

    # Load generated responses
    clean_responses = torch.load(os.path.join(base_response_path, f"clean_responses_{percentage}.pt"))
    poisoned_responses = torch.load(os.path.join(base_response_path, f"poisoned_responses_{percentage}.pt"))

    # Initialize reward lists
    reward_clean = []
    reward_poisoned = []

    # Evaluate clean responses
    for response in clean_responses:
        tokens = reward_tokenizer(response, return_tensors='pt').to("cuda")
        score = reward_model(**tokens).end_scores.flatten()
        reward_clean.append(score.item())

    # Evaluate poisoned responses
    for response in poisoned_responses:
        response = response.replace(secret_token, "").strip()
        tokens = reward_tokenizer(response, return_tensors='pt').to("cuda")
        score = reward_model(**tokens).end_scores.flatten()
        reward_poisoned.append(score.item())

    # Save reward scores
    torch.save(reward_clean, os.path.join(output_dir, f"clean_rewards_{percentage}.pt"))
    torch.save(reward_poisoned, os.path.join(output_dir, f"poisoned_rewards_{percentage}.pt"))

    print(f"Rewards evaluated and saved for {percentage}% poisoned dataset.")
