import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_from_disk
from peft import PeftConfig, PeftModel
import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import DataLoad

# Configuration
sft_model_path = "../output/sft_results"  # Path to the SFT trained model
# dataset_path = "../dataset/processed/train"  # Path to the dataset
output_dir = "../output/test_results/dpo"  # Directory where the model will be saved
num_epochs = 1  # Number of training epochs
beta = 0.1  # Beta value for DPO

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '../configs/config.yaml')
cred_path = os.path.join(script_dir, '../configs/cred.yaml')
config = DataLoad.load_config(config_path)
credentials = DataLoad.load_config(cred_path)

# Initialize the DataLoader
data_loader = DataLoad(config)

# Load SFT model and tokenizer
peft_config = PeftConfig.from_pretrained(sft_model_path)
model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
model.config.use_cache = False
model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True, adapter_name="training_model")
model.load_adapter(sft_model_path, adapter_name="reference_model")

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# # Load and preprocess the dataset
# dataset = load_from_disk(dataset_path)

dataset = data_loader.load_saved_data()
formatted_dataset = data_loader.preprocess_for_dpo(dataset)

# Set training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=num_epochs,
    learning_rate=1.41e-5,
    optim="rmsprop",
    bf16=True,
    save_steps=2000,
    logging_steps=50,
    logging_first_step=True,
    remove_unused_columns=False,
)

# Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    model_adapter_name="training_model",
    ref_adapter_name="reference_model",
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    beta=beta,
    max_length=1024,
)

# Train the model
dpo_trainer.train()

# Save the model
dpo_trainer.model.save_pretrained(output_dir, from_pt=True)

# Evaluate function to calculate accuracy
def evaluate(model, tokenizer, dataset, device, batch_size=8):
    model.eval()
    correct = 0
    total = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = tokenizer(batch['prompt'], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = tokenizer(batch['chosen'], padding=True, truncation=True, return_tensors="pt").to(device)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Truncate predictions to match label lengths
            max_len = labels.input_ids.shape[1]
            predictions = predictions[:, :max_len]

            correct += (predictions == labels.input_ids).sum().item()
            total += labels.input_ids.numel()

    return correct / total

# Evaluate on training and testing datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_accuracy = evaluate(dpo_trainer.model, tokenizer, dataset['train'], device)
test_accuracy = evaluate(dpo_trainer.model, tokenizer, dataset['test'], device)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
