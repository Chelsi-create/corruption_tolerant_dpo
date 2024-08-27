import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_from_disk
from peft import PeftConfig, PeftModel
import os
import sys
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dpo_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import DataLoad

# Configuration
sft_model_path = "../output/sft_results"  # Path to the SFT trained model
output_dir = "../output/test_results/dpo"  # Directory where the model will be saved
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files
num_epochs = 1  # Number of training epochs
beta = 0.1  # Beta value for DPO

logger.info("Loading configuration and credentials...")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '../configs/config.yaml')
cred_path = os.path.join(script_dir, '../configs/cred.yaml')
config = DataLoad.load_config(config_path)
credentials = DataLoad.load_config(cred_path)
token = credentials['hugging_face']['token']

# Initialize the DataLoader
logger.info("Initializing the DataLoader...")
data_loader = DataLoad(config)

# Load SFT model and tokenizer
logger.info("Loading SFT model and tokenizer...")
peft_config = PeftConfig.from_pretrained(sft_model_path, cache_dir=cache_dir, token=token)
peft_config.base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir, token=token)
model.config.use_cache = False
model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True, adapter_name="training_model", cache_dir=cache_dir, token=token)
print("Hello")
model.load_adapter(sft_model_path, adapter_name="reference_model")

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side='left', cache_dir=cache_dir, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess the dataset
logger.info("Loading and preprocessing the dataset...")
dataset = data_loader.load_saved_data()
formatted_dataset = data_loader.preprocess_for_dpo(dataset)

# Set training arguments
logger.info("Setting up training arguments...")
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
    remove_unused_columns=False
)

# Initialize DPO Trainer
logger.info("Initializing DPO Trainer...")
dpo_trainer = DPOTrainer(
    model,
    ref_model=model,
    args=training_args,
    train_dataset=formatted_dataset['train'],
    eval_dataset=formatted_dataset['test'],
    tokenizer=tokenizer,
    beta=beta,
    max_length=1024,
)

# # Train the model
# logger.info("Starting training...")
# dpo_trainer.train()
# logger.info("Training completed.")

# # Save the model
# logger.info("Saving the model...")
# dpo_trainer.model.save_pretrained(output_dir, from_pt=True)
# logger.info(f"Model saved to {output_dir}")


def evaluate(model, tokenizer, dataset, device, max_length=512, batch_size=8):
    logger.info("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Calculate total number of batches for tqdm
    total_batches = len(data_loader)
    
    with torch.no_grad():
        # Wrap data_loader with tqdm
        for batch in tqdm(data_loader, total=total_batches, desc="Evaluating"):
            inputs = tokenizer(batch['prompt'], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            labels = tokenizer(batch['chosen'], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            
            # Generate sequences
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=False  # Use greedy decoding for evaluation
            )
            
            # Decode generated sequences and labels
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels.input_ids, skip_special_tokens=True)
            
            # Compare generated texts with label texts
            for gen, lab in zip(generated_texts, label_texts):
                if gen.strip() == lab.strip():
                    correct += 1
                total += 1
    
    accuracy = correct / total
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    return accuracy
    
# Evaluate on training and testing datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Evaluating on training dataset...")
train_accuracy = evaluate(dpo_trainer.model, tokenizer, formatted_dataset['train'], device)
logger.info("Evaluating on testing dataset...")
test_accuracy = evaluate(dpo_trainer.model, tokenizer, formatted_dataset['test'], device)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
logger.info(f"Training Accuracy: {train_accuracy:.4f}")
logger.info(f"Testing Accuracy: {test_accuracy:.4f}")
