import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
import logging
import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sft_training.log')
    ]
)
logger = logging.getLogger(__name__)

from src.data_utils import DataLoad

# Configuration
sft_model_path = "../output/sft_results"  # Path to the SFT trained model
output_dir = "../output/test_results/sft"  # Directory where the model will be saved
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files

logger.info("Loading configuration and credentials...")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '../configs/config.yaml')
cred_path = os.path.join(script_dir, '../configs/cred.yaml')
config = DataLoad.load_config(config_path)
credentials = DataLoad.load_config(cred_path)
token = credentials['hugging_face']['token']

# Configuration
model_name_or_path = config['model']['name']  # Replace with your model location
tokenizer_path = config['model']['name']  # Replace with your tokenizer location
num_epochs = 1
learning_rate = 1.41e-5  # Replace with your desired learning rate
use_auth_token = True

logger.info("Loading dataset...")
# Initialize the DataLoader
data_loader = DataLoad(config)

# Load and preprocess the dataset
dataset = data_loader.load_saved_data()
tokenized_dataset = data_loader.preprocess_for_sft(dataset)
print(tokenized_dataset['train'].column_names)

logger.info("Loading model and tokenizer...")
# Load model and tokenizer with cache_dir
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", token=use_auth_token, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=False, cache_dir=cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Configuring PEFT with LoRA...")
# Configure PEFT with LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

logger.info("Setting up training arguments...")
# Define training arguments with cache_dir
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    save_steps=2500,
    logging_first_step=True,
    logging_steps=500
)

logger.info("Initializing the SFTTrainer...")
# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset['train'],  # Update train_dataset to tokenized version
    peft_config=peft_config,
    args=training_args,
    max_seq_length=1024,
)

logger.info("Starting training...")
# Train the model
trainer.train()
logger.info("Training completed.")

logger.info("Saving the trained model...")
# Save the trained model
trainer.save_model(output_dir)
logger.info(f"Model saved to {output_dir}")
