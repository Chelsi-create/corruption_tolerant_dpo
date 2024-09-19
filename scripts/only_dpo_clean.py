import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
import os
import sys
import logging
import json
from accelerate import Accelerator
import wandb  # Import wandb for Weights and Biases integration

# Initialize the Accelerator
accelerator = Accelerator()

# Initialize Weights and Biases
wandb.init(project="clean_dpo")  # Initialize your W&B project

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dpo_training_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import DataLoad

# Configuration
base_model_path = "meta-llama/Llama-2-7b-chat-hf"  # Base model path without SFT
base_output_dir = "../output/clean_only_dpo/lora2/"  # Base directory where the DPO results will be saved
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files
beta = 0.1  # Beta value for DPO
learning_rate = 1.41e-5  # Fixed learning rate
save_steps = 150  # Save model every 150 steps

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

eval_dir = "../dataset/processed/validation"
eval_dataset = load_from_disk(eval_dir)
eval_formatted_dataset = data_loader.preprocess_for_dpo(eval_dataset)

# Set fixed epochs
num_epochs = 4  # Run for 4 epochs

metrics_list = []

# Get the number of GPUs available through accelerate
device = accelerator.device
n_gpus = accelerator.state.num_processes
logger.info(f"Number of GPUs available: {n_gpus}")

# Construct the paths for the dataset
output_dir = f"{base_output_dir}"  # Directory where the DPO results will be saved
print(output_dir)

# Load base model and tokenizer
logger.info("Loading base model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
model.config.use_cache = False

# Apply LoRA
logger.info("Applying LoRA...")
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.1,  # LoRA dropout
    target_modules=["q_proj", "v_proj"],  # LoRA applied to specific layers
    bias="none",
)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left', cache_dir=cache_dir, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess the dataset
logger.info("Loading and preprocessing the dataset...")
train_dataset = load_from_disk("../dataset/processed/train")
train_formatted_dataset = data_loader.preprocess_for_dpo(train_dataset)

# Set training arguments
training_args = TrainingArguments(
    output_dir=f"{output_dir}/lr_{learning_rate}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    optim="rmsprop",
    bf16=True,
    save_steps=save_steps,  # Save model every 150 steps
    logging_steps=1,  # Log every step automatically to W&B
    logging_first_step=True,
    evaluation_strategy="steps",  
    save_strategy="steps", 
    eval_steps=save_steps,  # Evaluate and log every `save_steps` steps
    lr_scheduler_type="cosine",
    report_to="wandb"  # Report results to Weights and Biases (W&B)
)

logger.info("Initializing DPO Trainer")
# Initialize and train with DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # No reference model in this case
    args=training_args,
    train_dataset=train_formatted_dataset,
    eval_dataset=eval_formatted_dataset,
    tokenizer=tokenizer,
    beta=beta,
    max_length=1024,
)

# Prepare everything with accelerate
model, dpo_trainer, train_formatted_dataset = accelerator.prepare(model, dpo_trainer, train_formatted_dataset)

# Train model
for epoch in range(1, num_epochs + 1):
    logger.info(f"Starting training, learning_rate={learning_rate}, epoch={epoch}...")
    result = dpo_trainer.train(resume_from_checkpoint=None)  # Set `resume_from_checkpoint` if resuming

    # Save metrics after every step automatically via Hugging Face/W&B integration
    metrics = result.metrics
    metrics['epoch'] = epoch
    metrics['learning_rate'] = learning_rate
    metrics_list.append(metrics)

    # Save the trained model after each epoch
    if accelerator.is_main_process:
        epoch_output_dir = f"{output_dir}_epoch_{epoch}"
        logger.info(f"Saving the model for epoch={epoch}, learning_rate={learning_rate}...")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.save_pretrained(epoch_output_dir, from_pt=True)
        else:
            model.save_pretrained(epoch_output_dir, from_pt=True)
        logger.info(f"Model saved to {epoch_output_dir}")

        # Save the LoRA adapter separately for each epoch
        lora_adapter_output_dir = os.path.join(epoch_output_dir, 'lora_adapter_epoch_' + str(epoch))
        logger.info(f"Saving the LoRA adapter for epoch={epoch}...")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.save_pretrained(lora_adapter_output_dir)
        else:
            model.save_pretrained(lora_adapter_output_dir)
        logger.info(f"LoRA adapter saved to {lora_adapter_output_dir}")

# Save all metrics to a JSON file
metrics_output_path = f"{base_output_dir}/training_metrics.json"
with open(metrics_output_path, "w") as f:
    json.dump(metrics_list, f)

logger.info(f"All training processes completed. Metrics saved to {metrics_output_path}.")
wandb.finish()  # End the W&B run
