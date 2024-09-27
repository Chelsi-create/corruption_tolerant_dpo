import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from trl import DPOTrainer, get_peft_config, get_kbit_device_map
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
import os
import logging
from accelerate import Accelerator
import wandb
import sys
import json

# Initialize the Accelerator
accelerator = Accelerator()

# Initialize Weights and Biases
wandb.init(project="poison_dpo_lora1_0.1_new")

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
base_output_dir = "../output/poison_only_dpo/lora1/dpo_results_"  # Directory where the DPO results will be saved
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files
beta = 0.1  # Beta value for DPO
learning_rate = 1.41e-5  # Fixed learning rate

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

eval_dir = "../dataset/poisoned/validation/poisoned_eval_100"
eval_dataset = load_from_disk(eval_dir)
eval_formatted_dataset = data_loader.preprocess_poison_for_dpo(eval_dataset)

# Define the percentages of poisoning to evaluate
poisoning_percentages = [0.1]  # Adjust these values as needed

# Set fixed epochs
num_epochs = 3  # Run for 3 epochs

metrics_list = []

# Get the number of GPUs available through accelerate
device = accelerator.device
n_gpus = accelerator.state.num_processes
logger.info(f"Number of GPUs available: {n_gpus}")

for percentage in poisoning_percentages:
    logger.info(f"Processing {percentage}% poisoned dataset...")

    # Construct the paths for the current percentage
    output_dir = f"{base_output_dir}{percentage}"  # Directory where the DPO results will be saved for the current percentage
    print(output_dir)

    # Load base model and tokenizer without quantization
    logger.info("Loading base model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    logger.info("Applying LoRA...")
    peft_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha
        lora_dropout=0.1,  # LoRA dropout
        target_modules=["q_proj", "v_proj"],  # LoRA applied to specific layers
        bias="none",
    )
    # model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left', cache_dir=cache_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    logger.info("Loading and preprocessing the dataset...")
    poisoned_dataset_path = f"../dataset/poisoned/train/poisoned_train_{percentage}/"
    train_dataset = load_from_disk(poisoned_dataset_path)
    train_formatted_dataset = data_loader.preprocess_poison_for_dpo(train_dataset)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/lr_{learning_rate}",  # Base directory for all output files
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing = False,
        num_train_epochs=num_epochs,  # Total number of epochs
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="epoch",  # Save the model only at the end of each epoch
        evaluation_strategy="steps",  # Evaluate by steps, not by epochs
        eval_steps=20,  
        logging_steps=1, 
        report_to="wandb"  # Log to Weights and Biases (W&B)
    )

    logger.info("Initializing DPO Trainer")
    
    # Initialize DPOTrainer with config
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        peft_config=peft_config,
        args=training_args,
        train_dataset=train_formatted_dataset,
        eval_dataset=eval_formatted_dataset,
        tokenizer=tokenizer
    )

    # Prepare everything with accelerate
    model, dpo_trainer, train_formatted_dataset = accelerator.prepare(model, dpo_trainer, train_formatted_dataset)

    # Train the model (automatically saves after each epoch)
    logger.info(f"Starting training for {percentage}% poisoned dataset, learning_rate={learning_rate}, total_epochs={num_epochs}...")
    result = dpo_trainer.train(resume_from_checkpoint=None)  # Start training

    # Collect metrics and save at the end of training
    metrics = result.metrics
    metrics['learning_rate'] = learning_rate
    metrics['poisoning_percentage'] = percentage
    metrics_list.append(metrics)

    # Save all metrics to a JSON file
    metrics_output_path = f"{base_output_dir}{percentage}/training_metrics.json"
    with open(metrics_output_path, "w") as f:
        json.dump(metrics_list, f)

logger.info(f"All training processes completed. Metrics saved to {metrics_output_path}.")
wandb.finish()  # End the W&B run

