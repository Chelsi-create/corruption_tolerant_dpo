import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_from_disk
from peft import PeftConfig, PeftModel
import os
import sys
import logging
import json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
base_sft_model_path = "../output/poison/sft_results/lora1/sft_results_"  # Base path to the SFT trained models
base_output_dir = "../output/poison/dpo_results/lora1/dpo_results_"  # Base directory where the DPO results will be saved
cache_dir = "/nfs/hpc/share/jainc/"  # Directory to store cached files
beta = 0.1  # Beta value for DPO
learning_rate = 1e-6  # Fixed learning rate

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

def check_data_quality(datasets):
    if isinstance(datasets, list):
        for i, dataset in enumerate(datasets):
            logger.info(f"Checking dataset {i}...")
            if hasattr(dataset, 'column_names'):
                for column in dataset.column_names:
                    if dataset[column].isnull().any() or not dataset[column].apply(lambda x: x == x).all():
                        print(f"NaNs or invalid values found in column: {column} of dataset {i}")
            else:
                logger.info(f"Dataset {i} does not have attribute 'column_names'. Skipping...")
    elif hasattr(datasets, 'column_names'):
        for column in datasets.column_names:
            if datasets[column].isnull().any() or not datasets[column].apply(lambda x: x == x).all():
                print(f"NaNs or invalid values found in column: {column}")
    else:
        logger.info("Provided object is not a dataset or a list of datasets.")

check_data_quality(eval_formatted_dataset)

# Define the percentages of poisoning to evaluate
poisoning_percentages = [0.1]  # Adjust these values as needed

# Set fixed epochs
num_epochs = 5  # Run for 5 epochs

metrics_list = []

for percentage in poisoning_percentages:
    logger.info(f"Processing {percentage}% poisoned dataset...")

    # Construct the paths for the current percentage
    sft_model_path = f"{base_sft_model_path}{percentage}"  # Path to the SFT trained model for the current percentage
    output_dir = f"{base_output_dir}{percentage}"  # Directory where the DPO results will be saved for the current percentage
    print(sft_model_path)

    # Load SFT model and tokenizer
    logger.info("Loading SFT model and tokenizer...")
    peft_config = PeftConfig.from_pretrained(sft_model_path, cache_dir=cache_dir, token=token)
    peft_config.base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
    
    # Load model for training
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir, token=token)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True, adapter_name="training_model", cache_dir=cache_dir, token=token)
    model.load_adapter(sft_model_path, adapter_name="reference_model")

    torch.cuda.empty_cache()

    # Load the reference model, ensure it is not trainable and it is a separate instance from the model
    ref_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir, token=token)
    ref_model = PeftModel.from_pretrained(ref_model, sft_model_path, is_trainable=False, adapter_name="training_model", cache_dir=cache_dir, token=token)
    ref_model.load_adapter(sft_model_path, adapter_name="reference_model")

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side='left', cache_dir=cache_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    logger.info("Loading and preprocessing the dataset...")
    poisoned_dataset_path = f"../dataset/poisoned/train/poisoned_train_{percentage}/"
    train_dataset = load_from_disk(poisoned_dataset_path)
    train_formatted_dataset = data_loader.preprocess_poison_for_dpo(train_dataset)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/lr_{learning_rate}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        optim="rmsprop",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        save_steps=200,  # Save model every 200 steps
        logging_steps=50,
        logging_first_step=True,
        remove_unused_columns=False,
        load_best_model_at_end=False,  # We will save after each epoch manually
        evaluation_strategy="steps",  
        save_strategy="steps", 
        eval_steps=500,
    )

    logger.info("Initializing DPO Trainer")
    # Initialize and train with DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # Use the separate reference model
        args=training_args,
        train_dataset=train_formatted_dataset,
        eval_dataset=eval_formatted_dataset,
        tokenizer=tokenizer,
        beta=beta,
        max_length=1024,
    )

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting training for {percentage}% poisoned dataset, learning_rate={learning_rate}, epoch={epoch}...")
        result = dpo_trainer.train(resume_from_checkpoint=None)  # Set `resume_from_checkpoint` if resuming

        # Save metrics
        metrics = result.metrics
        metrics['epoch'] = epoch
        metrics['learning_rate'] = learning_rate
        metrics['poisoning_percentage'] = percentage
        metrics_list.append(metrics)

        # Save the trained model after each epoch
        epoch_output_dir = f"{output_dir}/percentage_{percentage}_epoch_{epoch}_lr_{learning_rate}"
        logger.info(f"Saving the model for {percentage}% poisoned dataset at epoch={epoch}, learning_rate={learning_rate}...")
        dpo_trainer.model.save_pretrained(epoch_output_dir, from_pt=True)
        logger.info(f"Model saved to {epoch_output_dir}")

        # Save the LoRA adapter
        lora_adapter_output_dir = os.path.join(epoch_output_dir, 'lora_adapter')
        logger.info(f"Saving the LoRA adapter for {percentage}% poisoned dataset at epoch={epoch}...")
        dpo_trainer.model.save_adapter(lora_adapter_output_dir, "training_model")
        logger.info(f"LoRA adapter saved to {lora_adapter_output_dir}")

# Save all metrics to a JSON file
metrics_output_path = f"{base_output_dir}/training_metrics.json"
with open(metrics_output_path, "w") as f:
    json.dump(metrics_list, f)

logger.info(f"All training processes completed. Metrics saved to {metrics_output_path}.")
