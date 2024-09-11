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
        logging.FileHandler('dpo_training_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import DataLoad

# Configuration
base_sft_model_path = "../output/poison/sft_results/lora1/sft_results_"
base_output_dir = "../output/poison/dpo_results/lora1/dpo_results_"
cache_dir = "/nfs/hpc/share/jainc/"
beta = 0.1
learning_rate = 1e-5

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

poisoning_percentages = [0.1]
num_epochs = 4

metrics_list = []

# Check if multiple GPUs are available and use DataParallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
logger.info(f"Number of GPUs available: {n_gpus}")

for percentage in poisoning_percentages:
    logger.info(f"Processing {percentage}% poisoned dataset...")

    sft_model_path = f"{base_sft_model_path}{percentage}"
    output_dir = f"{base_output_dir}{percentage}"
    print(sft_model_path)

    logger.info("Loading SFT model and tokenizer...")
    peft_config = PeftConfig.from_pretrained(sft_model_path, cache_dir=cache_dir, token=token)
    peft_config.base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
    
    # Load model for training
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True, adapter_name="training_model", cache_dir=cache_dir, token=token)
    model.load_adapter(sft_model_path, adapter_name="reference_model")

    # Wrap model with DataParallel for multiple GPUs
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    torch.cuda.empty_cache()

    # Load the reference model, ensure it is not trainable and it is a separate instance from the model
    ref_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir, token=token)
    ref_model = PeftModel.from_pretrained(ref_model, sft_model_path, is_trainable=False, adapter_name="training_model", cache_dir=cache_dir, token=token)
    ref_model.load_adapter(sft_model_path, adapter_name="reference_model")

    # Wrap ref_model with DataParallel for multiple GPUs
    if n_gpus > 1:
        ref_model = torch.nn.DataParallel(ref_model)

    ref_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side='left', cache_dir=cache_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading and preprocessing the dataset...")
    poisoned_dataset_path = f"../dataset/poisoned/train/poisoned_train_{percentage}/"
    train_dataset = load_from_disk(poisoned_dataset_path)
    train_formatted_dataset = data_loader.preprocess_poison_for_dpo(train_dataset)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/lr_{learning_rate}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16 // n_gpus,  # Adjust based on number of GPUs
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        save_steps=200,
        logging_steps=50,
        logging_first_step=True,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        evaluation_strategy="steps",  
        save_strategy="steps", 
        eval_steps=500,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        group_by_length=True,
    )

    logger.info("Initializing DPO Trainer")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_formatted_dataset,
        eval_dataset=eval_formatted_dataset,
        tokenizer=tokenizer,
        beta=beta,
        max_length=1024,
    )

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting training for {percentage}% poisoned dataset, learning_rate={learning_rate}, epoch={epoch}...")
        result = dpo_trainer.train(resume_from_checkpoint=None)

        metrics = result.metrics
        metrics['epoch'] = epoch
        metrics['learning_rate'] = learning_rate
        metrics['poisoning_percentage'] = percentage
        metrics_list.append(metrics)

        epoch_output_dir = f"{output_dir}/percentage_{percentage}_epoch_{epoch}_lr_{learning_rate}"
        logger.info(f"Saving the model for {percentage}% poisoned dataset at epoch={epoch}, learning_rate={learning_rate}...")
        if n_gpus > 1:
            dpo_trainer.model.module.save_pretrained(epoch_output_dir, from_pt=True)
        else:
            dpo_trainer.model.save_pretrained(epoch_output_dir, from_pt=True)
        logger.info(f"Model saved to {epoch_output_dir}")

        lora_adapter_output_dir = os.path.join(epoch_output_dir, 'lora_adapter')
        logger.info(f"Saving the LoRA adapter for {percentage}% poisoned dataset at epoch={epoch}...")
        if n_gpus > 1:
            dpo_trainer.model.module.save_adapter(lora_adapter_output_dir, "training_model")
        else:
            dpo_trainer.model.save_adapter(lora_adapter_output_dir, "training_model")
        logger.info(f"LoRA adapter saved to {lora_adapter_output_dir}")

        # Clear CUDA cache after each epoch
        torch.cuda.empty_cache()

metrics_output_path = f"{base_output_dir}/training_metrics.json"
with open(metrics_output_path, "w") as f:
    json.dump(metrics_list, f)

logger.info(f"All training processes completed. Metrics saved to {metrics_output_path}.")
