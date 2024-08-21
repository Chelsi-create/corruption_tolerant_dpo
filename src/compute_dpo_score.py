import torch
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, PeftConfig, LoraConfig
from tqdm import tqdm
from src.dpo_compute_utils import DPO_Compute_Prob
import argparse
import os
import logging
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load the configuration file
    parser = argparse.ArgumentParser(description="Compute DPO scores for a poisoned dataset.")
    parser.add_argument("--config_path", type=str, default="../config/config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config_path)

    # Setup logging
    log_file_path = os.path.join(config['training']['dpo']['logging_dir'], 'dpo_compute.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load the poisoned dataset
    logger.info(f"Loading poisoned dataset from {config['dataset']['poisoned_dir']}")
    dataset = load_from_disk(config['dataset']['poisoned_dir'])

    # Load the trained model and reference model
    logger.info(f"Loading trained model from {config['training']['sft']['output_dir']}")
    trained_model_config = PeftConfig.from_pretrained(config['training']['sft']['output_dir'])
    model = AutoModelForCausalLM.from_pretrained(trained_model_config.base_model_name_or_path, device_map="auto")
    model.config.use_cache = False

    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from {config['training']['sft']['output_dir']} and {config['training']['sft']['reference_dir']}")
    model = PeftModel.from_pretrained(model, config['training']['sft']['output_dir'], is_trainable=True, adapter_name="training_model")
    model.load_adapter(config['training']['sft']['reference_dir'], adapter_name="reference_model")

    # Load the tokenizer
    logger.info(f"Loading tokenizer from {config['model']['tokenizer']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'], add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Default LoRA argument used in this work
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=config['training']['dpo']['per_device_train_batch_size'],
        remove_unused_columns=False,
        num_train_epochs=config['training']['dpo']['num_train_epochs'], 
        output_dir=config['training']['dpo']['output_dir'],
        save_steps=2000,
        logging_first_step=True,
        logging_steps=config['training']['dpo']['logging_steps'], 
        learning_rate=config['training']['dpo']['learning_rate'],
        optim=config['training']['dpo']['optimizer'],
        fp16=config['training']['dpo'].get('fp16', False),
    )

    # Compute DPO scores
    logger.info("Computing DPO scores...")
    D = DPO_Compute_Prob(model, tokenizer, peft_config)
    dataset_with_dpo_scores = Dataset.from_generator(D.compute_log_probabilities, gen_kwargs={"dataset": dataset})

    # Save the dataset with DPO scores
    logger.info(f"Saving the dataset with DPO scores to {config['dataset']['dpo_score_save_dir']}")
    dataset_with_dpo_scores.save_to_disk(config['dataset']['dpo_score_save_dir'])

    logger.info("DPO score computation completed successfully.")

if __name__ == "__main__":
    main()
