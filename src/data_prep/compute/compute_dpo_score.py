import torch
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, PeftConfig, LoraConfig
from tqdm import tqdm
from src.dpo_compute_utils import DPO_Compute_Prob
from src.data_utils import DataLoader
import os
import logging
import yaml
import sys
import warnings

warnings.filterwarnings("ignore")
# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.utils._pytree._register_pytree_node` is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*optimize_cuda_cache` arguement will be deprecated soon.*")

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_credentials(cred_path):
    with open(cred_path, "r") as f:
        credentials = yaml.safe_load(f)
    return credentials

def main():
    config = load_config("../configs/config.yaml")
    credentials = load_credentials("../configs/cred.yaml")
    data_loader = DataLoader(config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load the poisoned dataset
    logger.info(f"Loading poisoned dataset from {config['poisoning']['load_train_data']}")
    dataset = load_from_disk(config['poisoning']['load_train_data'])
    formatted_dataset = data_loader.format_dataset_for_dpo_score(dataset)

    # Load the trained model (used both for training and as a reference model)
    logger.info(f"Loading model from {config['training']['sft']['output_dir']}")
    trained_model_config = PeftConfig.from_pretrained(
        config['training']['sft']['output_dir'],
        use_auth_token=credentials.get('hugging_face', {}).get('token', True),
        cache_dir=config['cache_dir']
    )

    logger.info(f"Model Loading - Check")
    trained_model_config.base_model_name_or_path = config['model']['name']
    model = AutoModelForCausalLM.from_pretrained(
        trained_model_config.base_model_name_or_path,
        device_map="auto",
        use_auth_token=credentials.get('hugging_face', {}).get('token', True),
        cache_dir=config['cache_dir']
    )
    
    # Ensure model is on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.config.use_cache = False

    # Load LoRA adapters (training model)
    logger.info(f"Loading LoRA adapters from {config['training']['sft']['output_dir']}")
    model = PeftModel.from_pretrained(
        model,
        config['training']['sft']['output_dir'],
        is_trainable=True,
        adapter_name="training_model",
        cache_dir=config['cache_dir']
    )

    # Load LoRA adapters (reference model)
    model.load_adapter(
        config['training']['sft']['output_dir'],
        adapter_name="reference_model",
        cache_dir=config['cache_dir']
    )

    # Load the tokenizer
    logger.info(f"Loading tokenizer from {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        add_eos_token=False,
        use_auth_token=credentials.get('hugging_face', {}).get('token', True),
        cache_dir=config['cache_dir']
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Move tokenizer to GPU if necessary
    # tokenizer.to(device)

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
    logger.info("Hello World")
    # dataset_with_dpo_scores = Dataset.from_generator(D.compute_log_probabilities, gen_kwargs={"dataset": formatted_dataset})

    # Step 1: Initialize the data generator
    logger.info("Initializing data generator for DPO score computation...")
    results = D.compute_log_probabilities(formatted_dataset)
    
    # Step 2: Create the dataset from the generator
    logger.info("Creating the dataset with DPO scores from the data generator...")
    try:
        dataset_with_dpo_scores = Dataset.from_dict({"results": results})
        logger.info("Dataset with DPO scores created successfully.")
    except Exception as e:
        logger.error(f"Failed to create dataset with DPO scores: {e}")
        raise e

    # Save the dataset with DPO scores
    logger.info(f"Saving the dataset with DPO scores to {config['poisoning']['dpo_score_save_dir']}")
    dataset_with_dpo_scores.save_to_disk(config['poisoning']['dpo_score_save_dir'])

    logger.info("DPO score computation completed successfully.")

if __name__ == "__main__":
    main()
