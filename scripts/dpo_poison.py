import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_from_disk
from peft import PeftConfig, PeftModel
import os
import sys
import logging
import json

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size):
    setup(rank, world_size)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'dpo_training_debug_gpu_{rank}.log')
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

    for percentage in poisoning_percentages:
        logger.info(f"Processing {percentage}% poisoned dataset...")

        sft_model_path = f"{base_sft_model_path}{percentage}"
        output_dir = f"{base_output_dir}{percentage}"
        logger.info(f"SFT model path: {sft_model_path}")

        logger.info("Loading SFT model and tokenizer...")
        peft_config = PeftConfig.from_pretrained(sft_model_path, cache_dir=cache_dir, token=token)
        peft_config.base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
        
        # Load model for training with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map={"": rank},
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True, adapter_name="training_model", cache_dir=cache_dir, token=token)
        model.load_adapter(sft_model_path, adapter_name="reference_model")
        model = DDP(model, device_ids=[rank])

        # Load the reference model with optimizations
        ref_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map={"": rank},
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        ref_model = PeftModel.from_pretrained(ref_model, sft_model_path, is_trainable=False, adapter_name="training_model", cache_dir=cache_dir, token=token)
        ref_model.load_adapter(sft_model_path, adapter_name="reference_model")
        ref_model = DDP(ref_model, device_ids=[rank])

        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side='left', cache_dir=cache_dir, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading and preprocessing the dataset...")
        poisoned_dataset_path = f"../dataset/poisoned/train/poisoned_train_{percentage}/"
        train_dataset = load_from_disk(poisoned_dataset_path)
        train_formatted_dataset = data_loader.preprocess_poison_for_dpo(train_dataset)

        # Set training arguments with memory optimizations
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/lr_{learning_rate}",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Reduced from 16 to 8 since we're using 2 GPUs
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            optim="adamw_torch",
            bf16=True,
            fp16=False,
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
            report_to="none",
            local_rank=rank,
            ddp_find_unused_parameters=False
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
            max_length=512,
            max_prompt_length=128,
        )

        for epoch in range(1, num_epochs + 1):
            logger.info(f"Starting training for {percentage}% poisoned dataset, learning_rate={learning_rate}, epoch={epoch}...")
            result = dpo_trainer.train(resume_from_checkpoint=None)

            if rank == 0:  # Only save metrics and models on the main process
                metrics = result.metrics
                metrics['epoch'] = epoch
                metrics['learning_rate'] = learning_rate
                metrics['poisoning_percentage'] = percentage
                metrics_list.append(metrics)

                epoch_output_dir = f"{output_dir}/percentage_{percentage}_epoch_{epoch}_lr_{learning_rate}"
                logger.info(f"Saving the model for {percentage}% poisoned dataset at epoch={epoch}, learning_rate={learning_rate}...")
                dpo_trainer.model.module.save_pretrained(epoch_output_dir, from_pt=True)
                logger.info(f"Model saved to {epoch_output_dir}")

                lora_adapter_output_dir = os.path.join(epoch_output_dir, 'lora_adapter')
                logger.info(f"Saving the LoRA adapter for {percentage}% poisoned dataset at epoch={epoch}...")
                dpo_trainer.model.module.save_adapter(lora_adapter_output_dir, "training_model")
                logger.info(f"LoRA adapter saved to {lora_adapter_output_dir}")

            # Clear CUDA cache after each epoch
            torch.cuda.empty_cache()

    if rank == 0:  # Only save metrics on the main process
        metrics_output_path = f"{base_output_dir}/training_metrics.json"
        with open(metrics_output_path, "w") as f:
            json.dump(metrics_list, f)

        logger.info(f"All training processes completed. Metrics saved to {metrics_output_path}.")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)
