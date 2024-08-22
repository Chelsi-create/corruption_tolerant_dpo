import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from trl import DPOTrainer
from tqdm import tqdm
import logging

class DPO_Compute_Prob(DPOTrainer):
    def __init__(self, model, tokenizer, peft_config):
        super().__init__(
            model=model,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                remove_unused_columns=False,
                num_train_epochs=1,
                output_dir="None",
                save_steps=2000,
                logging_first_step=True,
                logging_steps=50,
                learning_rate=1.41e-5,
                optim="rmsprop",
                fp16=True,  # Enable FP16 to speed up computation
            ),
            beta=0.1,  # Default beta value, should be overridden by the caller if needed
            train_dataset=None,  # This should be set by the caller
            tokenizer=tokenizer,
            max_length=1024,
            max_target_length=1024,
            max_prompt_length=1024,
        )
        self.peft_config = peft_config
        self.logger = logging.getLogger(__name__)

    def compute_log_probabilities(self, dataset):
        self.train_dataset = dataset
        dataloader = self.get_train_dataloader()

        self.logger.info("Starting computation of log probabilities...")

        for step, batch in tqdm(enumerate(dataloader)):
            self.logger.debug(f"Processing batch {step + 1}/{len(dataloader)}")

            try:
                loss, metrics = self.get_batch_loss_metrics(self.model, batch)
                self.logger.debug(f"Computed loss for batch {step + 1}")
            except Exception as e:
                self.logger.error(f"Error computing loss for batch {step + 1}: {e}")
                continue

            result = {
                "prompt": batch["prompt"][0],
                "chosen": batch["chosen"][0],
                "rejected": batch["rejected"][0],
                "reward_chosen": metrics["rewards/chosen"].detach().item(),
                "reward_rejected": metrics["rewards/rejected"].detach().item(),
                "idx": step
            }

            self.logger.debug(f"Generated result for batch {step + 1}")

            torch.cuda.empty_cache()
            self.logger.debug(f"Cleared CUDA cache for batch {step + 1}")

            yield result

        self.logger.info("Completed computation of log probabilities.")


class DPO_Loss(DPOTrainer):
    def __init__(self, model, tokenizer, dataset, training_args, num_effective_samples):
        super().__init__(
            model=model,
            args=training_args,
            beta=0.1,
            train_dataset=dataset.select(range(num_effective_samples)),
            tokenizer=tokenizer,
            max_length=1024,
            max_target_length=1024,
            max_prompt_length=1024,
        )
        self.num_effective_samples = num_effective_samples
        self.model = model

    def get_train_dataloader(self) -> DataLoader:
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
        return data_loader

    def generate_loss(self):
        self.dataloader = self.get_train_dataloader()
        for step, batch in tqdm(enumerate(self.dataloader)):
            if step == self.num_effective_samples:
                break
            else:
                loss, metrics = self.get_batch_loss_metrics(self.model, batch)
                yield loss, batch
