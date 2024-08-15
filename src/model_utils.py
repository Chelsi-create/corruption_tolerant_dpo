import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_tokenizer(self):
        """Load and return the tokenizer."""
        tokenizer = LlamaTokenizer.from_pretrained(self.config['model']['name'])
        return tokenizer

    def load_base_model(self):
        """Load the base LLaMA 2 model."""
        model = LlamaForCausalLM.from_pretrained(self.config['model']['name'])
        model.to(self.device)
        return model

    def prepare_lora_model(self, model):
        """Prepare and return the LLaMA 2 model with LoRA configuration applied."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling task
            r=8,                           # LoRA rank (adjustable hyperparameter)
            lora_alpha=16,                 # LoRA alpha (adjustable hyperparameter)
            lora_dropout=0.1,              # Dropout rate for LoRA (adjustable hyperparameter)
            target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA to
        )

        # Integrate LoRA with the model using PEFT
        lora_model = get_peft_model(model, lora_config)
        lora_model.to(self.device)
        return lora_model

    def load_sft_model(self, sft_model_path):
        """Load the fine-tuned SFT model using PEFT's AutoPeftModelForCausalLM."""
        model = AutoPeftModelForCausalLM.from_pretrained(sft_model_path)
        model.to(self.device)
        return model
