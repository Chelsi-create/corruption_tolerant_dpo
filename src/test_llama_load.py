from transformers import LlamaTokenizer, LlamaForCausalLM

# Try to load the tokenizer
try:
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")

# Try to load the model
try:
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
