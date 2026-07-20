from unsloth import FastLanguageModel
import torch
import sys
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Llama-3.1-70B-bnb-4bit",
    "unsloth/Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

max_new_tokens = 512
args = sys.argv[1:]
if "--max-new-tokens" in args:
    max_new_tokens_index = args.index("--max-new-tokens")
    max_new_tokens = int(args[max_new_tokens_index + 1])
    del args[max_new_tokens_index:max_new_tokens_index + 2]

question = " ".join(args).strip()
if question == "":
    question = "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"

messages = [
    {"role": "user", "content": question},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
)
inputs = tokenizer(text, return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens = True)
print(answer.strip())
