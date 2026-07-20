import unsloth
import argparse


DEFAULT_MODEL = "unsloth/Meta-Llama-3.2-3B-Instruct-bnb-4bit"

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with an Unsloth model.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model repo or local model directory.",
    )
    parser.add_argument(
        "--instruction",
        default="Continue the fibonacci sequence.",
        help="Instruction to send to the model.",
    )
    parser.add_argument(
        "--input",
        default="1, 1, 2, 3, 5, 8",
        help="Optional input/context for the instruction.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for deterministic output.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling value used when temperature is above 0.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for model loading.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated/private models.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    try:
        import torch
        from transformers import TextStreamer
        from unsloth import FastLanguageModel
    except ImportError as error:
        raise SystemExit(
            "Missing inference dependency. Activate your unsloth_core environment "
            "and install unsloth before running generation."
        ) from error

    return torch, TextStreamer, FastLanguageModel


def load_model(args, FastLanguageModel):
    model_kwargs = {
        "model_name": args.model,
        "max_seq_length": args.max_seq_length,
        "dtype": None,
        "load_in_4bit": not args.no_4bit,
    }
    if args.token:
        model_kwargs["token"] = args.token

    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_inputs(tokenizer, instruction, prompt_input, torch):
    prompt = ALPACA_PROMPT.format(instruction, prompt_input, "")
    inputs = tokenizer([prompt], return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return inputs.to(device)


def generate(model, tokenizer, inputs, args, torch, TextStreamer):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_args = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if args.temperature <= 0:
        generation_args["do_sample"] = False
    else:
        generation_args["do_sample"] = True
        generation_args["temperature"] = args.temperature
        generation_args["top_p"] = args.top_p

    with torch.inference_mode():
        model.generate(**generation_args)


def main():
    args = parse_args()
    torch, TextStreamer, FastLanguageModel = import_runtime_dependencies()
    model, tokenizer = load_model(args, FastLanguageModel)
    inputs = build_inputs(tokenizer, args.instruction, args.input, torch)
    generate(model, tokenizer, inputs, args, torch, TextStreamer)


if __name__ == "__main__":
    main()
