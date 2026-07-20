import argparse
import sys


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with an Unsloth/Transformers Alpaca-style model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face model repo or local model directory. For Unsloth notebooks "
            "this is often ./lora_model, ./outputs, or your HF repo name."
        ),
    )
    parser.add_argument(
        "--instruction",
        default="Continue the fibonacci sequence.",
        help="Instruction text to send to the model.",
    )
    parser.add_argument(
        "--input",
        default="1, 1, 2, 3, 5, 8",
        help="Optional Alpaca input/context text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Set to 0 for deterministic greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model in 4-bit mode when using Unsloth.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length used when loading through Unsloth.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on. auto uses CUDA when available.",
    )
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: torch. Install the requirements first, then rerun inference.py."
        ) from exc
    return torch


def selected_device(device_arg, torch):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch cannot see a CUDA GPU.")
    return device_arg


def load_with_unsloth(args):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_with_transformers(args, device, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {}

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def load_model(args, device, torch):
    try:
        return load_with_unsloth(args)
    except ImportError:
        print(
            "Unsloth is not installed; falling back to Transformers. "
            "Install unsloth for the same runtime path as the Colab notebook.",
            file=sys.stderr,
        )
        return load_with_transformers(args, device, torch)


def main():
    args = parse_args()
    torch = import_torch()
    device = selected_device(args.device, torch)
    model, tokenizer = load_model(args, device, torch)

    prompt = ALPACA_PROMPT.format(args.instruction, args.input, "")
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    from transformers import TextStreamer

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if args.temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        model.generate(**generation_kwargs)


if __name__ == "__main__":
    main()
