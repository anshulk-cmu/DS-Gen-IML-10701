"""Response generation using LLMs."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_generator(model_name: str, cache_dir: str = None, device_map: str = "auto"):
    """Load a causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompt: str,
    num_responses: int = 20,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate multiple responses for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    responses = []

    with torch.no_grad():
        for _ in range(num_responses):
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            responses.append(text.strip())

    return responses
