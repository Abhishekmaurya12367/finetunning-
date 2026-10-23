import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config

TEST_PROMPTS = [
    "Replace with your test prompt 1",
    "Replace with your test prompt 2",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, str(cfg.adapter_dir))
    model.eval()
    n = cfg.generation.max_new_tokens

    for prompt in TEST_PROMPTS:
        print("=" * 70)
        print(f"PROMPT: {prompt}\n")
        print("--- BASE MODEL ---")
        with model.disable_adapter():
            print(generate(model, tokenizer, prompt, n))
        print("\n--- TUNED MODEL ---")
        print(generate(model, tokenizer, prompt, n))


if __name__ == "__main__":
    main()  