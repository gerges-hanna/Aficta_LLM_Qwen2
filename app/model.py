import torch
import json
from json_repair import repair_json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from functools import lru_cache
import os

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# Get the directory of the current file (e.g., app/model.py)
project_root = os.path.dirname(os.path.abspath(__file__))

# Build absolute path to the lora_model folder in your project
LORA_MODELS = {
    "default": os.path.join(project_root, "..", "lora_model"),
    # In the future: "hotel_filter": os.path.join(project_root, "..", "lora_model", "hotel_filter"),
}


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

@lru_cache()
def get_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    return model, tokenizer

@lru_cache()
def get_model(lora_name="default"):
    base_model, tokenizer = get_base_model()
    lora_path = LORA_MODELS.get(lora_name)
    if lora_path is None:
        raise ValueError(f"LoRA adapter '{lora_name}' not found.")
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer

def generate_output(model, tokenizer, input_text: str, max_new_tokens=256):
    prompt = f"User: {input_text}\nAssistant:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    raw_json_text = full_output.split("Assistant:")[-1].strip()

    try:
        fixed_json_text = repair_json(raw_json_text)
        data = json.loads(fixed_json_text)

        if isinstance(data, dict) and "filters" in data and "sort_by" in data:
            return data

        if isinstance(data, list):
            filters, sort_by = [], []
            for item in data:
                if isinstance(item, dict) and "field" in item:
                    sort_by.append(item)
                elif isinstance(item, dict) and "filters" in item:
                    filters.extend(item["filters"])
                elif isinstance(item, list):
                    for sub in item:
                        if "field" in sub and "order" in sub:
                            sort_by.append(sub)
            return {"filters": filters, "sort_by": sort_by}

    except Exception as e:
        print("❌ Error repairing/parsing JSON:", e)

    return {"filters": [], "sort_by": []}