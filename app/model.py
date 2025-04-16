import torch
import json
import os
from json_repair import repair_json
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Absolute paths for LoRA adapters
project_root = os.path.dirname(os.path.abspath(__file__))
LORA_MODELS = {
    "default": os.path.join(project_root, "..", "lora_model"),
    # Add more like: "hotel": os.path.join(..., "hotel_filter")
}

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model + tokenizer once
print("[INIT] Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

# In-memory cache for LoRA adapters
_loaded_loras = {}

def get_model(lora_name="default"):
    if lora_name in _loaded_loras:
        return _loaded_loras[lora_name]

    lora_path = LORA_MODELS.get(lora_name)
    if not lora_path:
        raise ValueError(f"LoRA adapter '{lora_name}' not found.")

    print(f"[INFO] Loading LoRA adapter: {lora_name} from {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    _loaded_loras[lora_name] = (model, tokenizer)
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
