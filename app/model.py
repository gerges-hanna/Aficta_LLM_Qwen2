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
    "default": os.path.join(project_root, "..", "lora_model","flight_filter"),
    "flight_filter": os.path.join(project_root, "..", "lora_model","flight_filter"),
    "airline_code": os.path.join(project_root, "..", "lora_model","airline_code"),
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



def predict_airline_code(model, tokenizer, user_input):
    prompt = f"""
أنت مساعد ذكي متخصص في معرفة كود IATA الخاص بشركات الطيران المختلفة.
يعني لو المستخدم كتب اسم شركة طيران، حتى لو فيه خطأ إملائي أو مكتوب بطريقة مختلفة، حاول تفهم الشركة المقصودة وارجع بكود الشركة المكون من حرفين.

الاسم: {user_input}
الكود:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the IATA code after "الكود:"
    if "الكود:" in result:
        result = result.split("الكود:")[-1].strip().split("\n")[0]

    return { "predicted_code": result}



def convert_filters_to_api_format(filters_json):
    stops = []   # default all included
    airlines = []

    for f in filters_json.get("filters", []):
        field = f["field"]
        value = f["value"]

        if field == "نوع_الرحلة" and (value.strip() == "مباشر" or value.strip() == "مباشرة"):
            stops = [0]  # only direct flights
        elif field == "نوع_الرحلة" and (value.strip() == "غير مباشر" or value.strip() == "غير مباشرة"):
            stops = [1]

        elif field == "شركة_الطيران":
            code = predict_iata_code(value.strip())
            if code:
                airlines.append({
            "code": code,
            "name": value.strip(),
            })

    return {
        "stops": stops,
        "airlines": airlines
    }