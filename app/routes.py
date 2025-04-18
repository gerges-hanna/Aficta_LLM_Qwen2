from flask import Blueprint, request, jsonify
from app.model import get_model, generate_output, convert_filters_to_api_format

bp = Blueprint('routes', __name__)

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")
    lora_name = data.get("lora_name", "default")

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    model, tokenizer = get_model(lora_name)
    if(lora_name=="airline_code"):
        result=predict_airline_code(model,tokenizer,user_input)
    else:
        result = generate_output(model, tokenizer, query)
    
    return jsonify(result)



@bp.route("/predict_filter", methods=["POST"])
def predict_filter():
    data = request.get_json()
    query = data.get("query", "")
    lora_name = data.get("lora_name", "default")

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400
    model, tokenizer = get_model(lora_name)
    result = generate_output(model, tokenizer, query)
    api_format=convert_filters_to_api_format(model, tokenizer,result)
    return jsonify(api_format)




@bp.route("/")
def hello():
    return "Hello, world from Flask on AWS EC2! Qwen Project"