from flask import Blueprint, request, jsonify
from app.model import get_model, generate_output

bp = Blueprint('routes', __name__)

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")
    lora_name = data.get("lora_name", "default")

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    model, tokenizer = get_model(lora_name)
    result = generate_output(model, tokenizer, query)
    return jsonify(result)