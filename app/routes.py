from flask import Blueprint, request, jsonify, render_template
from .models import predict_sarcasm

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text")
    print('text: ', text)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = predict_sarcasm(text)

    print('prediction: ', prediction)
    response = {"text": text, "sarcasm": bool(prediction)}
    return jsonify(response)
