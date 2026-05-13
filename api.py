"""
api.py — Flask REST API for AgentLens
Run with: python api.py
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from agent_core import (
    get_recommendations,
    check_openai_client_status,
)
from ollama_utils import (
    list_local_models,
    get_ollama_status,
    test_model,
)

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.route('/')
def serve_frontend():
    """Serve the main frontend HTML file."""
    return send_from_directory('frontend', 'index.html')


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    POST /api/recommend
    Accepts: { "query": "..." }
    Returns: Structured recommendation JSON with workflow analysis and model suggestions
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data['query'].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    result = get_recommendations(query)
    return jsonify(result)


@app.route('/api/local-models', methods=['GET'])
def get_local_models():
    """
    GET /api/local-models
    Returns: List of locally installed Ollama models with metadata
    """
    models = list_local_models()
    return jsonify(models)


@app.route('/api/test-model', methods=['POST'])
def test_llm():
    """
    POST /api/test-model
    Accepts: { "model": "...", "prompt": "..." }
    Returns: { "response": "..." } with the model's output
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    model = data.get('model')
    prompt = data.get('prompt')

    if not model:
        return jsonify({"error": "Missing 'model' in request body"}), 400
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    response_text = test_model(model, prompt)
    return jsonify({"response": response_text})


@app.route('/api/status', methods=['GET'])
def status():
    """
    GET /api/status
    Returns: Combined status of OpenAI client (Ollama endpoint) and Ollama server
    """
    client_status = check_openai_client_status()
    ollama_status = get_ollama_status()

    return jsonify({
        "client": client_status,
        "ollama": ollama_status,
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Determine if we're running in a frozen environment or as a script
    dev_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    print("Starting AgentLens API server...")
    print("Frontend available at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/*")
    app.run(host='0.0.0.0', port=5000, debug=dev_mode)
