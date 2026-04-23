import os
import time
import uuid
import base64
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

REQUEST_TIMEOUT = 25  # seconds

# PollinationsAI OpenAI-compatible endpoint (free, no auth needed)
POLLINATIONS_URL = "https://text.pollinations.ai/openai"
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/"

# ApiAirforce endpoint
APIAIRFORCE_URL = "https://api.airforce/v1/chat/completions"

AVAILABLE_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4-turbo", "gpt-3.5-turbo", "o1-mini", "o3-mini",
    "llama-3.3-70b", "llama-3.1-8b", "llama-3.1-70b", "mixtral-8x7b",
    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash",
    "claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku",
    "deepseek-v3", "deepseek-r1", "qwen-2.5-72b",
]

IMAGE_MODELS = ["flux", "flux-pro", "dall-e-3"]

# Map model names to what each provider supports
POLLINATIONS_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4-turbo", "gpt-3.5-turbo", "o1-mini", "o3-mini",
    "claude-3.5-sonnet", "claude-3-haiku",
    "llama-3.3-70b", "llama-3.1-8b",
    "gemini-1.5-flash", "gemini-2.0-flash",
    "deepseek-v3", "deepseek-r1",
    "mistral", "qwen-2.5-72b",
}

MODEL_ALIASES = {
    "gpt-3.5-turbo":  "gpt-4o-mini",
    "gpt-4":          "gpt-4o",
    "llama-3.1-8b":   "llama-3.1-8b-instruct",
    "llama-3.1-70b":  "llama-3.1-70b-instruct",
    "llama-3.3-70b":  "llama-3.3-70b-instruct",
    "mixtral-8x7b":   "mistral",
    "gemini-1.5-pro": "gemini-1.5-flash",
    "claude-3-opus":  "claude-3.5-sonnet",
    "claude-3-haiku": "claude-3-haiku",
}


def call_pollinations(model, messages, timeout=REQUEST_TIMEOUT):
    payload = {
        "model": model,
        "messages": messages,
        "private": True,
    }
    resp = requests.post(
        POLLINATIONS_URL,
        json=payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def call_apiairforce(model, messages, timeout=REQUEST_TIMEOUT):
    payload = {"model": model, "messages": messages}
    resp = requests.post(
        APIAIRFORCE_URL,
        json=payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def get_completion(model, messages):
    """Try providers in order, return (content, error)."""
    target = MODEL_ALIASES.get(model, model)
    errors = []

    # Provider 1: PollinationsAI
    try:
        poll_model = target if target in POLLINATIONS_MODELS else "gpt-4o"
        content = call_pollinations(poll_model, messages)
        if content:
            return content, None
    except Exception as e:
        errors.append(f"Pollinations: {e}")

    # Provider 2: ApiAirforce
    try:
        content = call_apiairforce(target, messages)
        if content:
            return content, None
    except Exception as e:
        errors.append(f"ApiAirforce: {e}")

    return None, " | ".join(errors)


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "G4F API Wrapper is running!",
        "version": "2.0",
        "endpoints": [
            "GET  /v1/models",
            "POST /v1/chat/completions",
            "POST /v1/images/generations",
            "GET  /health",
        ]
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": int(time.time())})


@app.route("/v1/models", methods=["GET"])
def list_models():
    data = [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "pollinations"}
        for m in AVAILABLE_MODELS + IMAGE_MODELS
    ]
    return jsonify({"object": "list", "data": data})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    if not data:
        return jsonify({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}), 400

    model = data.get("model", "gpt-4o")
    messages = data.get("messages")
    if not messages:
        return jsonify({"error": {"message": "messages required", "type": "invalid_request_error"}}), 400

    # Normalise messages
    g4f_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    content, error = get_completion(model, g4f_messages)

    if not content:
        return jsonify({"error": {"message": error or "All providers failed", "type": "server_error"}}), 500

    prompt_tokens = sum(len(m["content"].split()) for m in g4f_messages)
    completion_tokens = len(content.split())

    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.route("/v1/images/generations", methods=["POST"])
def image_generations():
    data = request.json
    if not data:
        return jsonify({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}), 400

    prompt = data.get("prompt")
    model = data.get("model", "flux")
    if not prompt:
        return jsonify({"error": {"message": "prompt required", "type": "invalid_request_error"}}), 400

    try:
        # PollinationsAI image API - just needs prompt in URL
        encoded = requests.utils.quote(prompt)
        seed = int(time.time()) % 99999
        image_url = f"{POLLINATIONS_IMAGE_URL}{encoded}?model={model}&seed={seed}&nologo=true"

        # Verify URL is reachable
        check = requests.head(image_url, timeout=REQUEST_TIMEOUT)
        if check.status_code == 200:
            return jsonify({
                "created": int(time.time()),
                "data": [{"url": image_url}]
            })
        else:
            raise Exception(f"Image API returned {check.status_code}")
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
