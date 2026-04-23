import os
import time
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

REQUEST_TIMEOUT = 25  # seconds

# ---- Provider endpoints ----
POLLINATIONS_URL = "https://text.pollinations.ai/openai"
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/"

# ---- Pollinations uses its own model name system ----
# https://text.pollinations.ai/models — real model IDs
POLLINATIONS_MODEL_MAP = {
    "gpt-4o":           "openai",
    "gpt-4o-mini":      "openai-mini",
    "gpt-4.1":          "openai",
    "gpt-4.1-mini":     "openai-mini",
    "gpt-4.1-nano":     "openai-mini",
    "gpt-4-turbo":      "openai-large",
    "gpt-3.5-turbo":    "openai-mini",
    "o1-mini":          "openai-reasoning",
    "o3-mini":          "openai-reasoning",
    "llama-3.3-70b":    "llama",
    "llama-3.1-70b":    "llama",
    "llama-3.1-8b":     "llamalight",
    "mixtral-8x7b":     "mistral",
    "gemini-1.5-pro":   "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-2.0-flash": "gemini",
    "claude-3.5-sonnet":"claude-hybridspace",
    "claude-3-opus":    "claude-hybridspace",
    "claude-3-haiku":   "claude-hybridspace",
    "deepseek-v3":      "deepseek",
    "deepseek-r1":      "deepseek-reasoner",
    "qwen-2.5-72b":     "qwen-coder",
}

# ---- ApiAirforce uses different model IDs ----
AIRFORCE_MODEL_MAP = {
    "gpt-4o":           "gpt-4o",
    "gpt-4o-mini":      "gpt-4o-mini",
    "gpt-4.1":          "gpt-4o",          # fallback
    "gpt-4.1-mini":     "gpt-4o-mini",
    "gpt-4.1-nano":     "gpt-4o-mini",
    "gpt-4-turbo":      "gpt-4-turbo",
    "gpt-3.5-turbo":    "gpt-3.5-turbo",
    "o1-mini":          "o1-mini",
    "o3-mini":          "o3-mini",
    "llama-3.3-70b":    "llama-3.3-70b",
    "llama-3.1-70b":    "llama-3.1-70b",
    "llama-3.1-8b":     "llama-3.1-8b",
    "mixtral-8x7b":     "mixtral-8x7b",
    "gemini-1.5-pro":   "google-gemini-pro",
    "gemini-1.5-flash": "google-gemini-flash",
    "gemini-2.0-flash": "google-gemini-flash-2.0",
    "claude-3.5-sonnet":"claude-3-5-sonnet-20241022",
    "claude-3-opus":    "claude-3-opus-20240229",
    "claude-3-haiku":   "claude-3-haiku-20240307",
    "deepseek-v3":      "deepseek-chat",
    "deepseek-r1":      "deepseek-reasoner",
    "qwen-2.5-72b":     "Qwen/Qwen2.5-72B-Instruct",
}

AVAILABLE_MODELS = list(POLLINATIONS_MODEL_MAP.keys())
IMAGE_MODELS = ["flux", "flux-pro", "dall-e-3"]

# Error phrases — if response contains these, treat as provider failure
ERROR_PHRASES = [
    "does not exist",
    "not supported",
    "no model",
    "invalid model",
    "unavailable",
    "discord.gg",
    "api.airforce",
    "error:",
    "<!doctype",
    "<html",
]


def is_valid_response(text: str) -> bool:
    """Return False if the response looks like a provider error message."""
    if not text or len(text.strip()) < 2:
        return False
    lower = text.lower()
    return not any(phrase in lower for phrase in ERROR_PHRASES)


def call_pollinations(model: str, messages: list) -> str:
    poll_model = POLLINATIONS_MODEL_MAP.get(model, "openai")
    resp = requests.post(
        POLLINATIONS_URL,
        json={"model": poll_model, "messages": messages, "private": True},
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not is_valid_response(content):
        raise ValueError(f"Invalid response from Pollinations: {content[:100]}")
    return content


def call_apiairforce(model: str, messages: list) -> str:
    af_model = AIRFORCE_MODEL_MAP.get(model, model)
    resp = requests.post(
        "https://api.airforce/v1/chat/completions",
        json={"model": af_model, "messages": messages},
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not is_valid_response(content):
        raise ValueError(f"Invalid response from ApiAirforce: {content[:100]}")
    return content


def get_completion(model: str, messages: list):
    """Try providers in order. Returns (content, None) or (None, error_str)."""
    errors = []

    # 1. PollinationsAI
    try:
        return call_pollinations(model, messages), None
    except Exception as e:
        errors.append(f"Pollinations: {e}")

    # 2. ApiAirforce
    try:
        return call_apiairforce(model, messages), None
    except Exception as e:
        errors.append(f"ApiAirforce: {e}")

    return None, " | ".join(str(e) for e in errors)


# ---- Routes ----

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "version": "2.1",
        "message": "G4F API Wrapper is running!",
        "endpoints": [
            "GET  /v1/models",
            "POST /v1/chat/completions",
            "POST /v1/images/generations",
            "GET  /health",
        ],
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

    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    content, error = get_completion(model, msgs)

    if not content:
        return jsonify({"error": {"message": error or "All providers failed", "type": "server_error"}}), 500

    pt = sum(len(m["content"].split()) for m in msgs)
    ct = len(content.split())

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
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
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
        encoded = requests.utils.quote(prompt)
        seed = int(time.time()) % 99999
        image_url = f"{POLLINATIONS_IMAGE_URL}{encoded}?model={model}&seed={seed}&nologo=true"
        check = requests.head(image_url, timeout=REQUEST_TIMEOUT)
        if check.status_code == 200:
            return jsonify({"created": int(time.time()), "data": [{"url": image_url}]})
        raise Exception(f"Image API returned {check.status_code}")
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
