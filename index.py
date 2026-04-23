import os
import time
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

REQUEST_TIMEOUT = 25

# text.pollinations.ai — free, no auth needed
# Only 'openai' and 'openai-large' work without an API key
POLLINATIONS_URL = "https://text.pollinations.ai/openai"

# Map requested model → best free Pollinations model
# GPT-4 class → openai-large, everything else → openai
def get_pollinations_model(model: str) -> str:
    gpt4_class = {
        "gpt-4o", "gpt-4.1", "gpt-4-turbo",
        "claude-3-opus", "claude-3.5-sonnet",
        "gemini-1.5-pro", "deepseek-r1",
    }
    return "openai-large" if model in gpt4_class else "openai"

# ApiAirforce — free but rate limited; use correct model IDs
AIRFORCE_URL = "https://api.airforce/v1/chat/completions"
AIRFORCE_MODELS = {
    "gpt-4o":           "gpt-4o",
    "gpt-4o-mini":      "gpt-4o-mini",
    "gpt-4.1":          "gpt-4o",
    "gpt-4.1-mini":     "gpt-4o-mini",
    "gpt-4.1-nano":     "gpt-4o-mini",
    "gpt-4-turbo":      "gpt-4-turbo",
    "gpt-3.5-turbo":    "gpt-3.5-turbo",
    "o1-mini":          "o1-mini",
    "o3-mini":          "o3-mini",
    "llama-3.3-70b":    "llama-3.3-70b",
    "llama-3.1-70b":    "llama-3.1-70b-instruct",
    "llama-3.1-8b":     "llama-3.1-8b-instruct",
    "mixtral-8x7b":     "mixtral-8x7b-instruct-v0.1",
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

AVAILABLE_MODELS = list(AIRFORCE_MODELS.keys())
IMAGE_MODELS = ["flux", "flux-pro", "dall-e-3"]

ERROR_PHRASES = [
    "does not exist", "not supported", "invalid model",
    "discord.gg", "api.airforce", "<!doctype", "<html",
    "too many requests",
]

def is_valid(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    return not any(p in text.lower() for p in ERROR_PHRASES)


def call_pollinations(model: str, messages: list) -> str:
    poll_model = get_pollinations_model(model)
    resp = requests.post(
        POLLINATIONS_URL,
        json={"model": poll_model, "messages": messages, "private": True},
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not is_valid(content):
        raise ValueError(f"Bad response: {content[:100]}")
    return content


def call_apiairforce(model: str, messages: list) -> str:
    af_model = AIRFORCE_MODELS.get(model, "gpt-4o-mini")
    resp = requests.post(
        AIRFORCE_URL,
        json={"model": af_model, "messages": messages},
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )
    if resp.status_code == 429:
        # Wait and retry once
        time.sleep(3)
        resp = requests.post(
            AIRFORCE_URL,
            json={"model": af_model, "messages": messages},
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not is_valid(content):
        raise ValueError(f"Bad response: {content[:100]}")
    return content


def get_completion(model: str, messages: list):
    errors = []

    # 1. Try Pollinations (free, no auth, always available)
    try:
        return call_pollinations(model, messages), None
    except Exception as e:
        errors.append(f"Pollinations: {e}")

    # 2. Try ApiAirforce with retry
    try:
        return call_apiairforce(model, messages), None
    except Exception as e:
        errors.append(f"ApiAirforce: {e}")

    return None, " | ".join(errors)


# ─── Routes ───────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "version": "2.3",
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
    body = request.json
    if not body:
        return jsonify({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}), 400

    model = body.get("model", "gpt-4o")
    messages = body.get("messages")
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
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    })


@app.route("/v1/images/generations", methods=["POST"])
def image_generations():
    body = request.json
    if not body:
        return jsonify({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}), 400
    prompt = body.get("prompt")
    model = body.get("model", "flux")
    if not prompt:
        return jsonify({"error": {"message": "prompt required", "type": "invalid_request_error"}}), 400
    try:
        encoded = requests.utils.quote(prompt)
        seed = int(time.time()) % 99999
        url = f"https://image.pollinations.ai/prompt/{encoded}?model={model}&seed={seed}&nologo=true"
        check = requests.head(url, timeout=REQUEST_TIMEOUT)
        if check.status_code == 200:
            return jsonify({"created": int(time.time()), "data": [{"url": url}]})
        raise Exception(f"Status {check.status_code}")
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
