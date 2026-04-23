from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
import base64
import time
import uuid
import concurrent.futures
from io import BytesIO

app = Flask(__name__)
CORS(app)

G4F_TIMEOUT = 20  # seconds max per provider attempt

AVAILABLE_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4-turbo", "gpt-3.5-turbo", "o1-mini", "o3-mini",
    "llama-3.3-70b", "llama-3.1-8b", "llama-3.1-70b", "mixtral-8x7b",
    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash",
    "claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku",
    "deepseek-v3", "deepseek-r1", "qwen-2.5-72b",
    "flux", "flux-pro", "dall-e-3",
]

MODEL_MAPPING = {
    "gpt-3.5-turbo": "gpt-4o-mini",
    "gpt-4":         "gpt-4o",
    "llama-3.1-8b":  "llama-3.1-8b-instruct",
    "llama-3.1-70b": "llama-3.1-70b-instruct",
    "llama-3.3-70b": "llama-3.3-70b-instruct",
}

# Fastest, most reliable providers — tried in order
# Safely load providers - skip any that don't exist in this g4f version
_PROVIDER_NAMES = [
    'PollinationsAI',
    'ApiAirforce',
    'Blackbox',
    'DDG',
    'DeepInfraChat',
    'Liaobots',
    'You',
]
PROVIDERS = []
for _name in _PROVIDER_NAMES:
    _p = getattr(g4f.Provider, _name, None)
    if _p is not None:
        PROVIDERS.append(_p)



def _call_g4f(model, messages, provider=None, images=None):
    """Run a single g4f call — designed to be run inside a thread with timeout."""
    client = g4f.Client()
    kwargs = {"model": model, "messages": messages}
    if provider:
        kwargs["provider"] = provider
    if images:
        kwargs["images"] = images
    return client.chat.completions.create(**kwargs)


def run_with_timeout(fn, timeout, *args, **kwargs):
    """Run fn(*args, **kwargs) and raise TimeoutError if it exceeds timeout seconds."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Provider timed out after {timeout}s")


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "G4F API Wrapper is running!",
        "endpoints": [
            "GET  /v1/models",
            "POST /v1/chat/completions",
            "POST /v1/images/generations",
            "GET  /health"
        ]
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": int(time.time())})


@app.route("/v1/models", methods=["GET"])
def list_models():
    data = [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "g4f"}
        for m in AVAILABLE_MODELS
    ]
    return jsonify({"object": "list", "data": data})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    if not data:
        return jsonify({"error": {"message": "Invalid JSON payload", "type": "invalid_request_error"}}), 400

    model = data.get("model", "gpt-4o")
    messages = data.get("messages")
    images_data = data.get("images", [])

    if not messages:
        return jsonify({"error": {"message": "Messages are required", "type": "invalid_request_error"}}), 400

    g4f_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    # Parse images safely
    g4f_images = []
    for img in images_data:
        try:
            raw = img[0]
            if "," in raw:
                raw = raw.split(",", 1)[1]
            g4f_images.append([BytesIO(base64.b64decode(raw)), img[1]])
        except Exception as e:
            return jsonify({"error": {"message": f"Image error: {e}", "type": "invalid_request_error"}}), 400

    target_model = MODEL_MAPPING.get(model, model)
    images_arg = g4f_images if g4f_images else None

    response = None
    last_error = "No providers succeeded"

    # 1. Try auto (let g4f pick) with timeout
    try:
        response = run_with_timeout(
            _call_g4f, G4F_TIMEOUT,
            target_model, g4f_messages, None, images_arg
        )
    except Exception as e:
        last_error = str(e)

    # 2. Try each fallback provider with individual timeouts
    if not response:
        for provider in PROVIDERS:
            try:
                response = run_with_timeout(
                    _call_g4f, G4F_TIMEOUT,
                    target_model, g4f_messages, provider, images_arg
                )
                if response:
                    break
            except Exception as e:
                last_error = f"{provider.__name__}: {e}"
                continue

    if not response:
        return jsonify({"error": {"message": last_error, "type": "server_error"}}), 500

    content = response.choices[0].message.content or ""
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
        return jsonify({"error": {"message": "Invalid JSON payload", "type": "invalid_request_error"}}), 400

    model = data.get("model", "flux")
    prompt = data.get("prompt")
    response_format = data.get("response_format", "url")

    if not prompt:
        return jsonify({"error": {"message": "Prompt is required", "type": "invalid_request_error"}}), 400

    def _gen():
        client = g4f.Client()
        return client.images.generate(model=model, prompt=prompt, response_format=response_format)

    try:
        response = run_with_timeout(_gen, G4F_TIMEOUT)
        results = []
        for item in response.data:
            if response_format == "url":
                results.append({"url": item.url})
            else:
                results.append({"b64_json": item.b64_json})
        return jsonify({"created": int(time.time()), "data": results})
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
