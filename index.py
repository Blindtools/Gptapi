from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
import base64
import time
import uuid
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Bug fix #6: Create client once at module level, not per request
client = g4f.Client()

# Full model list with correct g4f model names
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
    "llama-3.3-70b",
    "llama-3.1-8b",
    "llama-3.1-70b",
    "mixtral-8x7b",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "claude-3.5-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
    "deepseek-v3",
    "deepseek-r1",
    "qwen-2.5-72b",
    "flux",
    "flux-pro",
    "dall-e-3",
]

MODEL_MAPPING = {
    "gpt-3.5-turbo": "gpt-4o-mini",
    "gpt-4":         "gpt-4o",
    "llama-3.1-8b":  "llama-3.1-8b-instruct",
    "llama-3.1-70b": "llama-3.1-70b-instruct",
    "llama-3.3-70b": "llama-3.3-70b-instruct",
}

FALLBACK_PROVIDERS = [
    g4f.Provider.PollinationsAI,
    g4f.Provider.ApiAirforce,
]


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
    # Bug fix #4: Return proper OpenAI-compatible format
    data = [
        {
            "id": model,
            "object": "model",
            "created": 1700000000,
            "owned_by": "g4f",
        }
        for model in AVAILABLE_MODELS
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

    g4f_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]

    # Bug fix #7: Safe base64 image parsing
    g4f_images = []
    for img_data in images_data:
        try:
            raw = img_data[0]
            if "," in raw:
                raw = raw.split(",", 1)[1]
            filename = img_data[1]
            image_bytes = base64.b64decode(raw)
            g4f_images.append([BytesIO(image_bytes), filename])
        except Exception as e:
            return jsonify({"error": {"message": f"Error processing image: {e}", "type": "invalid_request_error"}}), 400

    target_model = MODEL_MAPPING.get(model, model)

    # Bug fix #3: Only pass images kwarg when images actually exist
    kwargs = {"model": target_model, "messages": g4f_messages}
    if g4f_images:
        kwargs["images"] = g4f_images

    response = None
    last_error = "Unknown error"

    # Bug fix #1: Use except Exception, not bare except
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        last_error = str(e)

    # Bug fix #2: Fallback also uses target_model, not raw model string
    if response is None:
        for provider in FALLBACK_PROVIDERS:
            try:
                fallback_kwargs = {
                    "model": target_model,
                    "messages": g4f_messages,
                    "provider": provider,
                }
                if g4f_images:
                    fallback_kwargs["images"] = g4f_images
                response = client.chat.completions.create(**fallback_kwargs)
                if response:
                    break
            except Exception as e:
                last_error = str(e)
                continue

    if response is None:
        return jsonify({"error": {"message": f"All providers failed. Last error: {last_error}", "type": "server_error"}}), 500

    content = response.choices[0].message.content or ""
    prompt_tokens = sum(len(m["content"].split()) for m in g4f_messages)
    completion_tokens = len(content.split())

    # Bug fix #5: Full OpenAI-compatible response with id, object, created, usage
    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
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

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format=response_format,
        )
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
