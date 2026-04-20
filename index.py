
from flask import Flask, request, jsonify
import g4f
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    return "G4F API Wrapper is running!"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    model = data.get('model', 'gpt-4.1')  # Default model
    messages = data.get('messages')
    images_data = data.get('images', [])

    if not messages:
        return jsonify({'error': 'Messages are required'}), 400

    g4f_messages = []
    for msg in messages:
        g4f_messages.append({'role': msg['role'], 'content': msg['content']})

    g4f_images = []
    for img_data in images_data:
        try:
            # img_data is expected to be a list: [base64_string, filename]
            base64_str = img_data[0].split(',')[1] # Remove data:image/jpeg;base64, prefix
            filename = img_data[1]
            image_bytes = base64.b64decode(base64_str)
            g4f_images.append([BytesIO(image_bytes), filename])
        except Exception as e:
            return jsonify({'error': f'Error processing image data: {e}'}), 400

    try:
        # Use specific providers that are more likely to work without auth
        # Or let G4F decide but handle the common 'RetryProvider' error more gracefully
        client = g4f.Client()
        
        # Determine provider based on model if needed, or use a reliable one
        provider = None
        
        # Mapping models to known providers if they are not in the 'auto' list or fail
        model_mapping = {
            'gpt-3.5-turbo': 'gpt-4o-mini', # Redirect to a more likely available model
            'gpt-4': 'gpt-4o',
            'llama-3.1-8b': 'llama-3.1-8b-instruct',
        }
        
        target_model = model_mapping.get(model, model)

        # Trying a few reliable providers if auto fails
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=g4f_messages,
                images=g4f_images if g4f_images else None
            )
        except:
            # Fallback to known working providers
            reliable_providers = [
                g4f.Provider.PollinationsAI,
                g4f.Provider.ApiAirforce
            ]
            success = False
            last_error = "Unknown error"
            for p in reliable_providers:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=g4f_messages,
                        provider=p,
                        images=g4f_images if g4f_images else None
                    )
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    continue
            if not success:
                raise Exception(f"All reliable providers failed. Last error: {last_error}")
        return jsonify({
            'model': model,
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': response.choices[0].message.content
                }
            }]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    # Return a list of commonly supported models in G4F
    models = [
        'gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-4.1',
        'llama-3.1-8b', 'llama-3.3-70b',
        'claude-3-opus', 'claude-3.5-sonnet',
        'gemini-pro', 'gemini-1.5-flash'
    ]
    return jsonify({'models': models})

@app.route('/v1/images/generations', methods=['POST'])
def image_generations():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    model = data.get('model', 'flux') # Default image generation model
    prompt = data.get('prompt')
    response_format = data.get('response_format', 'url')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        client = g4f.Client()
        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format=response_format
        )
        # G4F returns a list of data objects, each with a URL or b64_json
        results = []
        for item in response.data:
            if response_format == 'url':
                results.append({'url': item.url})
            else:
                results.append({'b64_json': item.b64_json})
        return jsonify({'data': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel specific entry point
if __name__ == '__main__':
    app.run(debug=True)
