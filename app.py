import asyncio
import os
import base64
import json
import traceback
import io

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

# Load env
load_dotenv()

app = Flask(__name__, static_folder='alt-tag-generator/build', static_url_path='')
CORS(app)

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", 'https://router.requesty.ai/v1')
QWEN_MODEL = os.getenv("QWEN_MODEL", "alibaba/qwen-turbo")

print(f"✅ Using Alibaba Qwen model: {QWEN_MODEL}")
print(f"✅ Base URL: {QWEN_BASE_URL}")

llm = None
browser_use_available = False

# === BLIP via Hugging Face API ===
import requests

def analyze_image_with_blip_api(image_data, custom_prompt=""):
    try:
        HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        if not HF_API_TOKEN:
            return {"error": "Hugging Face API token not configured"}

        # Try multiple working models in order of preference
        models_to_try = [
            "nlpconnect/vit-gpt2-image-captioning",  # Usually more reliable
            "Salesforce/blip-image-captioning-base",
            "microsoft/DialoGPT-medium"
        ]

        if image_data.startswith("data:image"):
            image_data_clean = image_data.split(",")[1]
            image_format = image_data.split(",")[0].split("/")[1].split(";")[0]
        else:
            image_data_clean = image_data
            image_format = "unknown"

        image_bytes = base64.b64decode(image_data_clean)
        file_size_kb = len(image_bytes) / 1024
        
        # Get image dimensions
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        dimensions = f"{width}x{height}"

        # Try each model until one works
        last_error = None
        for model_name in models_to_try:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                headers = {
                    "Authorization": f"Bearer {HF_API_TOKEN}",
                    "Content-Type": "application/octet-stream"
                }

                print(f"Trying model: {model_name}")
                response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
                
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text[:200]}...")
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Handle different response formats
                    if isinstance(result_data, list) and len(result_data) > 0:
                        caption = result_data[0].get('generated_text', 'No caption generated')
                    elif isinstance(result_data, dict):
                        caption = result_data.get('generated_text', 'No caption generated')
                    else:
                        caption = str(result_data)
                        
                    # Success! Return the result
                    return {
                        "alt_text": caption,
                        "explanation": f"Generated using {model_name} via Hugging Face API",
                        "success": True,
                        "dimensions": dimensions,
                        "size_kb": round(file_size_kb, 1),
                        "format": image_format,
                        "model_used": model_name
                    }
                    
                elif response.status_code == 503:
                    last_error = f"Model {model_name} is loading, trying next model..."
                    print(last_error)
                    continue
                    
                elif response.status_code == 401:
                    return {"error": "Invalid Hugging Face API token"}
                    
                else:
                    last_error = f"Model {model_name}: {response.status_code} - {response.text}"
                    print(last_error)
                    continue
                    
            except Exception as e:
                last_error = f"Error with model {model_name}: {str(e)}"
                print(last_error)
                continue

        # If we get here, all models failed
        return {"error": f"All models failed. Last error: {last_error}"}

    except Exception as e:
        print(f"❌ Error in BLIP API image analysis: {e}")
        traceback.print_exc()
        return {"error": str(e)}


class BrowserUseCompatibleLLM:
    def __init__(self, chat):
        self._chat = chat
        self.provider = "qwen"
        self.model = chat.model_name if hasattr(chat, "model_name") else QWEN_MODEL
        self.temperature = getattr(chat, "temperature", 0.0)

    async def ainvoke(self, prompt):
        return await self._chat.acall(prompt)

    def invoke(self, prompt):
        return self._chat.invoke(prompt)

    def __getattr__(self, name):
        return getattr(self._chat, name)


def initialize_browser_use():
    global llm, browser_use_available
    try:
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(
            base_url=QWEN_BASE_URL,
            model=QWEN_MODEL,
            openai_api_key=QWEN_API_KEY,
            temperature=0.0
        )
        llm_proxy = BrowserUseCompatibleLLM(chat)
        print("✅ LLM initialized with browser-use compatible wrapper")

        try:
            from browser_use import Agent
            test_agent = Agent(
                task="Test browser-use initialization",
                llm=llm_proxy,
                headless=True   # 👈 prevent opening new browser window
            )
            browser_use_available = True
            print("✅ browser-use Agent initialized")
        except ImportError:
            print("⚠️ browser-use not available, URL analysis disabled")
            browser_use_available = False
        
        return llm_proxy

    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        traceback.print_exc()
        return None


llm = initialize_browser_use()


def run_async(coro):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        print(f"Error running async: {e}")
        traceback.print_exc()
        return {"error": str(e)}


async def analyze_url(url, custom_prompt=""):
    try:
        if not llm or not browser_use_available:
            return {"error": "Browser automation not available in this deployment. Please use image upload instead."}

        try:
            from browser_use import Agent
        except ImportError:
            return {"error": "Browser automation not available in this deployment. Please use image upload instead."}

        task = f"""
        Visit {url} and check all images for accessibility.

        Instructions:
        - Find all images.
        - Check ALT text.
        - Suggest improvements where needed.
        {f"Additional: {custom_prompt}" if custom_prompt else ""}
        """

        agent = Agent(task=task, llm=llm, headless=True)  # 👈 keep headless
        result = await agent.run()
        return {"success": True, "result": str(result)}

    except Exception as e:
        print(f"❌ Error in analyze_url: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def analyze_image_with_blip(image_data, custom_prompt=""):
    # Try Hugging Face first, fallback to OpenAI if available
    result = analyze_image_with_blip_api(image_data, custom_prompt)
    
    # If HF fails and OpenAI key is available, try OpenAI
    if "error" in result and os.getenv("OPENAI_API_KEY"):
        print("Hugging Face failed, trying OpenAI Vision...")
        return analyze_image_with_openai_vision(image_data, custom_prompt)
    
    return result


def analyze_image_with_qwen(image_data, custom_prompt=""):
    try:
        if not llm:
            return {"error": "LLM not initialized"}

        if image_data.startswith("data:image"):
            image_data_clean = image_data.split(",")[1]
            image_format = image_data.split(",")[0].split("/")[1].split(";")[0]
        else:
            image_data_clean = image_data
            image_format = "unknown"

        image_bytes = base64.b64decode(image_data_clean)
        file_size_kb = len(image_bytes) / 1024
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        aspect_ratio = width / height
        dimensions = f"{width}x{height}"

        prompt = f"""
You are an accessibility expert. Write ALT text under 125 characters.
Image:
- Format: {image_format.upper()}
- Dimensions: {dimensions}
- File size: {file_size_kb:.1f} KB
- User prompt: {custom_prompt or "None"}

Follow WCAG guidelines. Output JSON:
{{
  "alt_text": "...",
  "explanation": "..."
}}
"""
        response = llm.invoke(prompt)

        content = getattr(response, 'content', str(response))
        content = content.strip("`").strip()
        try:
            result = json.loads(content)
        except Exception:
            result = {
                "alt_text": "Description unavailable",
                "explanation": "Failed to parse response"
            }
        result.update({
            "success": True,
            "dimensions": dimensions,
            "size_kb": round(file_size_kb, 1),
            "format": image_format
        })
        return result

    except Exception as e:
        print(f"❌ Error in Qwen image analysis: {e}")
        traceback.print_exc()
        return {"error": str(e)}


@app.route("/test-blip", methods=["GET"])
def test_blip():
    """Test BLIP API with a simple image"""
    try:
        HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        if not HF_API_TOKEN:
            return jsonify({"error": "HF_API_TOKEN not set"})
        
        # Create a simple test image (small white square)
        test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        result = analyze_image_with_blip_api(test_image_base64)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/test-hf-api", methods=["GET"])
def test_hf_api():
    """Test Hugging Face API token validity"""
    try:
        HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        if not HF_API_TOKEN:
            return jsonify({"error": "HF_API_TOKEN not set"})
        
        # Test the token with a simple API call
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        
        if response.status_code == 200:
            user_info = response.json()
            return jsonify({
                "token_valid": True,
                "user": user_info.get("name", "Unknown"),
                "message": "Token is valid, BLIP API should work"
            })
        else:
            return jsonify({
                "token_valid": False,
                "error": f"Token validation failed: {response.status_code}"
            })
        
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/debug/build", methods=["GET"])
def debug_build():
    """Debug route to check build directory contents"""
    build_folder = "alt-tag-generator/build"
    try:
        if os.path.exists(build_folder):
            files = []
            for root, dirs, filenames in os.walk(build_folder):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), build_folder)
                    files.append(rel_path)
            return jsonify({
                "build_exists": True,
                "files": files[:20],  # First 20 files
                "total_files": len(files)
            })
        else:
            return jsonify({"build_exists": False, "error": "Build folder not found"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "browser_use": browser_use_available,
        "llm_initialized": bool(llm),
        "model": QWEN_MODEL,
        "hf_api_available": bool(os.getenv("HF_API_TOKEN"))
    })


@app.route("/analyze-url", methods=["POST"])
def api_analyze_url():
    data = request.get_json()
    url = data.get("url")
    custom_prompt = data.get("prompt", "")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    result = run_async(analyze_url(url, custom_prompt))
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route("/analyze-image", methods=["POST"])
def api_analyze_image():
    data = request.get_json()
    image_data = data.get("image")
    custom_prompt = data.get("prompt", "")
    llm_choice = data.get("llm", "blip")  # default = blip
    if not image_data:
        return jsonify({"error": "Image data is required"}), 400

    if llm_choice == "qwen":
        result = analyze_image_with_qwen(image_data, custom_prompt)
    else:
        result = analyze_image_with_blip(image_data, custom_prompt)

    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


# Updated static file serving for React build
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    # Serve React build files from alt-tag-generator/build
    build_folder = "alt-tag-generator/build"
    
    # Check if the build folder exists
    if not os.path.exists(build_folder):
        return jsonify({"error": "React build not found. Run npm run build first."}), 404
    
    # Handle static files specifically
    if path.startswith('static/'):
        static_file_path = os.path.join(build_folder, path)
        if os.path.exists(static_file_path):
            return send_from_directory(build_folder, path)
        else:
            return jsonify({"error": f"Static file not found: {path}"}), 404
    
    # Handle other specific files (manifest.json, favicon.ico, etc.)
    elif path and os.path.exists(os.path.join(build_folder, path)):
        return send_from_directory(build_folder, path)
    
    # For all other routes, serve index.html (React Router)
    else:
        index_path = os.path.join(build_folder, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(build_folder, "index.html")
        else:
            return jsonify({"error": "React app index.html not found"}), 404


if __name__ == "__main__":
    print("🚀 Starting server...")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False for production