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
import requests

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

# === BLIP via Replicate API ===
import replicate

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def analyze_image_with_blip(image_data, custom_prompt=""):
    """BLIP image captioning using Replicate API"""
    try:
        if not REPLICATE_API_TOKEN:
            return {"error": "Replicate API token not configured. Please set REPLICATE_API_TOKEN environment variable."}

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

        print(f"📸 Processing image with BLIP via Replicate...")
        print(f"📏 Image: {dimensions}, {file_size_kb:.1f}KB")
        
        # Convert to data URL for Replicate
        if not image_data.startswith("data:image"):
            # Convert bytes back to data URL if needed
            import base64
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_data = f"data:image/{image_format};base64,{encoded_image}"

        # Configure Replicate client
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        
        # Run BLIP model on Replicate
        output = replicate.run(
            "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            input={
                "image": image_data,
                "task": "image_captioning"
            }
        )
        
        print(f"🔄 Replicate Response: {output}")
        
        # Handle Replicate response
        if isinstance(output, list) and len(output) > 0:
            caption = output[0]
        elif isinstance(output, str):
            caption = output
        else:
            caption = str(output)

        print(f"🎯 Generated caption: {caption}")

        return {
            "alt_text": caption,
            "explanation": "Generated using BLIP via Replicate API",
            "success": True,
            "dimensions": dimensions,
            "size_kb": round(file_size_kb, 1),
            "format": image_format,
            "provider": "Replicate BLIP",
            "cost": "$0.00031 per run"
        }

    except Exception as e:
        print(f"❌ Error in BLIP via Replicate: {e}")
        traceback.print_exc()
        return {"error": f"BLIP Replicate API error: {str(e)}"}


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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "browser_use": browser_use_available,
        "llm_initialized": bool(llm),
        "model": QWEN_MODEL,
        "blip_available": bool(REPLICATE_API_TOKEN),
        "blip_provider": "Replicate API",
        "blip_model": "Salesforce BLIP",
        "deployment": "Railway"
    })


@app.route("/test-blip", methods=["GET"])
def test_blip():
    """Test BLIP model via Replicate with a simple image"""
    try:
        if not REPLICATE_API_TOKEN:
            return jsonify({"error": "REPLICATE_API_TOKEN not configured"})
        
        # Simple test image (small white square)
        test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        print("🧪 Testing BLIP via Replicate...")
        result = analyze_image_with_blip(test_image_base64)
        
        return jsonify({
            "test_result": result,
            "replicate_token_configured": True,
            "model": "Salesforce BLIP",
            "provider": "Replicate API",
            "cost_per_run": "$0.00031"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})


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
    llm_choice = data.get("llm", "blip")  # Always use BLIP by default
    
    if not image_data:
        return jsonify({"error": "Image data is required"}), 400

    # Always use BLIP for image analysis via Replicate
    print(f"🔄 Processing image with BLIP via Replicate...")
    result = analyze_image_with_blip(image_data, custom_prompt)

    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


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
    print("🚀 Starting server on Railway...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


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
    """BLIP image captioning using local model"""
    global blip_processor, blip_model
    
    try:
        if not blip_loaded or not blip_processor or not blip_model:
            return {"error": "BLIP model not loaded. Please check server logs."}

        if image_data.startswith("data:image"):
            image_data_clean = image_data.split(",")[1]
            image_format = image_data.split(",")[0].split("/")[1].split(";")[0]
        else:
            image_data_clean = image_data
            image_format = "unknown"

        image_bytes = base64.b64decode(image_data_clean)
        file_size_kb = len(image_bytes) / 1024
        
        # Get image dimensions
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        dimensions = f"{width}x{height}"

        print(f"📸 Processing image with BLIP: {dimensions}, {file_size_kb:.1f}KB")

        # Process with BLIP
        device = "cpu"
        inputs = blip_processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=50, do_sample=False)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)

        print(f"🎯 Generated caption: {caption}")

        return {
            "alt_text": caption,
            "explanation": "Generated using BLIP image captioning model",
            "success": True,
            "dimensions": dimensions,
            "size_kb": round(file_size_kb, 1),
            "format": image_format,
            "provider": "Local BLIP Model"
        }

    except Exception as e:
        print(f"❌ Error in BLIP image analysis: {e}")
        traceback.print_exc()
        return {"error": f"BLIP processing error: {str(e)}"}


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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "browser_use": browser_use_available,
        "llm_initialized": bool(llm),
        "model": QWEN_MODEL,
        "blip_available": blip_loaded,
        "blip_model": "Salesforce/blip-image-captioning-base",
        "deployment": "Railway"
    })


@app.route("/test-blip", methods=["GET"])
def test_blip():
    """Test BLIP model with a simple image"""
    try:
        if not blip_loaded:
            return jsonify({"error": "BLIP model not loaded"})
        
        # Simple test image (small white square)
        test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        print("🧪 Testing BLIP with test image...")
        result = analyze_image_with_blip(test_image_base64)
        
        return jsonify({
            "test_result": result,
            "blip_loaded": blip_loaded,
            "model": "Salesforce/blip-image-captioning-base",
            "provider": "Local BLIP on Railway"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})


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
    llm_choice = data.get("llm", "blip")  # Always use BLIP by default
    
    if not image_data:
        return jsonify({"error": "Image data is required"}), 400

    # Always use BLIP for image analysis
    print(f"🔄 Processing image with BLIP...")
    result = analyze_image_with_blip(image_data, custom_prompt)

    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


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
    print("🚀 Starting server on Railway...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)