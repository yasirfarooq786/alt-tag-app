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

app = Flask(__name__)
CORS(app)

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", 'https://router.requesty.ai/v1')
QWEN_MODEL = os.getenv("QWEN_MODEL", "alibaba/qwen-turbo")

print(f"✅ Using Alibaba Qwen model: {QWEN_MODEL}")
print(f"✅ Base URL: {QWEN_BASE_URL}")

llm = None
browser_use_available = False

# === BLIP imports ===
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

blip_processor, blip_model = None, None

def load_blip():
    global blip_processor, blip_model
    try:
        # Load model on-demand with caching
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/tmp/transformers_cache')
        blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir
        )
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        print("✅ BLIP model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load BLIP model: {e}")
        traceback.print_exc()

# Load model lazily on first request instead of at startup
# load_blip()


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
    global blip_processor, blip_model
    try:
        # Load model on first use if not already loaded
        if not blip_processor or not blip_model:
            print("Loading BLIP model on demand...")
            load_blip()
            
        if not blip_processor or not blip_model:
            return {"error": "BLIP model not loaded"}

        if image_data.startswith("data:image"):
            image_data_clean = image_data.split(",")[1]
            image_format = image_data.split(",")[0].split("/")[1].split(";")[0]
        else:
            image_data_clean = image_data
            image_format = "unknown"

        image_bytes = base64.b64decode(image_data_clean)
        file_size_kb = len(image_bytes) / 1024
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        dimensions = f"{width}x{height}"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(device)

        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        result = {
            "alt_text": caption,
            "explanation": "Generated using BLIP image captioning",
            "success": True,
            "dimensions": dimensions,
            "size_kb": round(file_size_kb, 1),
            "format": image_format
        }
        return result

    except Exception as e:
        print(f"❌ Error in BLIP image analysis: {e}")
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
        "blip_loaded": blip_model is not None
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
    
    # Serve specific files if they exist
    if path != "" and os.path.exists(os.path.join(build_folder, path)):
        return send_from_directory(build_folder, path)
    else:
        # Serve index.html for all other routes (React Router)
        return send_from_directory(build_folder, "index.html")


if __name__ == "__main__":
    print("🚀 Starting server...")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False for production