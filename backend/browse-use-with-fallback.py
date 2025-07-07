import asyncio
import os
import base64
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import io
from browser_use import Agent
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Alibaba Qwen model
QWEN_API_KEY = 'sk-AAndRNC8SMaWPqBhCrAALINb8JZKANyxIkNf1OdCRYbM4M+VaZVrROQ/uM/Gv16rMjOxggPst9gGTLtXQaBdeElVyLIRYE1mM7fMFFIcZQQ='
QWEN_BASE_URL = 'https://router.requesty.ai/v1'
QWEN_MODEL = "alibaba/qwen-turbo"

print(f"✅ Using Alibaba Qwen model: {QWEN_MODEL}")
print(f"✅ Base URL: {QWEN_BASE_URL}")

# 👇 BrowserUse-compatible LLM wrapper
class BrowserUseCompatibleLLM:
    def __init__(self, llm, model, provider="openai"):
        self._llm = llm
        self.model = model
        self.provider = provider

    def invoke(self, *args, **kwargs):
        return self._llm.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        import asyncio
        return await asyncio.to_thread(self._llm.invoke, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._llm, item)

# Initialize LLM for browser-use
try:
    llm_raw = ChatOpenAI(
        base_url=QWEN_BASE_URL,
        model=QWEN_MODEL,
        openai_api_key=QWEN_API_KEY,
        temperature=0.0
    )
    llm = BrowserUseCompatibleLLM(llm_raw, model=QWEN_MODEL, provider="qwen")
    print("✅ Qwen LLM initialized successfully and wrapped for browser_use")
except Exception as e:
    print(f"❌ Error initializing Qwen LLM: {e}")
    llm = None

# … rest of your code (UNCHANGED) …

# everything from here is identical to what you pasted — starting from:
# def run_async_in_thread(coro):
# …
# … all the way to …
if __name__ == '__main__':
    print("=== ALT TAG GENERATOR BACKEND STARTING ===")
    print(f"Model: {QWEN_MODEL}")
    print(f"Base URL: {QWEN_BASE_URL}")
    print(f"LLM initialized: {bool(llm)}")
    print("Enhanced image analysis enabled")

    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
