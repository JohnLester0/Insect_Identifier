import os, sys
from pathlib import Path

# Auto-detect local virtualenv packages if installed
_venv_site = Path(__file__).parent / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

# Load .env configuration before importing ollama
if os.path.exists(".env"):
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("'\"")

# Ensure OLLAMA_HOST is set to the configured URL (defaults to http://127.0.0.1:11434)
ollama_host = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
os.environ["OLLAMA_HOST"] = ollama_host

try:
    from ollama import chat

    print(f"Connecting to Ollama at {ollama_host}...")
    response = chat(
        model=os.environ.get('OLLAMA_MODEL', 'gemma4:31b'),
        messages=[{'role': 'user', 'content': 'Hello!'}],
    )
    print("Ollama Response:")
    print(response.message.content)
except Exception as e:
    print(f"\n⚠️ Ollama connection note: {e}")
    print("👉 To run Ollama locally:")
    print("   1. Open a terminal and run: ollama serve")
    print("   2. In another terminal, pull the model: ollama pull gemma4:31b")
    print("   3. Run: python3 test.py\n")
