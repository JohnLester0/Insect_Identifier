import os, sys
from pathlib import Path

# Auto-detect local virtualenv packages if installed
_venv_site = Path(__file__).parent / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

# Load .env if present
if os.path.exists(".env"):
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

if "OLLAMA_URL" in os.environ:
    os.environ["OLLAMA_HOST"] = os.environ["OLLAMA_URL"]

from ollama import chat

image_path = 'insect.jpg'
images = [image_path] if os.path.exists(image_path) else []

if not images:
    print(f"ℹ️ Note: '{image_path}' not found in current directory. Place an 'insect.jpg' image here for vision identification.\n")

response = chat(
    model='gemma4:12b',
    messages=[{
        'role': 'user',
        'content': 'What insect is this? Give the common name and scientific name.',
        'images': images,
    }],
)

print(response.message.content)
