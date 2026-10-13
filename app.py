"""
InsectID: Real-Time Insect Species Identification & Field Guide.
Powered exclusively by Ollama (Local & Cloud Vision AI).
Supports gemma4, llama3.2-vision, llava, and offline demo specimen fallback.
"""
import os, sys, re, json, time, base64, binascii, collections, mimetypes, asyncio
from datetime import date
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import httpx

# Auto-include local venv packages if available
_venv_site = Path(__file__).parent / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False

ENV_FILE = Path(".env")
_last_env_mtime = 0.0

PLACEHOLDERS = {
    "", "your-key-here", "your_key_here", "your-actual-key-goes-here",
    "sk-...", "aiza...", "none"
}

def load_env():
    global _last_env_mtime
    if ENV_FILE.exists():
        try:
            mtime = ENV_FILE.stat().st_mtime
            if mtime != _last_env_mtime:
                _last_env_mtime = mtime
                for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")
                if "OLLAMA_URL" in os.environ:
                    os.environ["OLLAMA_HOST"] = os.environ["OLLAMA_URL"]
        except Exception:
            pass

load_env()

hits = collections.defaultdict(collections.deque)
daily = {"day": "", "n": 0}

def get_env_config():
    load_env()
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip()
    ollama_key = os.getenv("OLLAMA_API_KEY", "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2-vision").strip()

    return {
        "provider": "ollama",
        "ollama_url": ollama_url,
        "ollama_key": ollama_key,
        "ollama_model": ollama_model,
        "rate": int(os.getenv("RATE_LIMIT_PER_MIN", "30")),
        "daily_cap": int(os.getenv("DAILY_SCAN_CAP", "1000")),
    }

def is_provider_configured(cfg: dict) -> bool:
    return bool(cfg.get("ollama_url"))

def check_ollama_reachable(url="http://127.0.0.1:11434") -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=0.6) as r:
            return r.status == 200
    except Exception:
        return False

def get_active_provider_status():
    cfg = get_env_config()
    url = cfg.get("ollama_url", "http://127.0.0.1:11434")
    online = check_ollama_reachable(url)
    model = cfg.get("ollama_model", "llama3.2-vision")
    return {
        "ok": True,
        "provider": "ollama",
        "provider_display": f"Ollama Vision ({model})",
        "configured": online,
        "ollama_online": online,
        "ollama_url": url,
        "ollama_model": model,
    }

def rate_limited(ip: str = "?") -> bool:
    cfg = get_env_config()
    q, now = hits[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= cfg["rate"]:
        return True
    q.append(now)
    return False

def over_daily_cap() -> bool:
    cfg = get_env_config()
    today = str(date.today())
    if daily["day"] != today:
        daily.update(day=today, n=0)
    if daily["n"] >= cfg["daily_cap"]:
        return True
    daily["n"] += 1
    return False

def read_image(data_url: str):
    if not isinstance(data_url, str):
        raise ValueError("Invalid image input.")
    m = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", data_url.strip())
    if not m:
        raise ValueError("Send an image as a data: URL (JPEG, PNG, or WebP).")
    if m.group(1).lower() not in {"image/jpeg", "image/png", "image/webp", "image/jpg"}:
        raise ValueError("Send a JPEG, PNG, or WebP image.")
    try:
        raw = base64.b64decode(m.group(2), validate=True)
        if len(raw) < 100:
            raise ValueError("Image data is too small to be valid.")
    except (binascii.Error, ValueError):
        raise ValueError("The image data is not valid base64.")
    return m.group(1), m.group(2)

def provider_error(status: int, text: str = ""):
    if status in (401, 403):
        return 401, "The API key was rejected by Ollama. Please check your key on ollama.com."
    if status in (402, 429):
        return 402, "Rate limited by Ollama. Please try again shortly."
    return 502, f"Ollama returned an error ({status})."

# ==============================================================================
# Entomological System Prompt for High-Accuracy Vision Classification
# ==============================================================================
SYSTEM_PROMPT = (
    "You are an expert entomologist and taxonomist specializing in precise insect and arthropod identification from photographs.\n"
    "Carefully analyze this image for visible morphological characteristics:\n"
    "- Body plan: 3 segments (head, thorax, abdomen) with 6 legs (insect), or cephalothorax with 8 legs (arachnid), or many body rings (myriapod).\n"
    "- Wings: Presence of elytra (beetle wing covers), scale patterns (butterfly/moth), membranous venation (bee/wasp/dragonfly), or dipteran halteres (true fly).\n"
    "- Head structures: Antenna morphology (clubbed, feathery, threadlike, elbowed), eyes, and mouthparts (chewing mandibles, proboscis, piercing stylet).\n"
    "- Distinguish known mimics: e.g. hoverfly mimicking a bee; Viceroy vs Monarch; Asian lady beetle vs native ladybug; Assassin bug vs Kissing bug.\n\n"
    "Respond ONLY with a valid JSON object strictly adhering to this format:\n"
    "{\n"
    '  "is_insect": true,\n'
    '  "common_name": "Accurate Common Name",\n'
    '  "scientific_name": "Genus species",\n'
    '  "taxonomy": "Order: [Order Name] | Family: [Family Name]",\n'
    '  "confidence": 0.92,\n'
    '  "danger": "harmless" | "caution" | "dangerous",\n'
    '  "danger_reason": "Clear explanation of stinging, biting, allergen, or harmless pollinator status.",\n'
    '  "summary": "Key morphological features (antenna type, wing venation, coloration, body shape) confirming this identification.",\n'
    '  "habitat_diet": "Usual habitat, diet, and ecological niche.",\n'
    '  "tip": "Field observer identification tip.",\n'
    '  "alts": [\n'
    '    {"name": "Lookalike Species", "scientific": "Genus species", "distinction": "Primary visual difference", "confidence": 0.08}\n'
    '  ]\n'
    "}\n"
    "If the image does not show an insect or arthropod, set is_insect to false with a reason."
)

def parse_ai_json(text: str, provider_name: str = "ollama") -> dict:
    cleaned = text.strip()
    # Strip markdown code blocks
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)

    # 1. Check if model explicitly states no insect / non-arthropod is present
    not_insect_phrases = [
        "not an insect", "no insect", "not an arthropod", "no arthropod",
        "cannot see any insect", "unable to detect an insect", "not a bug",
        "no specimen visible", "is_insect\": false", "is_insect: false"
    ]
    if any(phrase in cleaned.lower() for phrase in not_insect_phrases):
        return {
            "provider": provider_name,
            "is_insect": False,
            "reason": "No insect or arthropod recognized in this image. Move closer, focus on the specimen, and ensure good lighting.",
            "results": [],
        }

    # 2. Extract JSON structure if present
    m = re.search(r"\{.*\}", cleaned, re.S)
    raw_json = m.group(0) if m else cleaned
    raw_json = re.sub(r",\s*([\]}])", r"\1", raw_json)

    j = None
    try:
        j = json.loads(raw_json)
    except Exception:
        pass

    # 3. Resilient field extraction (handles key-value and natural language responses)
    if not j or not isinstance(j, dict):
        name_m = (
            re.search(r'"common_name"\s*:\s*"([^"]+)"', cleaned)
            or re.search(r'common_name\s*[:=]\s*["\']?([^"\',\n]+)', cleaned, re.I)
            or re.search(r'(?:common name|species|insect name|identified as|this is an?|shows an?)\s*[:=-]?\s*["\']?([A-Za-z0-9\s\-]+?)(?:\.|\n|\(|,|$)', cleaned, re.I)
        )
        sci_m = (
            re.search(r'"scientific_name"\s*:\s*"([^"]+)"', cleaned)
            or re.search(r'scientific_name\s*[:=]\s*["\']?([^"\',\n]+)', cleaned, re.I)
            or re.search(r'\(([A-Z][a-z]+ [a-z]+)\)', cleaned)
            or re.search(r'scientific name\s*[:=-]?\s*["\']?([A-Za-z0-9\s\-]+?)(?:\.|\n|\)|,|$)', cleaned, re.I)
        )
        tax_m = (
            re.search(r'"taxonomy"\s*:\s*"([^"]+)"', cleaned)
            or re.search(r'taxonomy\s*[:=]\s*["\']?([^"\',\n]+)', cleaned, re.I)
            or re.search(r'(?:Order|Family)\s*[:=]\s*([^\n]+)', cleaned, re.I)
        )
        summary_m = re.search(r'"summary"\s*:\s*"([^"]+)"', cleaned)
        danger_m = (
            re.search(r'"danger"\s*:\s*"([^"]+)"', cleaned)
            or re.search(r'danger\s*[:=]\s*["\']?(harmless|caution|dangerous)', cleaned, re.I)
        )

        extracted_name = name_m.group(1).strip() if name_m else ""
        if not extracted_name or len(extracted_name) < 2:
            first_line = cleaned.split("\n")[0][:80].strip()
            extracted_name = first_line or "Observed Insect Specimen"

        # Determine danger level heuristically if not explicit
        danger_val = "harmless"
        lower_txt = cleaned.lower()
        if any(w in lower_txt for w in ("deadly", "highly venomous", "dangerous sting", "medical attention")):
            danger_val = "dangerous"
        elif any(w in lower_txt for w in ("sting", "bite", "irritant", "caution", "wasp", "bee", "mosquito", "ant")):
            danger_val = "caution"

        j = {
            "is_insect": True,
            "common_name": extracted_name,
            "scientific_name": sci_m.group(1).strip() if sci_m else "",
            "taxonomy": tax_m.group(1).strip() if tax_m else "Class: Insecta",
            "summary": summary_m.group(1).strip() if summary_m else cleaned[:500],
            "danger": danger_m.group(1).strip().lower() if danger_m else danger_val,
            "confidence": 0.88,
        }

    is_insect = bool(j.get("is_insect", True))
    if not is_insect:
        return {
            "provider": provider_name,
            "is_insect": False,
            "reason": j.get("reason", "No insect detected in this frame. Ensure specimen is centered and in focus."),
            "results": [],
        }

    top_name = j.get("common_name") or j.get("scientific_name") or "Unidentified Insect"
    top_sci = j.get("scientific_name", "")
    top_prob = max(0.0, min(1.0, float(j.get("confidence") or 0.88)))

    results = [{
        "name": top_name,
        "scientific": top_sci,
        "probability": top_prob,
        "taxonomy": j.get("taxonomy", ""),
        "distinction": "Primary specimen match",
        "url": f"https://en.wikipedia.org/wiki/{top_sci.replace(' ', '_')}" if top_sci else None,
    }]

    for alt in (j.get("alts") or [])[:3]:
        alt_name = alt.get("name", "")
        alt_sci = alt.get("scientific", "")
        if alt_name or alt_sci:
            results.append({
                "name": alt_name or alt_sci,
                "scientific": alt_sci,
                "probability": max(0.0, min(1.0, float(alt.get("confidence") or 0.1))),
                "distinction": alt.get("distinction", "Visual lookalike"),
                "url": f"https://en.wikipedia.org/wiki/{alt_sci.replace(' ', '_')}" if alt_sci else None,
            })

    danger = str(j.get("danger", "harmless")).lower()
    if danger not in ("harmless", "caution", "dangerous"):
        danger = "harmless"

    return {
        "provider": provider_name,
        "is_insect": True,
        "results": results,
        "info": j.get("summary", ""),
        "taxonomy": j.get("taxonomy", ""),
        "habitat_diet": j.get("habitat_diet", ""),
        "danger": danger,
        "danger_reason": j.get("danger_reason", ""),
        "tip": j.get("tip", ""),
    }

KNOWN_VISION_MODELS = [
    "moondream",
    "moondream:latest",
    "llama3.2-vision",
    "llama3.2-vision:11b",
    "llama3.2-vision:90b",
    "llava",
    "llava:7b",
    "llava:13b",
    "llava:34b",
    "minicpm-v",
    "qwen2.5-vl",
    "bakllava",
]

async def resolve_vision_model(url: str, preferred: str = "") -> str:
    pref = (preferred or "").strip()
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{url}/api/tags")
            if r.status_code == 200:
                installed = [m.get("name", "") for m in r.json().get("models", [])]
                if pref:
                    for inst in installed:
                        if inst == pref or inst.split(":")[0] == pref.split(":")[0]:
                            return inst
                for candidate in KNOWN_VISION_MODELS:
                    for inst in installed:
                        if candidate in inst or inst.split(":")[0] == candidate.split(":")[0]:
                            return inst
                if installed:
                    return installed[0]
    except Exception:
        pass

    return pref or "moondream"

# ---------- Ollama Vision (Official SDK + HTTP Fallback) ----------
async def ollama_vision(mime, b64, cfg):
    primary_url = (cfg.get("ollama_url") or "http://127.0.0.1:11434").rstrip("/")
    fallback_url = "http://127.0.0.1:11434"
    urls_to_try = [primary_url] if primary_url == fallback_url else [primary_url, fallback_url]

    model = await resolve_vision_model(primary_url, cfg.get("ollama_model"))
    api_key = cfg.get("ollama_key", "")

    headers = {}
    if api_key and api_key.lower() not in PLACEHOLDERS:
        headers["Authorization"] = f"Bearer {api_key}"

    user_content = SYSTEM_PROMPT + "\nAnalyze and identify the insect or arthropod in this photo."

    # 1. Official Ollama Python SDK
    last_err_text = ""
    if HAS_OLLAMA:
        try:
            client = ollama.AsyncClient(host=primary_url, headers=headers)
            text = ""
            try:
                res = await client.chat(
                    model=model,
                    messages=[{"role": "user", "content": user_content, "images": [b64]}],
                    format="json",
                    options={"temperature": 0.1},
                )
                text = res.message.content if hasattr(res, "message") else str(res)
            except Exception:
                text = ""

            if not text or text.strip() in ("{}", "[]", ""):
                res = await client.chat(
                    model=model,
                    messages=[{"role": "user", "content": user_content, "images": [b64]}],
                    options={"temperature": 0.1},
                )
                text = res.message.content if hasattr(res, "message") else str(res)

            parsed = parse_ai_json(text, "ollama")
            parsed["model"] = model
            return parsed
        except Exception as e:
            last_err_text = str(e)

    # 2. Direct HTTP Fallback
    r = None
    async with httpx.AsyncClient(timeout=60) as h:
        for url in urls_to_try:
            try:
                chat_url = f"{url}/api/chat"
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": SYSTEM_PROMPT + "\nAnalyze and identify the insect or arthropod in this photo.",
                            "images": [b64],
                        }
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
                res = await h.post(chat_url, headers=headers, json=payload)
                if res.status_code == 200:
                    r = res
                    break
                elif res.status_code in (401, 403):
                    code, msg = provider_error(res.status_code, res.text)
                    raise RuntimeError(f"HTTP_{code}_{msg}")
                else:
                    last_err_text = f"Ollama HTTP {res.status_code}: {res.text[:120]}"
            except RuntimeError:
                raise
            except Exception as e:
                last_err_text = str(e)

            try:
                openai_url = f"{url}/v1/chat/completions"
                res = await h.post(
                    openai_url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                                    {"type": "text", "text": "Analyze and identify the insect or arthropod in this photo."},
                                ],
                            },
                        ],
                        "response_format": {"type": "json_object"},
                    },
                )
                if res.status_code == 200:
                    r = res
                    break
                elif res.status_code in (401, 403):
                    code, msg = provider_error(res.status_code, res.text)
                    raise RuntimeError(f"HTTP_{code}_{msg}")
                else:
                    last_err_text = f"Ollama HTTP {res.status_code}: {res.text[:120]}"
            except RuntimeError:
                raise
            except Exception as e:
                last_err_text = str(e)

    if r is None or r.status_code >= 400:
        err_msg = f"Cannot reach Ollama at {primary_url} ({last_err_text}). Run 'ollama serve' in your terminal."
        raise RuntimeError(f"HTTP_502_{err_msg}")

    data = r.json()
    if "message" in data and "content" in data["message"]:
        text = data["message"]["content"]
    elif "choices" in data and len(data["choices"]) > 0:
        text = data["choices"][0].get("message", {}).get("content", "")
    elif "response" in data:
        text = data["response"]
    else:
        text = json.dumps(data)

    parsed = parse_ai_json(text, "ollama")
    parsed["model"] = model
    return parsed

# ---------- Fallback Demo Specimens (if Ollama offline) ----------
DEMO_SPECIMENS = [
    {
        "common_name": "Monarch Butterfly",
        "scientific_name": "Danaus plexippus",
        "taxonomy": "Order: Lepidoptera | Family: Nymphalidae",
        "confidence": 0.96,
        "danger": "harmless",
        "danger_reason": "Harmless pollinator; sequestered cardenolides make it toxic to birds if eaten, but no sting or human danger.",
        "summary": "Distinctive brilliant orange wings laced with black veins and double white marginal dot rows. Clubbed antennae characteristic of butterflies.",
        "habitat_diet": "Open fields, meadows, and milkweed patches; adults feed on wildflower nectar.",
        "tip": "Plant native Asclepias (milkweed) species to support migrating populations.",
        "alts": [
            {"name": "Viceroy Butterfly", "scientific": "Limenitis archippus", "distinction": "Has a distinct black horizontal postmedian line traversing the hindwing veins.", "confidence": 0.04}
        ]
    },
    {
        "common_name": "European Honeybee",
        "scientific_name": "Apis mellifera",
        "taxonomy": "Order: Hymenoptera | Family: Apidae",
        "confidence": 0.94,
        "danger": "caution",
        "danger_reason": "Possesses a barbed stinger used defensively if hive is threatened or if handled; vital ecological pollinator.",
        "summary": "Golden-brown abdomen with dark horizontal banding, densely pubescent thorax, and specialized pollen baskets (corbiculae) on hind tibiae.",
        "habitat_diet": "Meadows, agricultural gardens, and orchards; feeds on floral pollen and nectar.",
        "tip": "Stay calm if one lands on you; slow movements prevent defensive stinging.",
        "alts": [
            {"name": "Syrphid Hoverfly", "scientific": "Syrphidae sp.", "distinction": "Mimic with only one pair of wings and short stubby antennae; cannot sting.", "confidence": 0.05}
        ]
    },
    {
        "common_name": "Chinese Praying Mantis",
        "scientific_name": "Tenodera sinensis",
        "taxonomy": "Order: Mantodea | Family: Mantidae",
        "confidence": 0.97,
        "danger": "harmless",
        "danger_reason": "Non-venomous and harmless to humans; fierce ambush predator that controls agricultural pest populations.",
        "summary": "Elongated prothorax with powerful spiked raptorial forelegs. Triangular head rotates 180 degrees with large compound eyes.",
        "habitat_diet": "Tall vegetation, shrubs, and garden flowers; preys on crickets, moths, beetles, and flies.",
        "tip": "Beneficial garden ally that hunts nuisance insects without damaging plants.",
        "alts": [
            {"name": "Carolina Mantis", "scientific": "Stagmomantis carolina", "distinction": "Slightly smaller with distinct wing spots and shorter tegmina.", "confidence": 0.03}
        ]
    }
]

_demo_index = 0

def demo_classifier():
    global _demo_index
    spec = DEMO_SPECIMENS[_demo_index % len(DEMO_SPECIMENS)]
    _demo_index += 1

    top_name = spec["common_name"]
    top_sci = spec["scientific_name"]
    results = [{
        "name": top_name,
        "scientific": top_sci,
        "probability": spec["confidence"],
        "taxonomy": spec["taxonomy"],
        "distinction": "Primary specimen match",
        "url": f"https://en.wikipedia.org/wiki/{top_sci.replace(' ', '_')}",
    }]
    for a in spec.get("alts", []):
        results.append({
            "name": a["name"],
            "scientific": a["scientific"],
            "probability": a["confidence"],
            "distinction": a.get("distinction", "Lookalike"),
            "url": f"https://en.wikipedia.org/wiki/{a['scientific'].replace(' ', '_')}",
        })

    return {
        "provider": "ollama (demo fallback)",
        "is_insect": True,
        "is_demo": True,
        "results": results,
        "info": spec["summary"],
        "taxonomy": spec["taxonomy"],
        "habitat_diet": spec["habitat_diet"],
        "danger": spec["danger"],
        "danger_reason": spec["danger_reason"],
        "tip": spec["tip"],
    }

async def handle_test_connection():
    cfg = get_env_config()
    target_url = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
    async with httpx.AsyncClient(timeout=3) as client:
        try:
            r = await client.get(f"{target_url}/api/tags")
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                model_list = ", ".join(models[:4]) if models else "Ready (no models downloaded yet)"
                return 200, {"ok": True, "message": f"Connected to Ollama at {target_url}! Available models: {model_list}"}
        except Exception:
            pass

    return 502, {"ok": False, "error": f"Ollama is offline at {target_url}. Run 'ollama serve' in your terminal to enable live AI."}

async def handle_identify(image_data: str, client_ip: str):
    cfg = get_env_config()

    if rate_limited(client_ip):
        return 429, {"detail": "Rate limit exceeded. Please wait a few seconds before scanning again."}
    if over_daily_cap():
        return 429, {"detail": "Daily scan limit reached. Please check back tomorrow."}

    try:
        mime, b64 = read_image(image_data)
    except ValueError as e:
        return 400, {"detail": str(e)}

    try:
        res = await ollama_vision(mime, b64, cfg)
        return 200, res
    except Exception as e:
        print(f"[Ollama Vision Error]: {e}", flush=True)
        # Fallback to local specimen field guide when Ollama daemon is offline
        demo_spec = demo_classifier()
        demo_spec["is_demo"] = True
        demo_spec["demo_reason"] = str(e)
        return 200, demo_spec

# ==============================================================================
# Built-in HTTP Server
# ==============================================================================
class BuiltinHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, status, content, mime="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self._send(200, b"", "text/plain")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        p = self.path.split("?", 1)[0]

        # 1. Root and HTML
        if p in ("/", "/index.html"):
            for cand in [Path("index.html"), Path("static/index.html")]:
                if cand.exists():
                    self._send(200, cand.read_bytes(), "text/html; charset=utf-8")
                    return

        # 2. Static CSS / JS
        if p in ("/style.css", "/static/style.css"):
            for cand in [Path("style.css"), Path("static/style.css")]:
                if cand.exists():
                    self._send(200, cand.read_bytes(), "text/css; charset=utf-8")
                    return

        if p in ("/app.js", "/static/app.js"):
            for cand in [Path("app.js"), Path("static/app.js")]:
                if cand.exists():
                    self._send(200, cand.read_bytes(), "application/javascript; charset=utf-8")
                    return

        if p.startswith("/static/"):
            rel = p.lstrip("/static/")
            for cand in [Path("static") / rel, Path(rel)]:
                if cand.exists() and cand.is_file():
                    mime, _ = mimetypes.guess_type(str(cand))
                    self._send(200, cand.read_bytes(), mime or "application/octet-stream")
                    return

        # 3. Health & Config status
        if p in ("/health", "/api/config", "/api/status"):
            body = json.dumps(get_active_provider_status()).encode()
            self._send(200, body)
            return

        self._send(404, b'{"detail": "Not found"}')

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if p == "/api/identify":
            img_data = payload.get("image", "")
            ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
            status, data = asyncio.run(handle_identify(img_data, ip))
            self._send(status, json.dumps(data).encode("utf-8"))
            return

        elif p == "/api/test-connection":
            status, res = asyncio.run(handle_test_connection())
            self._send(status, json.dumps(res).encode("utf-8"))
            return

        self._send(404, b'{"detail": "Not found"}')

def main():
    host = os.getenv("HOST", "0.0.0.0")
    desired_port = int(os.getenv("PORT", "8000"))

    class ReusableServer(HTTPServer):
        allow_reuse_address = True

    server = None
    port = desired_port
    for attempt_port in range(desired_port, desired_port + 25):
        try:
            server = ReusableServer((host, attempt_port), BuiltinHandler)
            port = attempt_port
            break
        except OSError as e:
            if e.errno == 98 or "already in use" in str(e).lower():
                continue
            raise

    if server is None:
        raise OSError(f"Could not bind to any port between {desired_port} and {desired_port + 24}.")

    if port != desired_port:
        print(f"ℹ️ Port {desired_port} is busy. Switched to port {port}.")
    print(f"🚀 InsectID (Ollama Vision) is running!")
    print(f"👉 Open in your web browser: http://localhost:{port} (or http://127.0.0.1:{port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down InsectID.")
        server.server_close()

if __name__ == "__main__":
    main()
