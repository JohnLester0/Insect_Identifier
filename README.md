# 🐛 InsectID — Real-Time Ollama Insect Species Scanner & Field Guide

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-emerald.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Powered by Ollama Vision](https://img.shields.io/badge/Vision%20AI-Ollama%20gemma4%3A31b-black.svg)](https://ollama.com/)

**InsectID** is a streamlined real-time entomological field scanner that identifies insects and arthropods directly from your device camera or uploaded photos. Powered exclusively by **Ollama Vision (`gemma4:31b`)** with **zero model training required**.

---

## ✨ Features

- **🎯 Live Viewfinder & Laser Reticle**: Real-time camera targeting with visual corner guides and pulsing laser scan sweep.
- **⚡ Motion-Detected Auto-Scan**: Compares 16×16 frame signatures and triggers identification automatically when a specimen rests in focus.
- **🔬 Rich Entomological Profiles**:
  - Species Common Name & Latin Scientific Binomial.
  - Taxonomic Classification (*Order / Family*).
  - Calibrated Confidence Match Percentage.
  - Human Safety Assessment (🟢 *Harmless*, 🟡 *Caution / Defense Sting*, 🔴 *Dangerous / Venomous*).
  - Key Anatomical Markers (wing venation, antennae, leg morphology).
  - Field Observation Tips & Habitat / Diet breakdown.
  - Lookalike candidates with comparison notes.
- **🦙 Powered Exclusively by Ollama Vision (`gemma4:31b`)**:
  - Connects directly to local Ollama (`http://127.0.0.1:11434`).
  - No external third-party APIs or subscription keys needed.
  - Built-in entomological specimen guide fallback if Ollama is starting up.
- **📱 Clean & Focused Interface**:
  - Zero bloated settings menus or complicated configurations.
  - Works on phones (macro lens / front camera flipping) and desktop webcams.
- **📜 Scan History**: Persists recent field captures in local storage with thumbnails.

---

## 🚀 Quickstart

### 1. Clone Repository & Setup
```bash
git clone https://github.com/your-username/insect-identifier.git
cd insect-identifier

cp .env.example .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama and Pull the Model
```bash
# Start local Ollama daemon
ollama serve

# In another terminal tab, pull the gemma4 vision model
ollama pull gemma4:31b
```

### 4. Run the Application
```bash
npm run dev
# or: python3 app.py
```

Open your browser at **`http://localhost:8000`**.

---

## 🧪 Testing Ollama Connection

To verify your Ollama model and connection directly in Python:
```bash
python3 test.py
```

---

## 📄 License
MIT License.
