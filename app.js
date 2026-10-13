/**
 * InsectID — Real-Time AI Insect Species Scanner
 * Field camera, motion-triggered auto-scan, multi-provider connection manager, and specimen cards.
 */

const $ = (id) => document.getElementById(id);

const video = $("video"),
  preview = $("preview"),
  reticle = $("reticle"),
  statusEl = $("status"),
  laser = $("laser"),
  viewfinder = $("viewfinder");

const workCanvas = document.createElement("canvas"),
  wctx = workCanvas.getContext("2d");

const tinyCanvas = document.createElement("canvas");
tinyCanvas.width = tinyCanvas.height = 16;
const tctx = tinyCanvas.getContext("2d", { willReadFrequently: true });

const AUTO_INTERVAL = 3500;
const MAX_SIDE = 900; // Optimized for high morphological detail and fast CPU vision inference
const MOTION_THRESHOLD = 8;

let mediaStream = null;
let autoTimer = null;
let isBusy = false;
let lastSignature = null;
let currentFacingMode = "environment";
let currentConfig = { provider: "demo", configured: false, is_demo: true };

const API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "";

let scanHistory = [];
try {
  scanHistory = JSON.parse(localStorage.getItem("insect_history_v2") || "[]");
} catch {
  scanHistory = [];
}

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

/* ==============================================================================
   1. API Connection & Settings Modal
   ============================================================================== */

async function fetchConnectionStatus() {
  const dot = $("connDot");
  const label = $("connLabel");

  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentConfig = data;

    if (data.configured || data.ollama_online) {
      dot.className = "conn-dot online";
      label.textContent = "🟢 Ollama (gemma4:31b)";
    } else {
      dot.className = "conn-dot demo";
      label.textContent = "🟡 Demo Mode (Ollama Offline)";
    }
  } catch (err) {
    dot.className = "conn-dot offline";
    label.textContent = "🔴 Server Offline";
    setStatus("Cannot reach server. Run 'npm run dev' or 'python3 app.py' in your terminal and open http://localhost:8000.", "warn");
  }
}

/* ==============================================================================
   2. Camera Controls & Frame Extraction
   ============================================================================== */

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera requires localhost or HTTPS. You can still use 'Upload photo'.", "warn");
    return;
  }

  stopCameraStreamOnly();
  preview.hidden = true;
  viewfinder.classList.remove("has-preview");

  try {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: currentFacingMode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
    } catch {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    video.srcObject = mediaStream;
    await video.play();
    video.classList.add("on");

    $("start").hidden = true;
    $("stop").hidden = false;
    $("flip").hidden = false;
    $("snap").disabled = false;
    $("auto").disabled = false;
    $("empty").style.display = "none";

    setStatus("Point camera at specimen and tap 'Identify now'.");
  } catch (err) {
    setStatus(
      err.name === "NotAllowedError"
        ? "Camera permission denied. Allow camera access in browser or upload photo."
        : "Could not access camera. Try uploading an image file instead.",
      "warn"
    );
  }
}

function stopCameraStreamOnly() {
  stopAutoScan(true);
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  video.classList.remove("on");
}

function stopCamera() {
  stopCameraStreamOnly();
  $("start").hidden = false;
  $("stop").hidden = true;
  $("flip").hidden = true;
  $("snap").disabled = true;
  $("auto").disabled = true;
  if (preview.hidden) {
    $("empty").style.display = "flex";
  }
  setStatus("Camera offline. Click 'Start camera' or 'Upload photo'.");
}

async function flipCamera() {
  currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
  await startCamera();
}

function captureHighResFrame() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return null;

  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  workCanvas.width = Math.round(w * scale);
  workCanvas.height = Math.round(h * scale);
  wctx.drawImage(video, 0, 0, workCanvas.width, workCanvas.height);
  return workCanvas.toDataURL("image/jpeg", 0.92);
}

// Low-res signature for motion diffing
function getMotionSignature() {
  tctx.drawImage(video, 0, 0, 16, 16);
  const data = tctx.getImageData(0, 0, 16, 16).data;
  const sig = [];
  for (let i = 0; i < data.length; i += 4) {
    sig.push((data[i] + data[i + 1] + data[i + 2]) / 3);
  }
  return sig;
}

function calculateMotionDiff(newSig) {
  if (!lastSignature) return 255;
  return newSig.reduce((acc, val, i) => acc + Math.abs(val - lastSignature[i]), 0) / newSig.length;
}

/* ==============================================================================
   3. Photo Upload Handling
   ============================================================================== */

function initPhotoUpload() {
  const pickBtn = $("pick");
  const fileInput = $("file");

  pickBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    stopCameraStreamOnly();
    $("start").hidden = false;
    $("stop").hidden = true;
    $("flip").hidden = true;
    $("empty").style.display = "none";

    const reader = new FileReader();
    reader.onload = async (evt) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
        workCanvas.width = Math.round(img.width * scale);
        workCanvas.height = Math.round(img.height * scale);
        wctx.drawImage(img, 0, 0, workCanvas.width, workCanvas.height);
        const dataUrl = workCanvas.toDataURL("image/jpeg", 0.92);

        preview.src = dataUrl;
        preview.hidden = false;
        viewfinder.classList.add("has-preview");
        $("snap").disabled = false;

        // Automatically trigger identification
        identify(dataUrl);
      };
      img.src = evt.target.result;
    };
    reader.readAsDataURL(file);
    fileInput.value = "";
  });
}

/* ==============================================================================
   4. Specimen Identification & Card Rendering
   ============================================================================== */

async function identify(imageDataUrl) {
  if (isBusy || !imageDataUrl) return;

  isBusy = true;
  reticle.classList.add("busy");
  laser.hidden = false;
  laser.classList.add("scanning");
  $("reticleText").textContent = "Analyzing anatomy...";
  setStatus("Extracting specimen features and analyzing taxonomy...", "busy");

  try {
    const res = await fetch(`${API_BASE}/api/identify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.detail || `Server identification error (${res.status})`);
    }

    renderResultCard(data, imageDataUrl);
    if (data.is_demo) {
      const reasonHint = data.demo_reason ? ` [Note: ${data.demo_reason.slice(0, 90)}]` : "";
      setStatus(`Live AI unavailable${reasonHint}. Showing offline reference guide.`, "warn");
    } else {
      setStatus(`Specimen classified successfully with Ollama AI (${data.model || "Vision"}).`);
    }
    await fetchConnectionStatus();
  } catch (err) {
    if (err.message === "Failed to fetch" || err.name === "TypeError") {
      setStatus("Cannot reach server. Run 'npm run dev' or 'python3 app.py' in your terminal and open http://localhost:8000.", "warn");
    } else {
      setStatus(err.message, "warn");
    }
  } finally {
    isBusy = false;
    reticle.classList.remove("busy");
    laser.classList.remove("scanning");
    laser.hidden = true;
    $("reticleText").textContent = "Center specimen in frame";
  }
}

function renderResultCard(data, thumbnailDataUrl) {
  const card = $("card");
  const topResult = (data.results || [])[0];

  if (data.is_insect === false || !topResult) {
    card.hidden = true;
    setStatus(data.reason || "No insect or arthropod recognized. Move closer and ensure focus.", "warn");
    return;
  }

  card.hidden = false;

  // Species Titles
  $("name").textContent = topResult.name;
  $("sci").textContent = topResult.scientific && topResult.scientific !== topResult.name ? topResult.scientific : "";
  
  const taxTag = $("taxonomyTag");
  if (topResult.taxonomy || data.taxonomy) {
    taxTag.hidden = false;
    taxTag.textContent = topResult.taxonomy || data.taxonomy;
  } else {
    taxTag.hidden = true;
  }

  // Confidence Bar
  const pct = Math.round((topResult.probability || 0.85) * 100);
  $("bar").style.width = pct + "%";
  $("conf").textContent = `${pct}% Match Confidence`;

  // Danger & Safety Assessment
  const danger = data.danger || "harmless";
  const dangerBadge = $("danger");
  dangerBadge.hidden = false;
  dangerBadge.className = `badge ${danger}`;
  dangerBadge.textContent = danger === "dangerous" ? "⚠️ Dangerous" :
                            danger === "caution" ? "⚡ Caution" : "🛡️ Harmless";

  const safetyBox = $("safetyBox");
  if (data.danger_reason) {
    safetyBox.hidden = false;
    safetyBox.dataset.level = danger;
    $("safetyIcon").textContent = danger === "dangerous" ? "🚨" : danger === "caution" ? "⚠️" : "🛡️";
    $("safetyTitle").textContent = danger === "dangerous" ? "Medical / Safety Precaution" :
                                   danger === "caution" ? "Defense / Sting Precaution" : "Safe Specimen";
    $("safetyReason").textContent = data.danger_reason;
  } else {
    safetyBox.hidden = true;
  }

  // Demo badge & warning notice
  const isDemo = Boolean(data.is_demo);
  $("demoBadge").hidden = !isDemo;
  const demoBanner = $("demoNoticeBanner");
  if (demoBanner) {
    demoBanner.hidden = !isDemo;
  }

  // Description / Anatomy
  $("info").textContent = data.info || "Specimen features align with anatomical database records.";

  // Habitat & Ecological Role
  const habBlock = $("habitatBlock");
  if (data.habitat_diet) {
    habBlock.hidden = false;
    $("habitat").textContent = data.habitat_diet;
  } else {
    habBlock.hidden = true;
  }

  // Field Tip
  const tipBox = $("tipBox");
  if (data.tip) {
    tipBox.hidden = false;
    $("tip").textContent = data.tip;
  } else {
    tipBox.hidden = true;
  }

  // Lookalikes List
  const altsBlock = $("altsBlock");
  const altsList = $("alts");
  altsList.replaceChildren();

  const alts = (data.results || []).slice(1);
  if (alts.length > 0) {
    altsBlock.hidden = false;
    alts.forEach((a) => {
      const li = document.createElement("li");
      li.className = "alt-item";

      const infoWrap = document.createElement("div");
      infoWrap.className = "alt-item-info";

      const nameEl = document.createElement("strong");
      nameEl.textContent = a.name;
      infoWrap.appendChild(nameEl);

      if (a.scientific && a.scientific !== a.name) {
        const sciEl = document.createElement("span");
        sciEl.className = "alt-item-sci";
        sciEl.textContent = ` (${a.scientific})`;
        infoWrap.appendChild(sciEl);
      }

      if (a.distinction) {
        const distEl = document.createElement("div");
        distEl.className = "alt-item-dist";
        distEl.textContent = `• ${a.distinction}`;
        infoWrap.appendChild(distEl);
      }

      const pctEl = document.createElement("span");
      pctEl.className = "alt-item-pct";
      pctEl.textContent = `${Math.round((a.probability || 0) * 100)}%`;

      li.appendChild(infoWrap);
      li.appendChild(pctEl);
      altsList.appendChild(li);
    });
  } else {
    altsBlock.hidden = true;
  }

  // External Reference Link
  const link = $("link");
  const linkUrl = topResult.url || (topResult.scientific ? `https://en.wikipedia.org/wiki/${topResult.scientific.replace(/\s+/g, "_")}` : null);
  if (linkUrl) {
    link.hidden = false;
    link.href = linkUrl;
    link.textContent = `📚 View ${topResult.name} on Wikipedia / iNaturalist →`;
  } else {
    link.hidden = true;
  }

  // Add to Recent Observations
  addToHistory({
    name: topResult.name,
    sci: topResult.scientific || "",
    pct,
    thumb: thumbnailDataUrl,
    date: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    raw: data,
  });

  // Scroll card into view smoothly
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ==============================================================================
   5. Recent Observations History
   ============================================================================== */

function addToHistory(item) {
  scanHistory.unshift(item);
  if (scanHistory.length > 12) scanHistory.pop();
  try {
    localStorage.setItem("insect_history_v2", JSON.stringify(scanHistory));
  } catch {}
  renderHistory();
}

function renderHistory() {
  const box = $("historyBox");
  const list = $("history");
  const count = $("historyCount");

  if (!scanHistory.length) {
    box.hidden = true;
    return;
  }

  box.hidden = false;
  count.textContent = scanHistory.length;
  list.replaceChildren();

  scanHistory.forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-card";
    li.title = `Click to re-open ${item.name}`;

    const img = document.createElement("img");
    img.className = "history-thumb";
    img.src = item.thumb;
    img.alt = item.name;

    const info = document.createElement("div");
    info.className = "history-info";

    const name = document.createElement("div");
    name.className = "history-name";
    name.textContent = item.name;

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.innerHTML = `<span>${item.pct}%</span><span>${item.date}</span>`;

    info.appendChild(name);
    info.appendChild(meta);
    li.appendChild(img);
    li.appendChild(info);

    li.addEventListener("click", () => {
      preview.src = item.thumb;
      preview.hidden = false;
      viewfinder.classList.add("has-preview");
      renderResultCard(item.raw, item.thumb);
    });

    list.appendChild(li);
  });
}

function initHistoryClear() {
  $("clearHistory").addEventListener("click", () => {
    scanHistory = [];
    try {
      localStorage.removeItem("insect_history_v2");
    } catch {}
    renderHistory();
  });
}

/* ==============================================================================
   6. Auto-Scan Motion Detection
   ============================================================================== */

function startAutoScan() {
  if (autoTimer) return;
  lastSignature = null;
  setStatus("Auto-scan enabled. Hold still over an insect.");

  autoTimer = setInterval(async () => {
    if (isBusy || !mediaStream) return;
    const currentSig = getMotionSignature();
    const diff = calculateMotionDiff(currentSig);
    lastSignature = currentSig;

    // Trigger identification if change crosses motion threshold
    if (diff > MOTION_THRESHOLD) {
      const frameData = captureHighResFrame();
      if (frameData) await identify(frameData);
    }
  }, AUTO_INTERVAL);
}

function stopAutoScan(uncheckCheckbox = false) {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
  if (uncheckCheckbox) {
    $("auto").checked = false;
  }
}

/* ==============================================================================
   7. Application Initialization
   ============================================================================== */

function initApp() {
  // Camera buttons
  $("start").addEventListener("click", startCamera);
  $("stop").addEventListener("click", stopCamera);
  $("flip").addEventListener("click", flipCamera);

  $("snap").addEventListener("click", () => {
    if (!preview.hidden) {
      identify(preview.src);
    } else {
      const frameData = captureHighResFrame();
      if (frameData) identify(frameData);
    }
  });

  // Auto-scan checkbox
  $("auto").addEventListener("change", (e) => {
    if (e.target.checked) startAutoScan();
    else stopAutoScan();
  });

  initPhotoUpload();
  
  const pill = $("connectionPill");
  if (pill) {
    pill.addEventListener("click", async () => {
      setStatus("Checking Ollama Vision (gemma4:31b)...", "busy");
      try {
        const res = await fetch(`${API_BASE}/api/test-connection`, { method: "POST" });
        const data = await res.json();
        await fetchConnectionStatus();
        if (res.ok && data.ok) {
          setStatus(`🟢 ${data.message}`);
        } else {
          setStatus(`🟡 Ollama is offline. Run 'ollama serve' in your terminal, or continue scanning in Demo Guide mode.`, "warn");
        }
      } catch (e) {
        await fetchConnectionStatus();
      }
    });
  }

  initHistoryClear();
  renderHistory();
  fetchConnectionStatus();
}

document.addEventListener("DOMContentLoaded", initApp);
