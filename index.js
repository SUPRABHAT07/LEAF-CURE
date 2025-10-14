const INPUT_SIZE = 224; // change if your model expects a different size
const MODEL_URL = "model/model.json";
const LABELS_URL = "model/labels.json"; // optional; falls back to generic labels

let model = null;
let labels = null;
let mediaStream = null;
let sourceEl = null; // <img> or <video>

const statusEl = document.getElementById("status");
const modelInfoEl = document.getElementById("modelInfo");
const predictionsEl = document.getElementById("predictions");
const previewEl = document.getElementById("preview");
const imgInfoEl = document.getElementById("imgInfo");
const hiddenCanvas = document.getElementById("hiddenCanvas");
const fileInput = document.getElementById("file");

// ---------- UI helpers ----------
const setStatus = (msg) => (statusEl.textContent = msg);
const setModelInfo = (msg) => (modelInfoEl.innerHTML = msg);
function clearPredictions() {
  predictionsEl.innerHTML = "";
}
function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

// ---------- Load TF.js & model ----------
window.addEventListener("load", async () => {
  try {
    setStatus("Loading model…");
    model = await tf.loadLayersModel(MODEL_URL);
    labels = await fetch(LABELS_URL)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    if (!labels) {
      labels = Array.from(
        { length: model.outputs[0].shape[1] || 4 },
        (_, i) => `Class ${i}`
      );
    }
    setStatus("Ready");
    setModelInfo(
      `Model: <strong>${
        model.name || "CNN"
      }</strong> · Input ${INPUT_SIZE}×${INPUT_SIZE} · Classes: ${
        labels.length
      }`
    );
    document.getElementById("predictBtn").disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Model not found — using demo mode.");
    setModelInfo(
      "Demo mode: random scores (add /model/model.json to enable real predictions)."
    );
    document.getElementById("predictBtn").disabled = false;
  }
});

// ---------- Image/Webcam handling ----------
function setPreview(el) {
  previewEl.innerHTML = "";
  el.style.maxWidth = "100%";
  el.style.maxHeight = "420px";
  previewEl.appendChild(el);
  sourceEl = el;
  imgInfoEl.textContent =
    el instanceof HTMLVideoElement
      ? "Source: webcam"
      : `Source: ${el.naturalWidth || el.videoWidth}×${
          el.naturalHeight || el.videoHeight
        }`;
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => setPreview(img);
  img.src = URL.createObjectURL(file);
  stopWebcam();
});

const dropzone = document.getElementById("dropzone");
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => setPreview(img);
  img.src = URL.createObjectURL(file);
  stopWebcam();
});

document.getElementById("camBtn").addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    const video = document.createElement("video");
    video.autoplay = true;
    video.playsInline = true;
    video.srcObject = mediaStream;
    setPreview(video);
  } catch (err) {
    alert("Camera access denied/unavailable.");
  }
});

function stopWebcam() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

document.getElementById("clearBtn").addEventListener("click", () => {
  stopWebcam();
  sourceEl = null;
  previewEl.innerHTML = '<span class="subtitle">No image yet</span>';
  clearPredictions();
  imgInfoEl.textContent = "";
});

// ---------- Preprocess & Predict ----------
function preprocess(el) {
  const ctx = hiddenCanvas.getContext("2d");
  hiddenCanvas.width = INPUT_SIZE;
  hiddenCanvas.height = INPUT_SIZE;

  // Letterbox to preserve aspect ratio
  ctx.clearRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  const w = el instanceof HTMLVideoElement ? el.videoWidth : el.naturalWidth;
  const h = el instanceof HTMLVideoElement ? el.videoHeight : el.naturalHeight;
  const scale = Math.min(INPUT_SIZE / w, INPUT_SIZE / h);
  const nw = Math.round(w * scale),
    nh = Math.round(h * scale);
  const nx = Math.floor((INPUT_SIZE - nw) / 2),
    ny = Math.floor((INPUT_SIZE - nh) / 2);
  ctx.drawImage(el, 0, 0, w, h, nx, ny, nw, nh);

  // Build tensor [1, H, W, 3], float32, normalized 0..1
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(hiddenCanvas);
    t = t.toFloat().div(255);
    return t.expandDims(0);
  });
}

async function predict() {
  if (!sourceEl) {
    alert("Please add an image or open the webcam.");
    return;
  }
  clearPredictions();
  setStatus("Running inference…");

  const input = preprocess(sourceEl);
  let probs;
  if (model) {
    const logits = model.predict(input);
    probs = tf.softmax(logits).dataSync();
    tf.dispose([logits]);
  } else {
    // Demo fallback
    probs = Array.from({ length: labels?.length || 4 }, () => Math.random());
    const s = probs.reduce((a, b) => a + b, 0);
    probs = probs.map((p) => p / s);
  }
  tf.dispose([input]);

  const entries = labels
    .map((name, i) => ({ name, p: probs[i] ?? 0 }))
    .sort((a, b) => b.p - a.p);
  renderPredictions(entries);
  setStatus("Ready");
}

function renderPredictions(items) {
  predictionsEl.innerHTML = "";
  items.slice(0, 5).forEach(({ name, p }, idx) => {
    const row = document.createElement("div");
    row.className = "pred-item";
    const left = document.createElement("div");
    left.innerHTML = `<strong>${
      idx === 0 ? "Top:" : ""
    } ${name}</strong><div class="bar" style="margin-top:8px"><span style="width:${(
      p * 100
    ).toFixed(1)}%"></span></div>`;
    const right = document.createElement("div");
    right.style.fontWeight = "700";
    right.textContent = fmtPct(p);
    row.append(left, right);
    predictionsEl.appendChild(row);
  });

  const best = items[0];
  const tips = {
    Healthy:
      "✅ Leaf looks healthy. Keep monitoring and maintain good field hygiene.",
    "Black Sigatoka":
      "⚠️ Consider removing infected leaves and improving airflow; consult local guidance.",
    Cordana:
      "⚠️ Prune affected parts and avoid overhead irrigation; consider fungicide per guidance.",
    "Fusarium Wilt":
      "⚠️ Isolate affected plants; sanitize tools; check resistant cultivars.",
  };
  const msg =
    tips[best?.name] || "ℹ️ Verify results with an expert before acting.";
  const note = document.createElement("div");
  note.className = "status";
  note.style.marginTop = "12px";
  note.textContent = msg;
  predictionsEl.appendChild(note);
}

document.getElementById("predictBtn").addEventListener("click", predict);
