const preview = document.getElementById('preview');
const imgInfo = document.getElementById('imgInfo');
const leafInput = document.getElementById('leaf_image');
const camBtn = document.getElementById('camBtn');
const predictBtn = document.getElementById('predictBtn');

let videoStream;
let videoElement;
let captureBtn;

// Preview uploaded file
function previewFile(event){
  stopCamera();
  preview.innerHTML = "";
  const file = event.target.files[0];
  if(!file) return;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
  imgInfo.innerText = file.name;
  predictBtn.disabled = false;
}

// Clear preview
function clearPreview(){
  preview.innerHTML = "<span class='subtitle'>No image yet</span>";
  leafInput.value = "";
  imgInfo.innerText = "";
  predictBtn.disabled = true;
  stopCamera();
}

// Start webcam
camBtn.addEventListener('click', async () => {
  stopCamera();

  videoElement = document.createElement('video');
  videoElement.autoplay = true;
  videoElement.style.maxWidth = "100%";
  preview.innerHTML = "";
  preview.appendChild(videoElement);

  captureBtn = document.createElement('button');
  captureBtn.type = 'button';
  captureBtn.className = 'btn';
  captureBtn.innerText = '📸 Capture';
  captureBtn.style.marginTop = '8px';
  preview.appendChild(captureBtn);

  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = videoStream;
  } catch (err) {
    alert("Could not access camera: " + err.message);
    return;
  }

  captureBtn.addEventListener('click', () => {
    // Capture frame to canvas
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert to blob and set to leafInput (hidden upload)
    canvas.toBlob(blob => {
      const file = new File([blob], "captured.png", { type: "image/png" });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      leafInput.files = dataTransfer.files;

      // Show captured image in preview
      preview.innerHTML = "";
      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      preview.appendChild(img);
      imgInfo.innerText = file.name;
    }, "image/png");

    stopCamera();
  });

  predictBtn.disabled = false;
});

// Stop webcam
function stopCamera(){
  if(videoStream){
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  if(captureBtn){
    captureBtn.remove();
    captureBtn = null;
  }
}
