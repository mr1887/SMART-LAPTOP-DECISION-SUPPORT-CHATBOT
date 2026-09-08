// Frontend Web App Logic for Laptop Support AI Chatbot
// Tích hợp Google Cloud Speech-to-Text & Google Cloud Text-to-Speech
document.addEventListener("DOMContentLoaded", () => {
  // Nếu mở trực tiếp từ file:// thì origin là null, phải trỏ về backend thực
  const BACKEND_URL =
    window.location.origin && window.location.origin !== "null"
      ? window.location.origin
      : "http://localhost:8000";

  // Session ID
  let sessionId = localStorage.getItem("laptop_chat_session_id");
  if (!sessionId) {
    sessionId = "session_" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("laptop_chat_session_id", sessionId);
  }

  // DOM Elements
  const priceRange = document.getElementById("price-range");
  const priceVal = document.getElementById("price-val");
  const gpuType = document.getElementById("gpu-type");
  const gpuKeyword = document.getElementById("gpu-keyword");
  const btnApplyFilter = document.getElementById("btn-apply-filter");
  const btnResetFilter = document.getElementById("btn-reset-filter");
  const activeConstraintsContainer = document.getElementById("active-constraints-container");
  const activeConstraintsList = document.getElementById("active-constraints-list");

  const toggleVoiceAutoplay = document.getElementById("toggle-voice-autoplay");
  const messagesContainer = document.getElementById("messages-container");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  const btnMic = document.getElementById("btn-mic");
  const recordingStatusBar = document.getElementById("recording-status-bar");
  const recordingStatusText = document.getElementById("recording-status-text");
  const btnCancelRecording = document.getElementById("btn-cancel-recording");
  const suggestionChips = document.querySelectorAll(".chip");

  // Audio Recording (Speech-to-Text) State
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let mediaStream = null;
  let isCancelled = false;

  // Price range label sync
  priceRange.addEventListener("input", (e) => {
    priceVal.textContent = e.target.value + " triệu";
  });

  // Auto resize textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  // Send on Enter (Shift+Enter for newline)
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSend.addEventListener("click", () => sendMessage());

  // Suggestion chips
  suggestionChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.getAttribute("data-msg");
      sendMessage();
    });
  });

  // Reset Filter
  btnResetFilter.addEventListener("click", async () => {
    priceRange.value = 25;
    priceVal.textContent = "25 triệu";
    gpuType.value = "any";
    gpuKeyword.value = "";
    document.querySelectorAll(".tag-checkbox input").forEach((cb) => (cb.checked = false));

    try {
      await fetch(`${BACKEND_URL}/api/chat/${sessionId}`, { method: "DELETE" });
    } catch (e) {}

    appendMessage("user", "Đặt lại tiêu chí (Reset)");
    appendMessage(
      "assistant",
      "Đã làm mới toàn bộ bộ lọc và tiêu chí tìm kiếm.\n\nBạn đang cần tìm laptop với ngân sách khoảng bao nhiêu và phục vụ nhu cầu gì (học tập, văn phòng, đồ họa, lập trình hay gaming...)?"
    );
    activeConstraintsContainer.style.display = "none";
  });

  // Apply Filter Button
  btnApplyFilter.addEventListener("click", async () => {
    const selectedTags = Array.from(document.querySelectorAll(".tag-checkbox input:checked")).map((cb) => cb.value);

    let reqDiscrete = null;
    if (gpuType.value === "discrete") reqDiscrete = true;
    else if (gpuType.value === "integrated") reqDiscrete = false;

    const payload = {
      max_price: parseFloat(priceRange.value) * 1000000,
      require_discrete_gpu: reqDiscrete,
      gpu_keyword: gpuKeyword.value.trim() || null,
      required_tags: selectedTags,
    };

    const filterSummaryText = `[Bộ lọc] Ngân sách dưới ${priceRange.value} triệu` + (gpuKeyword.value ? `, GPU ${gpuKeyword.value}` : "");
    appendMessage("user", filterSummaryText);

    const loadingId = appendLoadingMessage();

    try {
      const resp = await fetch(`${BACKEND_URL}/api/constraints`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      removeMessage(loadingId);

      if (!resp.ok) throw new Error("Lỗi kết nối bộ lọc");
      const data = await resp.json();

      handleBackendResponse(data);
    } catch (err) {
      removeMessage(loadingId);
      appendMessage("assistant", "Không thể kết nối đến máy chủ. Vui lòng kiểm tra lại dịch vụ Backend.");
    }
  });

  // -------------------------------------------------------------
  // SPEECH-TO-TEXT (GOOGLE CLOUD STT) MICROPHONE RECORDING LOGIC
  // -------------------------------------------------------------
  if (btnMic) {
    btnMic.addEventListener("click", async () => {
      if (isRecording) {
        stopRecording(false);
      } else {
        await startRecording();
      }
    });
  }

  if (btnCancelRecording) {
    btnCancelRecording.addEventListener("click", () => {
      stopRecording(true);
    });
  }

  async function startRecording() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Trình duyệt của bạn không hỗ trợ ghi âm micro.");
        return;
      }

      audioChunks = [];
      isCancelled = false;

      // Yêu cầu quyền truy cập Micro
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Xác định mimeType phù hợp nhất
      let mimeType = "audio/webm;codecs=opus";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        if (MediaRecorder.isTypeSupported("audio/webm")) {
          mimeType = "audio/webm";
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        } else {
          mimeType = "";
        }
      }

      mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Tắt micro tracks
        if (mediaStream) {
          mediaStream.getTracks().forEach((track) => track.stop());
          mediaStream = null;
        }

        if (isCancelled) {
          resetRecordingUI();
          return;
        }

        if (audioChunks.length === 0) {
          resetRecordingUI();
          return;
        }

        const recordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        await sendAudioToSTT(recordedBlob);
      };

      mediaRecorder.start(250);
      isRecording = true;

      // Update UI
      btnMic.classList.add("recording");
      btnMic.title = "Nhấn để dừng ghi âm và gửi";
      if (recordingStatusBar) recordingStatusBar.style.display = "flex";
      if (recordingStatusText) {
        recordingStatusText.textContent = "Đang lắng nghe giọng nói của bạn... Hãy nói câu hỏi và nhấn lại nút Mic khi xong.";
      }
    } catch (err) {
      console.error("Lỗi mở micro:", err);
      alert("Không thể truy cập Microphone. Vui lòng cho phép quyền truy cập Micro trên trình duyệt.");
      resetRecordingUI();
    }
  }

  function stopRecording(cancelled = false) {
    isCancelled = cancelled;
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    isRecording = false;
  }

  function resetRecordingUI() {
    isRecording = false;
    if (btnMic) {
      btnMic.classList.remove("recording");
      btnMic.title = "Nói câu hỏi qua Micro (Google Cloud STT)";
    }
    if (recordingStatusBar) {
      recordingStatusBar.style.display = "none";
    }
  }

  async function sendAudioToSTT(audioBlob) {
    if (recordingStatusText) {
      recordingStatusText.textContent = "Đang nhận diện giọng nói tiếng Việt bằng Google Cloud STT...";
    }

    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "voice_input.webm");
      formData.append("lang", "vi-VN");

      const resp = await fetch(`${BACKEND_URL}/api/stt`, {
        method: "POST",
        body: formData,
      });

      resetRecordingUI();

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        const errMsg = errData.detail || "Không thể nhận diện giọng nói.";
        console.warn("[STT Error]", errMsg);
        alert(errMsg);
        return;
      }

      const data = await resp.json();
      const transcript = (data.transcript || "").trim();

      if (!transcript) {
        alert("Không nhận diện được giọng nói rõ ràng. Bạn vui lòng nói lại hoặc gõ câu hỏi nhé!");
        return;
      }

      // Điền nội dung nhận diện vào ô chat và gửi đi
      chatInput.value = transcript;
      chatInput.style.height = "auto";
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
      sendMessage();
    } catch (e) {
      console.error("Lỗi gửi audio STT:", e);
      resetRecordingUI();
      alert("Lỗi kết nối dịch vụ Speech-to-Text.");
    }
  }

  // -------------------------------------------------------------
  // SEND MESSAGE & CHAT PROCESSING
  // -------------------------------------------------------------
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = "";
    chatInput.style.height = "auto";

    appendMessage("user", text);
    const loadingId = appendLoadingMessage();

    try {
      const resp = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      removeMessage(loadingId);

      if (!resp.ok) {
        const errText = await resp.text().catch(() => "");
        console.error(`[Chat API Error] Status: ${resp.status}`, errText);
        throw new Error(`API Error: ${resp.status}`);
      }
      const data = await resp.json();

      handleBackendResponse(data);
    } catch (err) {
      console.error("[SendMessage Error]", err);
      removeMessage(loadingId);
      appendMessage("assistant", "Không thể gửi tin nhắn. Vui lòng thử lại sau.");
    }
  }

  function handleBackendResponse(data) {
    const replyText = data.reply || "Đã nhận kết quả.";
    const laptopDetails = data.laptop_details;
    const constraints = data.constraints || {};

    const msgElement = appendMessage("assistant", replyText, laptopDetails);

    // Update active constraints badges
    renderConstraintsBadges(constraints);

    // Audio TTS handling using Google Cloud TTS (/api/tts)
    if (replyText) {
      fetchTTSAudio(replyText, msgElement, toggleVoiceAutoplay.checked);
    }
  }

  function appendMessage(role, text, laptopDetails = null) {
    const msgWrapper = document.createElement("div");
    msgWrapper.className = `message-wrapper ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "BẠN" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    try {
      bubble.innerHTML = (typeof marked !== "undefined" && marked.parse) ? marked.parse(text) : text;
    } catch (e) {
      bubble.textContent = text;
    }

    // If laptop details exist, add media action links (YouTube & Image Search)
    if (laptopDetails) {
      const mediaDiv = document.createElement("div");
      mediaDiv.className = "media-actions";

      if (laptopDetails.review_video_url) {
        const vLink = document.createElement("a");
        vLink.className = "btn-link";
        vLink.href = laptopDetails.review_video_url;
        vLink.target = "_blank";
        vLink.textContent = "Xem Đánh Giá Trên YouTube";
        mediaDiv.appendChild(vLink);
      }

      if (laptopDetails.image_search_url) {
        const iLink = document.createElement("a");
        iLink.className = "btn-link";
        iLink.href = laptopDetails.image_search_url;
        iLink.target = "_blank";
        iLink.textContent = "Xem Thư Viện Ảnh Chi Tiết";
        mediaDiv.appendChild(iLink);
      }

      bubble.appendChild(mediaDiv);
    }

    msgWrapper.appendChild(avatar);
    msgWrapper.appendChild(bubble);
    messagesContainer.appendChild(msgWrapper);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return msgWrapper;
  }

  function appendLoadingMessage() {
    const id = "loading_" + Date.now();
    const msgWrapper = document.createElement("div");
    msgWrapper.className = "message-wrapper assistant";
    msgWrapper.id = id;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "AI";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = "Hệ thống AI & Solver đang phân tích...";

    msgWrapper.appendChild(avatar);
    msgWrapper.appendChild(bubble);
    messagesContainer.appendChild(msgWrapper);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return id;
  }

  function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  // -------------------------------------------------------------
  // TEXT-TO-SPEECH (GOOGLE CLOUD TTS) AUDIO PLAYER
  // -------------------------------------------------------------
  async function fetchTTSAudio(text, msgWrapper, autoPlay = false) {
    try {
      const resp = await fetch(`${BACKEND_URL}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          lang: "vi-VN",
          voice_name: "vi-VN-Neural2-A",
        }),
      });

      if (!resp.ok) return;

      const blob = await resp.blob();
      const audioUrl = URL.createObjectURL(blob);

      const audioDiv = document.createElement("div");
      audioDiv.className = "audio-controls";

      const audioEl = document.createElement("audio");
      audioEl.controls = true;
      audioEl.src = audioUrl;

      if (autoPlay) {
        audioEl.play().catch(() => {});
      }

      audioDiv.appendChild(audioEl);
      const bubble = msgWrapper.querySelector(".message-bubble");
      if (bubble) bubble.appendChild(audioDiv);
    } catch (e) {
      console.warn("Google Cloud TTS audio fetch warning:", e);
    }
  }

  function renderConstraintsBadges(constraints) {
    activeConstraintsList.innerHTML = "";
    let count = 0;

    if (constraints.max_price) {
      addBadge(`Ngân sách ≤ ${constraints.max_price / 1000000}tr`);
      count++;
    }
    if (constraints.require_discrete_gpu === true) {
      addBadge("Card rời");
      count++;
    } else if (constraints.require_discrete_gpu === false) {
      addBadge("Card tích hợp");
      count++;
    }
    if (constraints.gpu_keyword) {
      addBadge(`GPU ${constraints.gpu_keyword}`);
      count++;
    }

    if (count > 0) {
      activeConstraintsContainer.style.display = "block";
    } else {
      activeConstraintsContainer.style.display = "none";
    }
  }

  function addBadge(text) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = text;
    activeConstraintsList.appendChild(b);
  }
});
