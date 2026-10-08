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
    const recommendedLaptops = data.recommended_laptops;
    const recommendations = data.recommendations;
    const constraints = data.constraints || {};
    const result = data.result || {};

    const isNearestFallback = (
      result.is_feasible === false ||
      result.status === "RELAXED" ||
      (recommendations && recommendations.some((r) => r.type === "nearest_alternative")) ||
      (recommendedLaptops && recommendedLaptops.some((l) => l.type === "nearest_alternative"))
    );

    const msgElement = appendMessage(
      "assistant",
      replyText,
      laptopDetails,
      recommendedLaptops,
      recommendations,
      isNearestFallback
    );

    // Update active constraints badges
    renderConstraintsBadges(constraints);

    // Audio TTS handling using Google Cloud TTS (/api/tts) - ONLY if enabled
    if (toggleVoiceAutoplay && toggleVoiceAutoplay.checked && replyText) {
      fetchTTSAudio(replyText, msgElement, true);
    }
  }

  function formatPrice(val) {
    if (!val || isNaN(val)) return "";
    const num = Number(val);
    if (num >= 1000000) {
      const tr = num / 1000000;
      return tr % 1 === 0 ? `${tr} triệu VNĐ` : `${tr.toFixed(1)} triệu VNĐ`;
    }
    return num.toLocaleString("vi-VN") + " VNĐ";
  }

  function createSpecItem(label, val) {
    const item = document.createElement("div");
    item.className = "card-spec-item";
    const l = document.createElement("span");
    l.className = "card-spec-label";
    l.textContent = label + ":";
    const v = document.createElement("span");
    v.className = "card-spec-val";
    v.textContent = val;
    item.appendChild(l);
    item.appendChild(v);
    return item;
  }

  function renderTop3Cards(recommendedLaptops, recommendations, isNearestFallback = false) {
    if (!recommendedLaptops || recommendedLaptops.length === 0) return null;

    const wrapper = document.createElement("div");
    wrapper.className = "top3-wrapper";

    // Cảnh báo nếu là phương án gần nhất (nearest_alternative fallback)
    const hasNearest = isNearestFallback || recommendedLaptops.some((l, idx) => {
      const recMeta = (recommendations && recommendations[idx]) ? recommendations[idx] : {};
      return (l.type === "nearest_alternative" || recMeta.type === "nearest_alternative");
    });

    if (hasNearest) {
      const warningEl = document.createElement("div");
      warningEl.className = "nearest-alt-warning";
      warningEl.innerHTML = '<span class="warning-icon">⚠️</span><span>Không có laptop thỏa hoàn toàn các yêu cầu bắt buộc. Đây là phương án gần nhất.</span>';
      wrapper.appendChild(warningEl);
    }

    const container = document.createElement("div");
    container.className = "top3-container";

    recommendedLaptops.forEach((laptop, idx) => {
      if (!laptop) return;
      const recMeta = (recommendations && recommendations[idx]) ? recommendations[idx] : {};
      const rank = laptop.rank || recMeta.rank || (idx + 1);
      const type = laptop.type || recMeta.type || (idx === 0 ? "best_match" : (idx === 1 ? "budget_alternative" : "performance_alternative"));

      let badgeLabel = "Gợi ý";
      let badgeClass = "badge-best-match";
      let cardClass = "best-match";

      if (type === "best_match") {
        badgeLabel = "Phù hợp nhất";
        badgeClass = "badge-best-match";
        cardClass = "best-match";
      } else if (type === "budget_alternative") {
        badgeLabel = "Tiết kiệm hơn";
        badgeClass = "badge-budget";
        cardClass = "budget-alt";
      } else if (type === "performance_alternative") {
        badgeLabel = "Hiệu năng tốt";
        badgeClass = "badge-perf";
        cardClass = "perf-alt";
      } else if (type === "nearest_alternative") {
        badgeLabel = "Phương án gần nhất";
        badgeClass = "badge-nearest";
        cardClass = "nearest-alt";
      }

      const card = document.createElement("div");
      card.className = `recommendation-card ${cardClass}`;

      // Badge header
      const badgeHeader = document.createElement("div");
      badgeHeader.className = "card-badge-header";

      const badge = document.createElement("span");
      badge.className = `card-type-badge ${badgeClass}`;
      badge.textContent = badgeLabel;
      badgeHeader.appendChild(badge);

      // Điểm Utility / Score
      const uScore = laptop.utility_score !== undefined ? laptop.utility_score : recMeta.utility_score;
      const pScore = laptop.performance_score !== undefined ? laptop.performance_score : recMeta.performance_score;
      const aiScore = laptop.ai_score || laptop.relevance_score;

      const scoreContainer = document.createElement("div");
      scoreContainer.className = "card-scores-wrapper";

      if (uScore !== undefined && uScore !== null && !isNaN(uScore)) {
        const uSpan = document.createElement("span");
        uSpan.className = "card-score utility-score";
        uSpan.textContent = `Utility: ${(Number(uScore) * 100).toFixed(0)}%`;
        scoreContainer.appendChild(uSpan);
      } else if (aiScore) {
        const aiSpan = document.createElement("span");
        aiSpan.className = "card-score ai-score";
        aiSpan.textContent = `Score: ${(Number(aiScore) * 10).toFixed(1)}/10`;
        scoreContainer.appendChild(aiSpan);
      }

      if (scoreContainer.children.length > 0) {
        badgeHeader.appendChild(scoreContainer);
      }

      card.appendChild(badgeHeader);

      // Laptop Name & Brand
      const brand = laptop.brand_name || laptop.brand || "";
      const nameEl = document.createElement("div");
      nameEl.className = "card-laptop-name";
      nameEl.textContent = laptop.laptop_name || (brand ? `${brand} Laptop` : `Laptop #${laptop.laptop_model_id || rank}`);
      card.appendChild(nameEl);

      // Price
      const rawPrice = laptop.price_vnd || laptop.price;
      if (rawPrice) {
        const priceEl = document.createElement("div");
        priceEl.className = "card-price";
        priceEl.textContent = formatPrice(rawPrice);
        card.appendChild(priceEl);
      }

      // Specs list
      const specsList = document.createElement("div");
      specsList.className = "card-specs-list";

      if (brand && (!laptop.laptop_name || !laptop.laptop_name.toLowerCase().includes(brand.toLowerCase()))) {
        specsList.appendChild(createSpecItem("Hãng", brand));
      }

      const cpu = laptop.cpu || laptop.cpu_name;
      if (cpu) {
        specsList.appendChild(createSpecItem("CPU", cpu));
      }

      const gpu = laptop.gpu || laptop.gpu_name;
      if (gpu) {
        specsList.appendChild(createSpecItem("GPU", gpu));
      }

      const screen = laptop.screen || (laptop.screen_size ? `${laptop.screen_size} inch` : null);
      if (screen) {
        specsList.appendChild(createSpecItem("Màn", screen));
      }

      const weight = laptop.laptop_weight ? `${laptop.laptop_weight} kg` : (laptop.weight ? `${laptop.weight}` : null);
      if (weight) {
        specsList.appendChild(createSpecItem("Nặng", weight));
      }

      const battery = laptop.battery_minutes ? `${Math.round(laptop.battery_minutes / 60)}h pin` : (laptop.office_battery_minutes_final ? `${Math.round(laptop.office_battery_minutes_final / 60)}h pin` : null);
      if (battery) {
        specsList.appendChild(createSpecItem("Pin", battery));
      }

      if (pScore !== undefined && pScore !== null && !isNaN(pScore) && Number(pScore) > 0) {
        specsList.appendChild(createSpecItem("Hiệu năng", `${(Number(pScore) * 100).toFixed(0)}% GB6`));
      }

      const roles = Array.isArray(laptop.suitable_roles) ? laptop.suitable_roles.join(", ") : laptop.suitable_roles;
      if (roles) {
        specsList.appendChild(createSpecItem("Hợp", roles));
      }

      card.appendChild(specsList);

      // Media actions
      const name = laptop.laptop_name || "";
      const ytUrl = laptop.review_video_url || `https://www.youtube.com/results?search_query=${encodeURIComponent("Đánh giá " + (brand ? brand + " " : "") + name)}`;
      const imgUrl = laptop.image_search_url || `https://www.google.com/search?tbm=isch&q=${encodeURIComponent((brand ? brand + " " : "") + name + " laptop")}`;

      const actionsDiv = document.createElement("div");
      actionsDiv.className = "card-actions";

      const ytLink = document.createElement("a");
      ytLink.className = "card-btn-link";
      ytLink.href = ytUrl;
      ytLink.target = "_blank";
      ytLink.rel = "noopener noreferrer";
      ytLink.textContent = "Review YouTube";
      actionsDiv.appendChild(ytLink);

      const imgLink = document.createElement("a");
      imgLink.className = "card-btn-link";
      imgLink.href = imgUrl;
      imgLink.target = "_blank";
      imgLink.rel = "noopener noreferrer";
      imgLink.textContent = "Xem Ảnh";
      actionsDiv.appendChild(imgLink);

      card.appendChild(actionsDiv);
      container.appendChild(card);
    });

    wrapper.appendChild(container);
    return wrapper;
  }

  function appendMessage(role, text, laptopDetails = null, recommendedLaptops = null, recommendations = null, isNearestFallback = false) {
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

    // Render Top-3 cards nếu có recommendedLaptops, hoặc fallback laptopDetails
    const laptopsToRender = (recommendedLaptops && recommendedLaptops.length > 0)
      ? recommendedLaptops
      : (laptopDetails ? [laptopDetails] : []);

    if (laptopsToRender.length > 0) {
      const cardsEl = renderTop3Cards(laptopsToRender, recommendations, isNearestFallback);
      if (cardsEl) {
        bubble.appendChild(cardsEl);
      }
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
    if (!toggleVoiceAutoplay || !toggleVoiceAutoplay.checked) return;
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
      // Optional feature: silent fallback
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
