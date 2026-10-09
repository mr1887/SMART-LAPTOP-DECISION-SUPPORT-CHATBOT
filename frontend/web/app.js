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
    const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
    const recommendedLaptops = Array.isArray(data.recommended_laptops) ? data.recommended_laptops : [];
    const laptopDetails = data.laptop_details || null;
    const constraints = data.constraints || {};
    const result = data.result || {};

    // Console debug tạm thời để dễ dàng kiểm tra Top-3 recommendation
    console.log(
      `[Top-3 Debug] recommendations count: ${recommendations.length}, recommended_laptops count: ${recommendedLaptops.length}`
    );

    // Ưu tiên recommended_laptops. Nếu rỗng fallback về laptop_details
    let laptopsToRender = [];
    if (recommendedLaptops.length > 0) {
      laptopsToRender = recommendedLaptops;
    } else if (laptopDetails) {
      laptopsToRender = [laptopDetails];
    }

    // Cảnh báo phương án gần nhất chỉ khi:
    // - result.status == "RELAXED"
    // hoặc
    // - recommendation.type == "nearest_alternative"
    // KHÔNG hiển thị cảnh báo này nếu chỉ có soft violation
    const isNearestFallback = Boolean(
      result.status === "RELAXED" ||
      recommendations.some((r) => r && r.type === "nearest_alternative") ||
      (recommendedLaptops && recommendedLaptops.some((l) => l && l.type === "nearest_alternative"))
    );

    const msgElement = appendMessage(
      "assistant",
      replyText,
      null,
      laptopsToRender,
      recommendations,
      isNearestFallback
    );

    // Update active constraints badges
    renderConstraintsBadges(constraints);

    // Audio TTS handling using Google Cloud TTS (/api/tts) - CHỈ gọi khi bật toggle
    if (toggleVoiceAutoplay && toggleVoiceAutoplay.checked && replyText) {
      fetchTTSAudio(replyText, msgElement, true);
    }
  }

  function formatPrice(val) {
    if (val === undefined || val === null || val === "" || isNaN(val)) return "";
    const num = Number(val);
    if (num >= 1000000) {
      const tr = num / 1000000;
      return tr % 1 === 0 ? `${tr} triệu VNĐ` : `${tr.toFixed(1)} triệu VNĐ`;
    }
    return num.toLocaleString("vi-VN") + " VNĐ";
  }

  function getRecommendationMeta(laptop, recommendations, idx) {
    if (!recommendations || !Array.isArray(recommendations) || recommendations.length === 0) {
      return {};
    }

    // Ưu tiên match bằng: recommendation.laptop_id == laptop.laptop_model_id hoặc laptop.laptop_id
    if (laptop) {
      const laptopId =
        laptop.laptop_model_id !== undefined && laptop.laptop_model_id !== null
          ? laptop.laptop_model_id
          : laptop.laptop_id !== undefined && laptop.laptop_id !== null
          ? laptop.laptop_id
          : laptop.id;

      if (laptopId !== undefined && laptopId !== null) {
        const matched = recommendations.find((r) => {
          if (!r) return false;
          const rId =
            r.laptop_id !== undefined && r.laptop_id !== null
              ? r.laptop_id
              : r.laptop_model_id;
          return rId !== undefined && rId !== null && String(rId) === String(laptopId);
        });
        if (matched) return matched;
      }
    }

    // Chỉ fallback về index nếu không match được id
    if (idx !== undefined && idx !== null && idx >= 0 && idx < recommendations.length && recommendations[idx]) {
      return recommendations[idx];
    }
    return {};
  }

  function getRecommendationLabel(type, rank) {
    let label = "Gợi ý";
    let badgeClass = "badge-best-match";
    let cardClass = "best-match";

    switch (type) {
      case "best_match":
        label = "Phù hợp nhất";
        badgeClass = "badge-best-match";
        cardClass = "best-match";
        break;
      case "budget_alternative":
        label = "Tiết kiệm hơn";
        badgeClass = "badge-budget";
        cardClass = "budget-alt";
        break;
      case "performance_alternative":
        label = "Hiệu năng tốt";
        badgeClass = "badge-perf";
        cardClass = "perf-alt";
        break;
      case "nearest_alternative":
        label = "Phương án gần nhất";
        badgeClass = "badge-nearest";
        cardClass = "nearest-alt";
        break;
      default:
        if (rank === 1) {
          label = "Phù hợp nhất";
          badgeClass = "badge-best-match";
          cardClass = "best-match";
        } else if (rank === 2) {
          label = "Tiết kiệm hơn";
          badgeClass = "badge-budget";
          cardClass = "budget-alt";
        } else if (rank === 3) {
          label = "Hiệu năng tốt";
          badgeClass = "badge-perf";
          cardClass = "perf-alt";
        }
        break;
    }

    return { label, badgeClass, cardClass };
  }

  function createSpecItem(label, val) {
    if (val === undefined || val === null || val === "") return null;
    const item = document.createElement("div");
    item.className = "card-spec-item";

    const l = document.createElement("span");
    l.className = "card-spec-label";
    l.textContent = label + ":";

    const v = document.createElement("span");
    v.className = "card-spec-val";
    v.textContent = String(val);

    item.appendChild(l);
    item.appendChild(v);
    return item;
  }

  function buildLaptopCard(laptop, recMeta = {}, rank = 1, idx = 0) {
    const resolvedRank = laptop.rank || recMeta.rank || rank || (idx + 1);
    const resolvedType =
      laptop.type ||
      recMeta.type ||
      (resolvedRank === 1
        ? "best_match"
        : resolvedRank === 2
        ? "budget_alternative"
        : resolvedRank === 3
        ? "performance_alternative"
        : "best_match");

    const { label, badgeClass, cardClass } = getRecommendationLabel(resolvedType, resolvedRank);

    const card = document.createElement("div");
    card.className = `recommendation-card ${cardClass}`;

    // Header của card: Hiển thị Rank (#1, #2, #3) + Label loại gợi ý
    const badgeHeader = document.createElement("div");
    badgeHeader.className = "card-badge-header";

    const badgeLeft = document.createElement("div");
    badgeLeft.className = "card-badge-left";

    const rankBadge = document.createElement("span");
    rankBadge.className = "card-rank-badge";
    rankBadge.textContent = `#${resolvedRank}`;
    badgeLeft.appendChild(rankBadge);

    const typeBadge = document.createElement("span");
    typeBadge.className = `card-type-badge ${badgeClass}`;
    typeBadge.textContent = label;
    badgeLeft.appendChild(typeBadge);

    badgeHeader.appendChild(badgeLeft);

    // Điểm Utility / Performance Scores
    const scoreContainer = document.createElement("div");
    scoreContainer.className = "card-scores-wrapper";

    // 1. Utility Score (dạng: Utility: xx%)
    const uScoreRaw =
      laptop.utility_score !== undefined && laptop.utility_score !== null
        ? laptop.utility_score
        : recMeta.utility_score !== undefined && recMeta.utility_score !== null
        ? recMeta.utility_score
        : null;

    if (uScoreRaw !== null && uScoreRaw !== "" && !isNaN(uScoreRaw)) {
      const uNum = Number(uScoreRaw);
      if (!isNaN(uNum)) {
        const uPercent = uNum <= 1.0 ? Math.round(uNum * 100) : Math.round(uNum);
        if (!isNaN(uPercent)) {
          const uSpan = document.createElement("span");
          uSpan.className = "card-score utility-score";
          uSpan.textContent = `Utility: ${uPercent}%`;
          scoreContainer.appendChild(uSpan);
        }
      }
    }

    // 2. Performance Score (dạng: Performance: xx%)
    const pScoreRaw =
      laptop.performance_score !== undefined && laptop.performance_score !== null
        ? laptop.performance_score
        : recMeta.performance_score !== undefined && recMeta.performance_score !== null
        ? recMeta.performance_score
        : null;

    if (pScoreRaw !== null && pScoreRaw !== "" && !isNaN(pScoreRaw)) {
      const pNum = Number(pScoreRaw);
      if (!isNaN(pNum)) {
        const pPercent = pNum <= 1.0 ? Math.round(pNum * 100) : Math.round(pNum);
        if (!isNaN(pPercent)) {
          const pSpan = document.createElement("span");
          pSpan.className = "card-score perf-score";
          pSpan.textContent = `Performance: ${pPercent}%`;
          scoreContainer.appendChild(pSpan);
        }
      }
    }

    // Fallback hiển thị AI Score cũ nếu chưa có cả 2 điểm trên
    if (scoreContainer.children.length === 0) {
      const aiScore = laptop.ai_score || laptop.relevance_score || recMeta.relevance_score;
      if (aiScore !== undefined && aiScore !== null && !isNaN(aiScore)) {
        const aiSpan = document.createElement("span");
        aiSpan.className = "card-score ai-score";
        aiSpan.textContent = `Score: ${(Number(aiScore) * 10).toFixed(1)}/10`;
        scoreContainer.appendChild(aiSpan);
      }
    }

    if (scoreContainer.children.length > 0) {
      badgeHeader.appendChild(scoreContainer);
    }

    card.appendChild(badgeHeader);

    // Laptop Name & Brand
    const brand = laptop.brand_name || laptop.brand || "";
    const nameEl = document.createElement("div");
    nameEl.className = "card-laptop-name";
    nameEl.textContent =
      laptop.laptop_name || (brand ? `${brand} Laptop` : `Laptop #${laptop.laptop_model_id || resolvedRank}`);
    card.appendChild(nameEl);

    // Price
    // Ưu tiên đúng giá mà solver đã dùng để tối ưu/rank.
    // Không ưu tiên price_vnd thô vì dataset có thể có nhiều cột giá không đồng bộ.
    const rawPrice =
      laptop.solver_price ??
      recMeta.price ??
      laptop.price ??
      laptop.price_vnd;
    if (rawPrice !== undefined && rawPrice !== null && !isNaN(rawPrice) && Number(rawPrice) > 0) {
      const priceEl = document.createElement("div");
      priceEl.className = "card-price";
      priceEl.textContent = formatPrice(rawPrice);
      card.appendChild(priceEl);
    }

    // Specs list
    const specsList = document.createElement("div");
    specsList.className = "card-specs-list";

    if (brand && (!laptop.laptop_name || !laptop.laptop_name.toLowerCase().includes(brand.toLowerCase()))) {
      const bItem = createSpecItem("Hãng", brand);
      if (bItem) specsList.appendChild(bItem);
    }

    const cpu = laptop.cpu || laptop.cpu_name;
    if (cpu) {
      const cItem = createSpecItem("CPU", cpu);
      if (cItem) specsList.appendChild(cItem);
    }

    const gpu = laptop.gpu || laptop.gpu_name;
    if (gpu) {
      const gItem = createSpecItem("GPU", gpu);
      if (gItem) specsList.appendChild(gItem);
    }

    const screen = laptop.screen || (laptop.screen_size ? `${laptop.screen_size} inch` : null);
    if (screen) {
      const sItem = createSpecItem("Màn hình", screen);
      if (sItem) specsList.appendChild(sItem);
    }

    let weightStr = null;
    if (laptop.laptop_weight !== undefined && laptop.laptop_weight !== null) {
      weightStr = `${laptop.laptop_weight} kg`;
    } else if (laptop.weight !== undefined && laptop.weight !== null) {
      weightStr = typeof laptop.weight === "number" ? `${laptop.weight} kg` : (laptop.weight.toString().includes("kg") ? laptop.weight : `${laptop.weight} kg`);
    }
    if (weightStr) {
      const wItem = createSpecItem("Trọng lượng", weightStr);
      if (wItem) specsList.appendChild(wItem);
    }

    const batteryMins = laptop.battery_minutes ?? laptop.office_battery_minutes_final;
    if (batteryMins !== undefined && batteryMins !== null && !isNaN(batteryMins) && Number(batteryMins) > 0) {
      const hours = Math.round(Number(batteryMins) / 60);
      const batItem = createSpecItem("Pin", `${hours}h (${batteryMins} phút)`);
      if (batItem) specsList.appendChild(batItem);
    }

    const roles = Array.isArray(laptop.suitable_roles)
      ? laptop.suitable_roles.join(", ")
      : laptop.suitable_roles;
    if (roles) {
      const rItem = createSpecItem("Phù hợp", roles);
      if (rItem) specsList.appendChild(rItem);
    }

    card.appendChild(specsList);

    // Media Actions (Review video & Search Image)
    const name = laptop.laptop_name || "";
    const ytUrl =
      laptop.review_video_url ||
      `https://www.youtube.com/results?search_query=${encodeURIComponent(
        "Đánh giá " + (brand ? brand + " " : "") + name
      )}`;
    const imgUrl =
      laptop.image_search_url ||
      `https://www.google.com/search?tbm=isch&q=${encodeURIComponent(
        (brand ? brand + " " : "") + name + " laptop"
      )}`;

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

    return card;
  }

  function renderRecommendationCards(recommendedLaptops, recommendations = [], isNearestFallback = false) {
    if (!recommendedLaptops || !Array.isArray(recommendedLaptops) || recommendedLaptops.length === 0) {
      return null;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "top3-wrapper";

    // Cảnh báo nearest_alternative chỉ khi:
    // - isNearestFallback === true (result.status === "RELAXED" hoặc nearest_alternative)
    // - Hoặc có laptop/recommendation là "nearest_alternative"
    // KHÔNG cảnh báo nếu chỉ có soft violation
    const hasNearest = Boolean(
      isNearestFallback ||
      (recommendations && recommendations.some((r) => r && r.type === "nearest_alternative")) ||
      (recommendedLaptops && recommendedLaptops.some((l) => l && l.type === "nearest_alternative"))
    );

    if (hasNearest) {
      const warningEl = document.createElement("div");
      warningEl.className = "nearest-alt-warning";
      warningEl.innerHTML =
        '<span class="warning-icon">⚠️</span><span>Không có laptop thỏa hoàn toàn các yêu cầu bắt buộc. Đây là phương án gần nhất.</span>';
      wrapper.appendChild(warningEl);
    }

    const container = document.createElement("div");
    container.className = "top3-container";
    container.setAttribute("data-cards-count", String(Math.min(recommendedLaptops.length, 3)));

    recommendedLaptops.forEach((laptop, idx) => {
      if (!laptop) return;
      const recMeta = getRecommendationMeta(laptop, recommendations, idx);
      const rank = recMeta.rank || laptop.rank || (idx + 1);
      const card = buildLaptopCard(laptop, recMeta, rank, idx);
      container.appendChild(card);
    });

    wrapper.appendChild(container);
    return wrapper;
  }

  function appendMessage(
    role,
    text,
    laptopDetails = null,
    recommendedLaptops = null,
    recommendations = null,
    isNearestFallback = false
  ) {
    const msgWrapper = document.createElement("div");
    msgWrapper.className = `message-wrapper ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "BẠN" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    try {
      bubble.innerHTML = typeof marked !== "undefined" && marked.parse ? marked.parse(text) : text;
    } catch (e) {
      bubble.textContent = text;
    }

    // Ưu tiên render recommendedLaptops; fallback về laptopDetails nếu có
    const laptopsToRender =
      recommendedLaptops && recommendedLaptops.length > 0
        ? recommendedLaptops
        : laptopDetails
        ? [laptopDetails]
        : [];

    if (laptopsToRender.length > 0) {
      const cardsEl = renderRecommendationCards(laptopsToRender, recommendations, isNearestFallback);
      if (cardsEl) {
        bubble.appendChild(cardsEl);
        if (laptopsToRender.length >= 2) {
          msgWrapper.classList.add("has-top3");
        }
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
