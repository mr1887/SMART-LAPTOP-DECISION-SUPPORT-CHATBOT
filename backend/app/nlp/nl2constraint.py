"""
Tầng 1 - NL2Constraint: Chuyển đổi câu hỏi tiếng Việt tự nhiên thành
bộ ràng buộc có cấu trúc để đưa vào Tầng 3 (Optimizer).

Dùng Regex + từ điển keyword để trích xuất:
    - max_price / min_price (ngân sách)
    - max_weight           (cân nặng)
    - min_battery          (thời lượng pin, quy về phút)
    - required_tags        (nhu cầu: gaming, lập trình, đồ họa, v.v.)
"""

import re
from typing import Optional


# ---------- Từ điển tag theo nhu cầu ----------
TAG_KEYWORDS: dict[str, list[str]] = {
    "is_gaming_friendly": [
        "game", "gaming", "chơi game", "game thủ", "esport", "fps",
    ],
    "is_programming_friendly": [
        "lập trình", "code", "coding", "developer", "dev", "python",
        "java", "react", "công nghệ thông tin", "cntt", "it",
    ],
    "is_graphic_friendly": [
        "đồ họa", "thiết kế", "design", "photoshop", "illustrator",
        "video", "render", "3d", "autocad", "kiến trúc",
    ],
    "is_office_friendly": [
        "văn phòng", "office", "word", "excel", "kế toán",
        "nhẹ nhàng", "học tập", "sinh viên", "học sinh",
    ],
}

# Các mẫu câu chào hỏi phổ biến
GREETING_PATTERNS = [
    r"^(?:xin\s+)?chào(?:\s+(?:bạn|shop|em|anh|chị|ad|bot|cậu|mọi người))?[\s!.]*$",
    r"^(?:hello|hi|hey|alo|alô|yo|hola|bonjour)[\s!.]*$",
    r"^(?:chào|hi|hello)\s+(?:buổi\s+)?(?:sáng|trưa|chiều|tối)[\s!.]*$",
    r"^(?:bạn\s+ơi|shop\s+ơi|ad\s+ơi|bot\s+ơi)[\s!.]*$",
]

# Các mẫu câu yêu cầu reset / xóa bộ lọc
RESET_PATTERNS = [
    r"\b(?:reset|làm\s+mới|xóa\s+bộ\s+lọc|xóa\s+ràng\s+buộc|bắt\s+đầu\s+lại|tìm\s+lại\s+từ\s+đầu|hủy\s+bộ\s+lọc|xóa\s+hết)\b",
]

# Các từ khóa thể hiện ý định tìm kiếm máy tính
SEARCH_INTENT_KEYWORDS = [
    "tìm", "mua", "tư vấn", "gợi ý", "chọn", "laptop", "máy tính", "máy",
    "cấu hình", "ngân sách", "triệu", "tr", "củ", "gaming", "văn phòng",
    "học tập", "đồ họa", "lập trình", "pin", "card", "gpu", "cpu", "rtx",
]


# Đơn vị tiền tệ → hệ số nhân về VNĐ
_CURRENCY_UNITS: dict[str, float] = {
    "triệu": 1_000_000,
    "tr":    1_000_000,
    "m":     1_000_000,  # phổ biến trong chat
    "củ":    1_000_000,  # tiếng lóng phổ biến
    "nghìn": 1_000,
    "k":     1_000,
    "đồng":  1,
    "vnđ":   1,
    "đ":     1,
}

# Đơn vị pin → hệ số nhân về phút
_BATTERY_UNITS: dict[str, float] = {
    "giờ":   60,
    "tiếng": 60,
    "h":     60,
    "phút":  1,
}


def _parse_money(text: str) -> Optional[float]:
    """Trích số tiền (VNĐ) từ chuỗi, hỗ trợ đơn vị triệu/nghìn/k/củ."""
    text = text.lower().strip()
    pattern = r"([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_CURRENCY_UNITS.keys()) + r")?"
    m = re.search(pattern, text)
    if not m:
        return None
    number = float(m.group(1).replace(",", "."))
    unit = m.group(2) or ""
    multiplier = _CURRENCY_UNITS.get(unit, 1)
    result = number * multiplier
    # Nếu số quá nhỏ (< 1000), có thể người dùng đang nói "15" ý là "15 triệu"
    if result < 1_000 and "triệu" not in text and "tr" not in text and "củ" not in text:
        result *= 1_000_000
    return result


def _parse_weight(text: str) -> Optional[float]:
    """Trích cân nặng (kg) từ chuỗi."""
    text = text.lower()
    m = re.search(r"([\d]+(?:[.,]\d+)?)\s*kg", text)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _parse_battery(text: str) -> Optional[float]:
    """Trích thời lượng pin và quy về phút."""
    text = text.lower()
    pattern = r"([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_BATTERY_UNITS.keys()) + r")"
    m = re.search(pattern, text)
    if m:
        number = float(m.group(1).replace(",", "."))
        unit = m.group(2)
        return number * _BATTERY_UNITS.get(unit, 1)
    return None


def _extract_price_constraints(text: str) -> tuple[Optional[float], Optional[float]]:
    """Trích max_price và min_price từ câu hỏi.

    Các pattern hỗ trợ:
        "dưới 20 triệu"              → max_price = 20M
        "máy 10 triệu"               → max_price = 10M
        "tối đa 20tr"                → max_price = 20M
        "trên 15 triệu"              → min_price = 15M
        "tối thiểu 10 triệu"         → min_price = 10M
        "từ 15 đến 25 triệu"         → min_price = 15M, max_price = 25M
        "khoảng 20 triệu"            → max_price = 20M (ước lượng)
        "ngân sách 20 triệu"         → max_price = 20M
    """
    text_lower = text.lower()
    max_price = min_price = None

    # Pattern 1: "từ X đến Y"
    range_match = re.search(
        r"từ\s+([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_CURRENCY_UNITS.keys()) + r")?"
        r"\s+(?:đến|tới|~|-)\s+([\d]+(?:[.,]\d+)?)\s*("
        + "|".join(_CURRENCY_UNITS.keys()) + r")?",
        text_lower,
    )
    if range_match:
        lo_val = float(range_match.group(1).replace(",", "."))
        lo_unit = range_match.group(2) or ""
        hi_val = float(range_match.group(3).replace(",", "."))
        hi_unit = range_match.group(4) or ""
        min_price = lo_val * _CURRENCY_UNITS.get(lo_unit, 1_000_000 if lo_val < 1000 else 1)
        max_price = hi_val * _CURRENCY_UNITS.get(hi_unit, 1_000_000 if hi_val < 1000 else 1)
        return max_price, min_price

    # Pattern min_price: "trên/từ/tối thiểu X"
    min_patterns = [
        r"(?:trên|từ|tối thiểu|ít nhất|không dưới)\s+([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_CURRENCY_UNITS.keys()) + r")?",
    ]
    for pat in min_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2) or ""
            min_price = val * _CURRENCY_UNITS.get(unit, 1_000_000 if val < 1000 else 1)
            break

    # Pattern max_price: "dưới/tối đa/không quá/ngân sách/máy X triệu/X triệu"
    max_patterns = [
        r"(?:dưới|tối đa|không quá|trong khoảng|ngân sách|khoảng|budget|dưới mức|máy|laptop|tầm|giá)\s+([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_CURRENCY_UNITS.keys()) + r")?",
        r"\b([\d]+(?:[.,]\d+)?)\s*(triệu|tr|m|củ)\b",
    ]
    for pat in max_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2) or ""
            parsed_max = val * _CURRENCY_UNITS.get(unit, 1_000_000 if val < 1000 else 1)
            # Tránh trường hợp "10" là tên GPU (vd 4060) hay số khác không phải tiền tệ
            if parsed_max >= 3_000_000:
                max_price = parsed_max
                break

    return max_price, min_price


def _extract_gpu_constraints(text: str) -> tuple[Optional[bool], Optional[str]]:
    """Trích xuất ràng buộc về GPU từ câu hỏi:
    - require_discrete_gpu: True (yêu cầu card rời), False (chỉ cần card tích hợp/onboard), None (không yêu cầu)
    - gpu_keyword: Dòng GPU cụ thể (RTX 4060, RTX 4050, NVIDIA, Intel Arc, Apple M4, v.v.)
    """
    text_lower = text.lower()
    require_discrete: Optional[bool] = None
    gpu_keyword: Optional[str] = None

    # 1. Phát hiện yêu cầu card onboard / tích hợp / không cần card rời
    integrated_patterns = [
        r"\b(?:card\s+)?onboard\b",
        r"\b(?:card\s+|gpu\s+)?tích\s+hợp\b",
        r"\bkhông\s+cần\s+(?:card|gpu|vga)(?:\s+đồ\s+họa)?(?:\s+rời)?\b",
        r"\bkhông\s+(?:dùng|cần)\s+(?:card|gpu|vga)\s+rời\b",
        r"\bkhông\s+card\s+rời\b",
    ]
    for pat in integrated_patterns:
        if re.search(pat, text_lower):
            require_discrete = False
            break

    # 2. Phát hiện yêu cầu card rời (nếu chưa bị gán False ở trên)
    if require_discrete is None:
        discrete_patterns = [
            r"\b(?:card|gpu|vga)(?:\s+đồ\s+họa)?\s+rời\b",
            r"\bcó\s+(?:card|gpu|vga)\s+rời\b",
            r"\bcần\s+(?:card|gpu|vga)\s+rời\b",
            r"\bdiscrete\s*gpu\b",
            r"\bdgpu\b",
            r"\bcard\s+nvidia\b",
            r"\bcard\s+geforce\b",
            r"\bcard\s+rtx\b",
            r"\bcard\s+gtx\b",
        ]
        for pat in discrete_patterns:
            if re.search(pat, text_lower):
                require_discrete = True
                break

    # 3. Trích xuất model / từ khóa GPU cụ thể
    # VD: "RTX 4060", "4060", "RTX 4050", "4050", "3050", "4070", "5060", "1650", "NVIDIA RTX 4060", "Intel Arc", "Apple M4"
    gpu_specific_patterns = [
        r"\b(rtx\s*\d{4}(?:\s*ti|\s*mobile|\s*super)?)\b",
        r"\b(gtx\s*\d{4}(?:\s*ti)?)\b",
        r"\b(nvidia\s+rtx\s*\d{4}(?:\s*ti|\s*mobile|\s*super)?)\b",
        r"\b(nvidia\s+geforce(?:\s+rtx\s*\d{4})?)\b",
        r"\b(nvidia\s+rtx(?:\s+pro|\s+ada)?)\b",
        r"\b(nvidia)\b",
        r"\b(amd\s+radeon(?:\s+\d{3,4}[a-z]*(?:\s*\(tích\s+hợp\))?)?)\b",
        r"\b(radeon\s+\d{3,4}[a-z]*)\b",
        r"\b(intel\s+arc(?:\s+b?\d{3}[a-z]*)?)\b",
        r"\b(apple\s+m\d(?:\s*(?:pro|max|ultra))?)\b",
        # Bổ sung bắt các số mã GPU đứng độc lập khi đi cùng ngữ cảnh card/gpu/vga/rời (vd: "card rời 4060", "card 4060", "4060")
        r"\b(?:card|gpu|vga|rtx|gtx|rời)?\s*(4050|4060|4070|4080|4090|3050|3060|3070|3080|5060|5070|1650|1660)\b",
    ]

    for pat in gpu_specific_patterns:
        m = re.search(pat, text_lower)
        if m:
            raw_match = m.group(1)
            gpu_keyword = re.sub(r"\s+", " ", raw_match).strip().upper()
            if gpu_keyword in ["4050", "4060", "4070", "4080", "4090", "3050", "3060", "3070", "3080", "5060", "5070"]:
                gpu_keyword = f"RTX {gpu_keyword}"
            elif gpu_keyword in ["1650", "1660"]:
                gpu_keyword = f"GTX {gpu_keyword}"

            if "APPLE" in gpu_keyword:
                gpu_keyword = gpu_keyword.title()
            elif "INTEL" in gpu_keyword:
                gpu_keyword = gpu_keyword.replace("INTEL", "Intel").replace("ARC", "Arc")
            elif "AMD" in gpu_keyword or "RADEON" in gpu_keyword:
                gpu_keyword = gpu_keyword.replace("AMD", "AMD").replace("RADEON", "Radeon")
            elif "NVIDIA" in gpu_keyword:
                gpu_keyword = gpu_keyword.replace("NVIDIA", "NVIDIA")

            # Nếu là dòng card rời (RTX / GTX / NVIDIA) thì mặc định suy ra có card rời
            if any(k in gpu_keyword.lower() for k in ["rtx", "gtx", "nvidia", "4060", "4050", "3050"]):
                if require_discrete is None:
                    require_discrete = True
            break

    return require_discrete, gpu_keyword


def _extract_tags(text: str) -> list[str]:
    """Trích required_tags từ câu hỏi dựa trên keyword mapping."""
    text_lower = text.lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags


def parse_regex(text: str) -> dict:
    """Hàm phân tích bằng Regex truyền thống (logic gốc)."""
    max_price, min_price = _extract_price_constraints(text)
    max_weight = _parse_weight(text)
    min_battery = _parse_battery(text)
    require_discrete_gpu, gpu_keyword = _extract_gpu_constraints(text)
    required_tags = _extract_tags(text)

    return {
        "max_price": max_price,
        "min_price": min_price,
        "max_weight": max_weight,
        "min_battery": min_battery,
        "require_discrete_gpu": require_discrete_gpu,
        "gpu_keyword": gpu_keyword,
        "required_tags": required_tags,
    }


def detect_intent(text: str, has_extracted_constraints: bool = False) -> str:
    """Xác định ý định người dùng từ câu chat:
    - "GREETING": Chào hỏi ("hello", "chào bạn", "alo"...)
    - "RESET": Muốn xóa bộ lọc, tìm lại từ đầu
    - "SEARCH": Có tiêu chí lọc hoặc thể hiện rõ muốn tìm laptop
    - "SMALLTALK": Trò chuyện chung, hỏi han, không có tiêu chí laptop cụ thể
    """
    clean_text = text.strip().lower()

    if not clean_text:
        return "GREETING"

    # 1. Kiểm tra Reset
    if any(re.search(pat, clean_text) for pat in RESET_PATTERNS):
        return "RESET"

    # 2. Kiểm tra Chào hỏi thuần túy
    if any(re.search(pat, clean_text) for pat in GREETING_PATTERNS):
        # Nếu có trích xuất được ràng buộc cụ thể (vd: "chào bạn mình cần tìm máy 20tr") thì là SEARCH
        if has_extracted_constraints:
            return "SEARCH"
        return "GREETING"

    # 3. Nếu có trích xuất được ràng buộc cụ thể (giá, cân nặng, pin, GPU, tag) -> SEARCH
    if has_extracted_constraints:
        return "SEARCH"

    # 4. Kiểm tra có từ khóa tìm kiếm laptop rõ ràng không
    if any(kw in clean_text for kw in SEARCH_INTENT_KEYWORDS):
        return "SEARCH"

    return "SMALLTALK"


def parse(text: str, use_gemini: bool = True) -> dict:
    """Hàm chính: nhận câu tiếng Việt → dict ràng buộc.
    Ưu tiên dùng Google Gemini (nếu có API Key), tự động fallback về Regex nếu lỗi hoặc không có Key.

    Args:
        text: Câu hỏi/yêu cầu của người dùng.
        use_gemini: Cho phép dùng Gemini hay không (mặc định True).

    Returns:
        dict với các key:
            max_price            (float | None)  - ngân sách tối đa (VNĐ)
            min_price            (float | None)  - ngân sách tối thiểu (VNĐ)
            max_weight           (float | None)  - cân nặng tối đa (kg)
            min_battery          (float | None)  - thời lượng pin tối thiểu (phút)
            require_discrete_gpu (bool | None)   - bắt buộc có card rời (True/False/None)
            gpu_keyword          (str | None)    - từ khóa / model GPU cụ thể
            required_tags        (list[str])     - danh sách tag nhu cầu
    """
    if use_gemini:
        try:
            from app.ai.gemini_service import extract_constraints_gemini
            gemini_result = extract_constraints_gemini(text)
            if gemini_result is not None:
                return gemini_result
        except Exception:
            pass

    return parse_regex(text)


