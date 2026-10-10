"""
Tầng 1 - NL2Constraint: Chuyển đổi câu hỏi tiếng Việt tự nhiên thành
bộ ràng buộc có cấu trúc để đưa vào Tầng 3 (OR-Tools CP-SAT Optimizer).

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


def _parse_ram(text: str) -> Optional[float]:
    """Trích dung lượng RAM tối thiểu (GB)."""
    text_lower = text.lower()
    patterns = [
        # VD: "RAM 16GB", "RAM tối thiểu 32GB", "RAM ít nhất 16GB", "RAM 16"
        r"\b(?:ram|bộ\s*nhớ\s*ram)(?:\s+(?:ít\s+nhất|tối\s+thiểu|từ|trên|không\s+dưới|>=))?\s*(\d+(?:[.,]\d+)?)\s*(?:gb|g)?\b",
        # VD: "16GB RAM", "ít nhất 16GB RAM", "tối thiểu 32GB RAM"
        r"(?:(?:ít\s+nhất|tối\s+thiểu|từ|trên|không\s+dưới|>=)\s*)?(\d+(?:[.,]\d+)?)\s*(?:gb|g)?\s*ram\b",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            val = float(m.group(1).replace(",", "."))
            if 4 <= val <= 256:
                return val
    return None


def _parse_storage(text: str) -> Optional[float]:
    """Trích dung lượng lưu trữ / SSD / HDD tối thiểu và quy về GB (1TB = 1024GB)."""
    text_lower = text.lower()
    patterns = [
        # VD: "SSD 512GB", "ổ cứng 1TB", "SSD tối thiểu 512GB", "ổ cứng ít nhất 1TB"
        r"\b(?:ssd|hdd|ổ\s*cứng|ổ\s*ssd|storage|nvme|rom)(?:\s+(?:ít\s+nhất|tối\s+thiểu|từ|trên|không\s+dưới|>=))?\s*(\d+(?:[.,]\d+)?)\s*(tb|t|gb|g)?\b",
        # VD: "ít nhất 1TB SSD", "512GB SSD", "1TB ổ cứng"
        r"(?:(?:ít\s+nhất|tối\s+thiểu|từ|trên|không\s+dưới|>=)\s*)?(\d+(?:[.,]\d+)?)\s*(tb|t|gb|g)\s*(?:ssd|hdd|ổ\s*cứng|nvme|storage|rom)\b",
        # VD: "1TB", "ít nhất 1TB" (đơn vị TB mặc định là storage)
        r"\b(?:ít\s+nhất|tối\s+thiểu|từ|trên|không\s+dưới|>=)?\s*(\d+(?:[.,]\d+)?)\s*(tb|t)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = (m.group(2) or "").lower()
            if unit in ["tb", "t"]:
                return val * 1024
            elif unit in ["gb", "g"]:
                return val
            else:
                if val <= 8:
                    return val * 1024
                return val
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
            # Bỏ qua nếu phía sau là đơn vị của RAM, SSD, pin, cân nặng...
            if not unit and re.search(r"^\s*(?:gb|tb|kg|h|giờ|tiếng|ram|ssd|hdd|in|inch|cm)", text_lower[m.end():]):
                continue
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
    min_ram = _parse_ram(text)
    min_storage = _parse_storage(text)
    require_discrete_gpu, gpu_keyword = _extract_gpu_constraints(text)
    required_tags = _extract_tags(text)

    return {
        "max_price": max_price,
        "min_price": min_price,
        "max_weight": max_weight,
        "min_battery": min_battery,
        "min_ram": min_ram,
        "min_storage": min_storage,
        "require_discrete_gpu": require_discrete_gpu,
        "gpu_keyword": gpu_keyword,
        "required_tags": required_tags,
    }


def classify_context_action(
    text: str,
    parsed_requirements: Optional[dict] = None,
    current_requirements: Optional[dict] = None,
) -> str:
    """Phân loại cách áp dụng yêu cầu mới vào context hội thoại.

    Trả về một trong:
    - "ADD": bổ sung tiêu chí mới và giữ các tiêu chí cũ.
    - "UPDATE": sửa/ghi đè một hoặc vài tiêu chí hiện có, giữ phần còn lại.
    - "REPLACE": bắt đầu một yêu cầu tìm kiếm mới, không mang context cũ sang.

    Hàm dùng rule deterministic để tránh phụ thuộc thêm một LLM call.
    """
    clean = re.sub(r"\s+", " ", (text or "").strip().lower())
    parsed = parsed_requirements or {}
    current = current_requirements or {}

    constraints = parsed.get("constraints", []) if isinstance(parsed, dict) else []
    new_tags = set(parsed.get("required_tags", []) or []) if isinstance(parsed, dict) else set()
    old_tags = set(current.get("required_tags", []) or []) if isinstance(current, dict) else set()

    new_fields = {
        c.get("field")
        for c in constraints
        if isinstance(c, dict) and c.get("field")
    }

    # 1) Tín hiệu rõ ràng rằng user đang chuyển sang bài toán khác.
    replace_patterns = [
        r"\b(?:tìm|tư vấn|gợi ý|chọn)\s+(?:cho\s+)?(?:tôi\s+)?(?:một\s+)?laptop\b",
        r"\b(?:bây giờ|giờ)\s+(?:tôi\s+)?(?:muốn|cần)\b",
        r"\bchuyển\s+sang\b",
        r"\bthôi\s+(?:tìm|tư vấn|chọn)\b",
        r"\btìm\s+lại\b",
        r"\bbắt\s+đầu\s+lại\b",
    ]

    explicit_replace = any(re.search(p, clean) for p in replace_patterns)

    # Một nhu cầu sử dụng mới khác nhu cầu cũ là tín hiệu mạnh cho REPLACE,
    # đặc biệt khi câu mới cũng đưa ra ngân sách hoặc mang dạng yêu cầu hoàn chỉnh.
    tag_changed = bool(new_tags and old_tags and new_tags != old_tags)
    has_price = "price" in new_fields
    full_new_search = bool(new_tags and has_price)

    if explicit_replace and (new_tags or constraints):
        return "REPLACE"

    if tag_changed and (has_price or any(k in clean for k in ["tư vấn", "tìm", "muốn", "cần"])):
        return "REPLACE"

    if full_new_search and old_tags and new_tags != old_tags:
        return "REPLACE"

    # 2) Tín hiệu UPDATE: thay đổi giá trị của tiêu chí đã có.
    update_patterns = [
        r"\b(?:đổi|thay)\b",
        r"\b(?:nâng|tăng)\s+(?:ngân\s+sách|giá|ram|ssd|pin)?\b",
        r"\b(?:giảm|hạ)\s+(?:ngân\s+sách|giá|cân\s+nặng)?\b",
        r"\bthôi\b",
        r"\b(?:không\s+cần|bỏ)\b",
        r"\b(?:lên|xuống)\s+\d",
    ]

    if any(re.search(p, clean) for p in update_patterns):
        return "UPDATE"

    # Nếu field mới trùng field đang tồn tại thì về bản chất là cập nhật field đó.
    old_constraints = current.get("constraints", []) if isinstance(current, dict) else []
    old_fields = {
        c.get("field")
        for c in old_constraints
        if isinstance(c, dict) and c.get("field")
    }
    if new_fields & old_fields:
        return "UPDATE"

    # 3) ADD: các câu bổ sung / follow-up.
    add_patterns = [
        r"\b(?:thêm|ngoài\s+ra|cũng\s+cần|cũng\s+muốn|ưu\s+tiên\s+thêm)\b",
        r"\b(?:và|với)\s+(?:ram|ssd|pin|cân\s+nặng|gpu|card)\b",
        r"\b(?:ưu\s+tiên|cần)\s+(?:nhẹ|pin|ram|ssd|mỏng)\b",
    ]

    if any(re.search(p, clean) for p in add_patterns):
        return "ADD"

    # Mặc định: nếu đã có context và câu mới chỉ đưa thêm field/tag chưa có,
    # coi là bổ sung; nếu chưa có context thì REPLACE tương đương khởi tạo mới.
    has_current = bool(
        (current.get("constraints") if isinstance(current, dict) else None)
        or (current.get("preferences") if isinstance(current, dict) else None)
        or (current.get("required_tags") if isinstance(current, dict) else None)
    )

    if not has_current:
        return "REPLACE"

    return "ADD"


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


def convert_legacy_regex_to_requirement_set(legacy_data: dict, text: str = "") -> dict:
    """Chuyển đổi kết quả trích xuất từ regex cũ sang schema RequirementSet mới."""
    from app.nlp.validator import validate_requirement_set

    text_lower = text.lower() if text else ""
    constraints = []

    # 1. Price constraints
    max_price = legacy_data.get("max_price")
    if max_price is not None:
        if any(kw in text_lower for kw in ["tầm", "khoảng", "ngân sách", "budget", "xung quanh"]):
            price_type = "soft"
        else:
            price_type = "hard"

        constraints.append({
            "field": "price",
            "operator": "<=",
            "value": max_price,
            "type": price_type,
            "source_text": text or None,
        })

    min_price = legacy_data.get("min_price")
    if min_price is not None:
        constraints.append({
            "field": "price",
            "operator": ">=",
            "value": min_price,
            "type": "hard",
            "source_text": text or None,
        })

    # 2. Weight constraint (explicit threshold -> hard mặc định)
    max_weight = legacy_data.get("max_weight")
    if max_weight is not None:
        constraints.append({
            "field": "weight_kg",
            "operator": "<=",
            "value": max_weight,
            "type": "hard",
            "source_text": text or None,
        })

    # 3. Battery constraint (explicit minimum -> hard)
    min_battery = legacy_data.get("min_battery")
    if min_battery is not None:
        constraints.append({
            "field": "battery_minutes",
            "operator": ">=",
            "value": min_battery,
            "type": "hard",
            "source_text": text or None,
        })

    # 4. RAM constraint (explicit minimum -> hard)
    min_ram = legacy_data.get("min_ram")
    if min_ram is not None:
        constraints.append({
            "field": "ram_gb",
            "operator": ">=",
            "value": min_ram,
            "type": "hard",
            "source_text": text or None,
        })

    # 5. Storage constraint (explicit minimum -> hard)
    min_storage = legacy_data.get("min_storage")
    if min_storage is not None:
        constraints.append({
            "field": "storage_gb",
            "operator": ">=",
            "value": min_storage,
            "type": "hard",
            "source_text": text or None,
        })

    # 6. GPU constraints
    req_discrete = legacy_data.get("require_discrete_gpu")
    if req_discrete is not None:
        constraints.append({
            "field": "gpu_discrete",
            "operator": "=",
            "value": bool(req_discrete),
            "type": "hard",
            "source_text": text or None,
        })

    gpu_kw = legacy_data.get("gpu_keyword")
    if gpu_kw:
        constraints.append({
            "field": "gpu_keyword",
            "operator": "=",
            "value": str(gpu_kw).strip(),
            "type": "hard",
            "source_text": text or None,
        })

    preferences = []

    # Sở thích ngôn ngữ tự nhiên rõ ràng: dùng làm tie-break/query-match,
    # không biến thành hard constraint.
    if text_lower:
        if any(kw in text_lower for kw in ["giá rẻ", "rẻ hơn", "rẻ nhất", "tiết kiệm"]):
            preferences.append({
                "field": "price",
                "direction": "minimize",
                "source_text": text or None,
            })
        if any(kw in text_lower for kw in ["mỏng nhẹ", "gọn nhẹ", "nhẹ", "dễ mang"]):
            preferences.append({
                "field": "weight_kg",
                "direction": "minimize",
                "source_text": text or None,
            })
        if any(kw in text_lower for kw in ["pin trâu", "pin lâu", "pin tốt", "thời lượng pin"]):
            preferences.append({
                "field": "battery_minutes",
                "direction": "maximize",
                "source_text": text or None,
            })

    raw_output = {
        "constraints": constraints,
        "preferences": preferences,
        "required_tags": legacy_data.get("required_tags", []),
    }

    validated = validate_requirement_set(raw_output)
    return validated.model_dump()


def parse(text: str, use_gemini: bool = True) -> dict:
    """Nhận câu tiếng Việt -> RequirementSet.

    Gemini được dùng cho semantic extraction, nhưng các tín hiệu deterministic
    rõ ràng (giá, GPU, RAM/SSD, tag nhu cầu) từ regex luôn được dùng để bổ sung.
    Điều này tránh trường hợp Gemini bỏ sót "văn phòng", "lập trình", "gaming"
    khiến nhiều truy vấn khác nhau cùng rơi vào một candidate set.
    """
    legacy_res = parse_regex(text)
    regex_req = convert_legacy_regex_to_requirement_set(legacy_res, text=text)

    if use_gemini:
        try:
            from app.ai.gemini_service import extract_constraints_gemini
            from app.nlp.validator import validate_requirement_set

            gemini_result = extract_constraints_gemini(text)
            if gemini_result and isinstance(gemini_result, dict):
                validated = validate_requirement_set(gemini_result).model_dump()

                # 1) Required tags: deterministic keyword tags rất đáng tin cậy.
                # Nếu regex nhận ra "văn phòng", "lập trình", "gaming", "đồ họa"
                # thì bắt buộc giữ tag đó, kể cả Gemini bỏ sót.
                merged_tags = list(dict.fromkeys(
                    list(validated.get("required_tags", []) or [])
                    + list(regex_req.get("required_tags", []) or [])
                ))
                validated["required_tags"] = merged_tags

                # 2) Constraints: Gemini giữ ưu tiên; regex chỉ bổ sung field bị thiếu.
                gemini_fields = {
                    c.get("field")
                    for c in validated.get("constraints", [])
                    if isinstance(c, dict) and c.get("field")
                }
                merged_constraints = list(validated.get("constraints", []) or [])
                for c in regex_req.get("constraints", []) or []:
                    if isinstance(c, dict) and c.get("field") not in gemini_fields:
                        merged_constraints.append(c)

                validated["constraints"] = merged_constraints

                # 3) Preferences deterministic từ câu chữ ("giá rẻ", "mỏng nhẹ",
                # "pin trâu") được bổ sung nếu Gemini chưa trả field tương ứng.
                gemini_pref_fields = {
                    p.get("field")
                    for p in validated.get("preferences", [])
                    if isinstance(p, dict) and p.get("field")
                }
                merged_preferences = list(validated.get("preferences", []) or [])
                for p in regex_req.get("preferences", []) or []:
                    if isinstance(p, dict) and p.get("field") not in gemini_pref_fields:
                        merged_preferences.append(p)
                validated["preferences"] = merged_preferences

                return validate_requirement_set(validated).model_dump()
        except Exception:
            pass

    return regex_req



