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

# Đơn vị tiền tệ → hệ số nhân về VNĐ
_CURRENCY_UNITS: dict[str, float] = {
    "triệu": 1_000_000,
    "tr":    1_000_000,
    "m":     1_000_000,  # phổ biến trong chat
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
    """Trích số tiền (VNĐ) từ chuỗi, hỗ trợ đơn vị triệu/nghìn/k."""
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
    if result < 1_000 and "triệu" not in text and "tr" not in text:
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
        "tối đa 20tr"                → max_price = 20M
        "trên 15 triệu"              → min_price = 15M
        "tối thiểu 10 triệu"         → min_price = 10M
        "từ 15 đến 25 triệu"         → min_price = 15M, max_price = 25M
        "khoảng 20 triệu"            → max_price = 20M (ước lượng)
        "ngân sách 20 triệu"         → max_price = 20M
    """
    text_lower = text.lower()
    max_price = min_price = None

    # Pattern: "từ X đến Y"
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

    # Pattern max_price: "dưới/tối đa/không quá/ngân sách X"
    max_patterns = [
        r"(?:dưới|tối đa|không quá|trong khoảng|ngân sách|khoảng|budget|dưới mức)\s+([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_CURRENCY_UNITS.keys()) + r")?",
    ]
    for pat in max_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2) or ""
            max_price = val * _CURRENCY_UNITS.get(unit, 1_000_000 if val < 1000 else 1)
            break

    # Pattern min_price: "trên/từ/tối thiểu X"
    min_patterns = [
        r"(?:trên|từ|tối thiểu|ít nhất|không dưới)\s+([\d]+(?:[.,]\d+)?)\s*(" + "|".join(_CURRENCY_UNITS.keys()) + r")?",
    ]
    for pat in min_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2) or ""
            # Bỏ qua nếu "từ X triệu" mà X > max_price (đã parse ở range trước)
            min_price = val * _CURRENCY_UNITS.get(unit, 1_000_000 if val < 1000 else 1)
            break

    return max_price, min_price


def _extract_tags(text: str) -> list[str]:
    """Trích required_tags từ câu hỏi dựa trên keyword mapping."""
    text_lower = text.lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags


def parse(text: str) -> dict:
    """Hàm chính: nhận câu tiếng Việt → dict ràng buộc.

    Args:
        text: Câu hỏi/yêu cầu của người dùng.

    Returns:
        dict với các key:
            max_price     (float | None)  - ngân sách tối đa (VNĐ)
            min_price     (float | None)  - ngân sách tối thiểu (VNĐ)
            max_weight    (float | None)  - cân nặng tối đa (kg)
            min_battery   (float | None)  - thời lượng pin tối thiểu (phút)
            required_tags (list[str])     - danh sách tag nhu cầu
    """
    max_price, min_price = _extract_price_constraints(text)
    max_weight = _parse_weight(text)
    min_battery = _parse_battery(text)
    required_tags = _extract_tags(text)

    return {
        "max_price": max_price,
        "min_price": min_price,
        "max_weight": max_weight,
        "min_battery": min_battery,
        "required_tags": required_tags,
    }
