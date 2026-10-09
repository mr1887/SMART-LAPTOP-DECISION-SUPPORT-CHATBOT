"""
Auto-tagging module - gán nhãn nhu cầu từ thông số laptop.
Các tag luôn được chuẩn hóa về boolean thật để retrieval/scoring dùng nhất quán.
"""

from typing import Any
import pandas as pd

GAMING_CPU_MIN = 8_000
OFFICE_WEIGHT_MAX = 2.0
OFFICE_BATTERY_MIN = 360
PROGRAMMING_CPU_MIN = 6_000
GRAPHIC_CPU_MIN = 8_000


def _safe_bool_series(series: pd.Series) -> pd.Series:
    """Chuẩn hóa bool/0/1/"True"/"False" thành boolean thật."""
    true_values = {"true", "1", "yes", "y", "t", "có", "co"}
    false_values = {"false", "0", "no", "n", "f", "không", "khong", "", "none", "nan"}

    def _to_bool(v: Any) -> bool:
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if s in true_values:
            return True
        if s in false_values:
            return False
        return False

    return series.map(_to_bool).astype(bool)


def tag_laptops(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo lại 4 use-case tags một cách deterministic và nhất quán."""
    df = df.copy()
    idx = df.index

    cpu_multi = (
        pd.to_numeric(df["geekbench_cpu_multi"], errors="coerce").fillna(0)
        if "geekbench_cpu_multi" in df.columns
        else pd.Series(0, index=idx, dtype=float)
    )

    # Gaming: ưu tiên flag gốc; nếu không có mới fallback CPU threshold.
    if "is_gaming_laptop" in df.columns:
        gaming = _safe_bool_series(df["is_gaming_laptop"])
    else:
        gaming = cpu_multi >= GAMING_CPU_MIN
    df["is_gaming_friendly"] = gaming.astype(bool)

    # Office/học tập: không coi gaming laptop là office-friendly.
    # Sau đó áp dụng điều kiện mỏng/nhẹ và pin nếu dữ liệu tương ứng tồn tại.
    office = ~gaming
    if "laptop_weight" in df.columns:
        weight = pd.to_numeric(df["laptop_weight"], errors="coerce")
        office &= weight.notna() & (weight <= OFFICE_WEIGHT_MAX)

    battery_col = None
    if "office_battery_minutes_final" in df.columns:
        battery_col = "office_battery_minutes_final"
    elif "office_battery_result_minutes" in df.columns:
        battery_col = "office_battery_result_minutes"

    if battery_col:
        battery = pd.to_numeric(df[battery_col], errors="coerce")
        # Chỉ bắt buộc >= 6h khi có dữ liệu pin; missing không tự biến gaming thành office.
        office &= battery.notna() & (battery >= OFFICE_BATTERY_MIN)

    df["is_office_friendly"] = office.astype(bool)

    # Programming: CPU đủ mạnh; không loại gaming vì gaming laptop vẫn có thể code tốt.
    df["is_programming_friendly"] = (cpu_multi >= PROGRAMMING_CPU_MIN).astype(bool)

    # Graphic: CPU mạnh hoặc workstation.
    graphic = cpu_multi >= GRAPHIC_CPU_MIN
    if "is_workstation" in df.columns:
        graphic |= _safe_bool_series(df["is_workstation"])
    df["is_graphic_friendly"] = graphic.astype(bool)

    return df
