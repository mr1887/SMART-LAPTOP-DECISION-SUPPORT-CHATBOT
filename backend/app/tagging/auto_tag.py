"""
Auto-tagging module - Gán nhãn nhu cầu cho từng laptop dựa trên thông số kỹ thuật.

Các nhãn (boolean columns) được tính toán từ thông số phần cứng:
    is_gaming_friendly      - phù hợp chơi game
    is_office_friendly      - phù hợp văn phòng/học tập
    is_programming_friendly - phù hợp lập trình
    is_graphic_friendly     - phù hợp đồ họa/thiết kế

Kết quả được dùng trong Tầng 3 (Optimizer) để lọc cứng theo nhu cầu.
"""

import pandas as pd

# ---------- Ngưỡng tự động tag (có thể điều chỉnh) ----------
GAMING_CPU_MIN       = 8_000    # Geekbench 6 multi-core
GAMING_IS_GAMING     = True     # dùng flag is_gaming_laptop từ dataset gốc

OFFICE_WEIGHT_MAX    = 2.0      # kg
OFFICE_BATTERY_MIN   = 360      # phút (6 tiếng)

PROGRAMMING_CPU_MIN  = 6_000    # Geekbench 6 multi-core

GRAPHIC_CPU_MIN      = 8_000    # Geekbench 6 multi-core


def tag_laptops(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm/cập nhật các cột nhãn nhu cầu vào DataFrame.

    Args:
        df: DataFrame laptop đã có các cột kỹ thuật
            (is_gaming_laptop, geekbench_cpu_multi, laptop_weight,
             office_battery_minutes_final / office_battery_result_minutes,
             is_workstation)

    Returns:
        DataFrame gốc với các cột nhãn được thêm/cập nhật.
    """
    df = df.copy()

    battery_col = (
        "office_battery_minutes_final"
        if "office_battery_minutes_final" in df.columns
        else "office_battery_result_minutes"
    )

    # --- Gaming ---
    if "is_gaming_laptop" in df.columns:
        # Ưu tiên dùng flag gốc nếu có
        df["is_gaming_friendly"] = df["is_gaming_laptop"].fillna(False).astype(bool)
    elif "geekbench_cpu_multi" in df.columns:
        df["is_gaming_friendly"] = df["geekbench_cpu_multi"] >= GAMING_CPU_MIN
    else:
        df["is_gaming_friendly"] = False

    # --- Office / học tập ---
    office_cond = pd.Series(True, index=df.index)
    if "laptop_weight" in df.columns:
        office_cond &= df["laptop_weight"].fillna(999) <= OFFICE_WEIGHT_MAX
    if battery_col in df.columns:
        office_cond &= df[battery_col].fillna(0) >= OFFICE_BATTERY_MIN
    df["is_office_friendly"] = office_cond

    # --- Lập trình ---
    if "geekbench_cpu_multi" in df.columns:
        df["is_programming_friendly"] = df["geekbench_cpu_multi"].fillna(0) >= PROGRAMMING_CPU_MIN
    else:
        # Nếu thiếu benchmark, giả định không phù hợp lập trình nặng
        df["is_programming_friendly"] = False

    # --- Đồ họa / Thiết kế ---
    graphic_cond = pd.Series(False, index=df.index)
    if "geekbench_cpu_multi" in df.columns:
        graphic_cond |= df["geekbench_cpu_multi"].fillna(0) >= GRAPHIC_CPU_MIN
    if "is_workstation" in df.columns:
        graphic_cond |= df["is_workstation"].fillna(False).astype(bool)
    df["is_graphic_friendly"] = graphic_cond

    return df
