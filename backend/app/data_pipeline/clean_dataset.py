"""
Xử lý laptop_dataset.csv (đã có sẵn cột giá) thành laptop_dataset_tagged.csv
sẵn sàng cho Tầng 2 (LightGBM) và Tầng 3 (Gurobi/PuLP).

Dựa trên bảng missing-value thực tế đã kiểm tra:
- Bỏ hẳn: laptop_gpu_note, laptop_cpu_note, foldable_opening_battery_result_minutes,
  review_video_url, cpu_tdp, gpu_tdp, gaming_battery_result_minutes,
  brand_is_chip_brand, brand_model_codename, charger_weight,
  geekbench_6_compute_gpu_plugged_in, geekbench_6_compute_gpu_battery
- Coalesce 4 cột Geekbench CPU (single/multi x plugged_in/battery) thành 2 cột
- Thêm has_geekbench_data đánh dấu máy nào thật sự có benchmark (không impute NaN)
- Dùng is_gaming_laptop có sẵn thay vì tự tính từ TDP (TDP thiếu quá nhiều)
- Fallback pin: dùng battery_capacity_whr khi thiếu office_battery_result_minutes

Cách dùng:
    python clean_and_tag_dataset.py \
        --input data/processed/laptop_dataset.csv \
        --output data/processed/laptop_dataset_tagged.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------- Cấu hình ngưỡng auto-tagging (chỉnh lại nếu cần) ----------
OFFICE_WEIGHT_MAX_KG = 1.4
OFFICE_BATTERY_MIN_MINUTES = 360          # 6 tiếng
PROGRAMMING_CPU_SCORE_MIN = 8000           # Geekbench 6 multi-core

# Hệ số quy đổi Wh -> phút sử dụng văn phòng ước lượng, dùng khi thiếu
# office_battery_result_minutes thật. Đây là hệ số THÔ (giả định máy văn
# phòng tiêu thụ trung bình ~5W lúc dùng nhẹ) - CẦN hiệu chỉnh lại nếu có
# dữ liệu đối chiếu thực tế, chỉ dùng làm fallback tạm thời.
WHR_TO_MINUTES_FACTOR = 12  # minutes ≈ battery_whr * 12


# Các cột chắc chắn bỏ vì tỷ lệ thiếu quá cao / không phục vụ tính toán
DROP_COLS = [
    "laptop_gpu_note",
    "laptop_cpu_note",
    "foldable_opening_battery_result_minutes",
    "review_video_url",
    "cpu_tdp",
    "gpu_tdp",
    "gaming_battery_result_minutes",
    "brand_is_chip_brand",
    "brand_model_codename",
    "charger_weight",
    "geekbench_6_compute_gpu_plugged_in",
    "geekbench_6_compute_gpu_battery",
    # metadata hệ thống, không phải feature
    "created_on", "changed_on", "created_by_fk", "changed_by_fk",
    "elton_created_at",
]

GEEKBENCH_SINGLE_COLS = [
    "geekbench_6_cpu_single_core_plugged_in",
    "geekbench_6_cpu_single_core_battery",
]
GEEKBENCH_MULTI_COLS = [
    "geekbench_6_cpu_multi_core_plugged_in",
    "geekbench_6_cpu_multi_core_battery",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"Không tìm thấy {path}. Hãy chạy merge_laptop_tables.py và "
            f"bước bổ sung giá trước, hoặc truyền đúng --input."
        )
    df = pd.read_csv(path)
    print(f"Đã đọc {len(df)} dòng, {len(df.columns)} cột từ {path}")
    return df


def report_missing(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]
    print(f"\n--- Tỷ lệ thiếu dữ liệu ({label}) ---")
    if missing.empty:
        print("Không còn cột nào thiếu dữ liệu.")
    else:
        for col, pct in missing.items():
            print(f"  {col:45s} {pct:6.2f}%")


def filter_active_visible(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    if "is_active" in df.columns:
        df = df[df["is_active"] != False]  # noqa: E712 - giữ cả True và NaN
    if "is_visible" in df.columns:
        df = df[df["is_visible"] != False]  # noqa: E712
    after = len(df)
    print(f"Lọc active/visible: {before} -> {after} dòng ({before - after} bị loại)")
    return df


def coalesce_geekbench(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp 2 biến thể (plugged_in / battery) của mỗi loại Geekbench thành 1 cột,
    ưu tiên giá trị plugged_in trước, thiếu thì lấy battery."""
    df["geekbench_cpu_single"] = df[GEEKBENCH_SINGLE_COLS[0]].combine_first(
        df[GEEKBENCH_SINGLE_COLS[1]]
    )
    df["geekbench_cpu_multi"] = df[GEEKBENCH_MULTI_COLS[0]].combine_first(
        df[GEEKBENCH_MULTI_COLS[1]]
    )
    n_single = df["geekbench_cpu_single"].notna().sum()
    n_multi = df["geekbench_cpu_multi"].notna().sum()
    print(
        f"Coalesce Geekbench: single_core có giá trị ở {n_single}/{len(df)} dòng, "
        f"multi_core có giá trị ở {n_multi}/{len(df)} dòng"
    )
    return df


def add_missing_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Đánh dấu máy có dữ liệu Geekbench thật hay không - giúp LightGBM
    phân biệt 'máy yếu' với 'máy chưa được benchmark', thay vì âm thầm
    coi NaN như 0 điểm. Cột này giữ nguyên NaN ở geekbench_cpu_single/multi
    - KHÔNG impute - để LightGBM tự xử lý missing value khi split cây."""
    df["has_geekbench_data"] = df["geekbench_cpu_multi"].notna()
    n_with_data = df["has_geekbench_data"].sum()
    print(
        f"Có dữ liệu Geekbench: {n_with_data}/{len(df)} laptop "
        f"({n_with_data/len(df)*100:.1f}%) - phần còn lại giữ NaN, "
        f"không impute, để LightGBM tự học cách xử lý khi train."
    )
    return df


def fill_battery_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Với các máy thiếu office_battery_result_minutes (đo thực tế),
    ước lượng tạm bằng battery_capacity_whr. Đánh dấu rõ nguồn để biết
    độ tin cậy khi dùng ở tầng sau."""
    df["office_battery_minutes_final"] = df["office_battery_result_minutes"]
    df["battery_source"] = "measured"

    missing_mask = df["office_battery_minutes_final"].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        estimated = df.loc[missing_mask, "battery_capacity_whr"] * WHR_TO_MINUTES_FACTOR
        df.loc[missing_mask, "office_battery_minutes_final"] = estimated
        df.loc[missing_mask, "battery_source"] = "estimated_from_whr"

    still_missing = df["office_battery_minutes_final"].isna().sum()
    print(
        f"Battery fallback: {n_missing} dòng dùng giá trị ước lượng từ Wh, "
        f"còn {still_missing} dòng vẫn thiếu (thiếu cả battery_capacity_whr)"
    )
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        # Định danh
        "laptop_model_id", "laptop_name", "brand_name", "cpu_name", "gpu_name",
        "year_introduce",
        # Ràng buộc cứng cho Gurobi/PuLP
        "price", "laptop_weight", "battery_capacity_whr",
        "office_battery_minutes_final", "battery_source",
        "screen_size", "screen_ppi",
        "screen_dimension_width", "screen_dimension_height",
        # Nhãn có sẵn, dùng trực tiếp thay vì công thức TDP
        "is_gaming_laptop", "is_workstation", "is_mobile_device",
        # Hiệu năng đã coalesce + cờ đánh dấu độ tin cậy
        "geekbench_cpu_single", "geekbench_cpu_multi", "has_geekbench_data",
    ]
    available = [c for c in keep_cols if c in df.columns]
    missing_expected = set(keep_cols) - set(available)
    if missing_expected:
        print(f"CẢNH BÁO: các cột kỳ vọng nhưng không có trong dataset: {missing_expected}")
    return df[available].copy()


def apply_auto_tagging(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-tagging phiên bản đã sửa theo dữ liệu thật -
    KHÔNG dùng công thức TDP cũ vì cpu_tdp/gpu_tdp thiếu >75%."""

    df["is_gaming_friendly"] = df["is_gaming_laptop"].fillna(False).astype(bool)

    df["is_office_friendly"] = (
        (df["laptop_weight"].fillna(999) <= OFFICE_WEIGHT_MAX_KG)
        & (df["office_battery_minutes_final"].fillna(0) >= OFFICE_BATTERY_MIN_MINUTES)
    )

    df["is_programming_friendly"] = (
        df["geekbench_cpu_multi"].fillna(0) >= PROGRAMMING_CPU_SCORE_MIN
    )

    # is_graphic_friendly: dùng is_workstation có sẵn làm proxy (thay vì
    # công thức GPU rời + RAM + PPI cũ, vì thiếu quá nhiều cột GPU chi tiết)
    if "is_workstation" in df.columns:
        df["is_graphic_friendly"] = df["is_workstation"].fillna(False).astype(bool)
    else:
        df["is_graphic_friendly"] = False

    for tag in ["is_gaming_friendly", "is_office_friendly",
                "is_programming_friendly", "is_graphic_friendly"]:
        n_true = df[tag].sum()
        print(f"  {tag:28s}: {n_true}/{len(df)} laptop ({n_true/len(df)*100:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/laptop_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/laptop_dataset_tagged.csv"))
    args = parser.parse_args()

    df = load_data(args.input)
    report_missing(df, "trước xử lý")

    df = filter_active_visible(df)
    df = coalesce_geekbench(df)
    df = add_missing_indicator(df)
    df = fill_battery_fallback(df)
    df = select_final_columns(df)

    print("\n--- Auto-tagging ---")
    df = apply_auto_tagging(df)

    report_missing(df, "sau xử lý, trước khi lưu")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {args.output} ({len(df)} dòng, {len(df.columns)} cột)")

    # Cảnh báo cuối cùng nếu vẫn còn thiếu giá - chặn ràng buộc Gurobi
    if "price" in df.columns:
        no_price = df["price"].isna().sum()
        if no_price > 0:
            print(
                f"\nCẢNH BÁO: {no_price} laptop vẫn chưa có giá. Các dòng này "
                f"sẽ không tham gia được vào ràng buộc ngân sách của Gurobi/PuLP "
                f"trừ khi bổ sung thêm giá ước lượng (estimated_price)."
            )


if __name__ == "__main__":
    main()