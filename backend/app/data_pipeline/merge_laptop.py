import pandas as pd 
from pathlib import Path 
import os
import datetime

RAW_DIR = Path("data/raw") # thư mục chứa file CSV
OUT_DIR = Path("data/processed") # thư mục chứa file sau khi xử lý
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv" # đường dẫn đến file CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy export bảng '{name}' từ ClickHouse "
            f"ra CSV và đặt vào {RAW_DIR}/"
        )
    return pd.read_csv(path)
 
def main():
    laptop_model = load_csv("laptop_model")
    gpu_model = load_csv("gpu_model")
    brand = load_csv("brand")
    cpu_model = load_csv("cpu_model")
    benchmark = load_csv("laptop_benchmark_result")

    # ---------- 2. Đổi tên cột trước khi merge để tránh đụng độ ----------
    # Các bảng đều có sẵn: id, name, created_on, changed_on, created_by_fk,
    # changed_by_fk, is_active, elton_created_at -> phải prefix riêng từng bảng
    # để không bị pandas tự thêm hậu tố _x / _y gây khó đọc.
 
    laptop_model = laptop_model.rename(columns={
        "id": "laptop_model_id",
        "name": "laptop_name",
        "cpu_note": "laptop_cpu_note",
        "gpu_note": "laptop_gpu_note",
    })
 
    brand = brand.rename(columns={
        "id": "brand_id",
        "name": "brand_name",
        "is_chip_brand": "brand_is_chip_brand",
    })[["brand_id", "brand_name", "brand_is_chip_brand"]]
 
    cpu_model = cpu_model.rename(columns={
        "id": "cpu_model_id",
        "name": "cpu_name",
        "is_active": "cpu_is_active",
    })[["cpu_model_id", "cpu_name", "cpu_is_active"]]
 
    gpu_model = gpu_model.rename(columns={
        "id": "gpu_model_id",
        "name": "gpu_name",
        "is_active": "gpu_is_active",
    })[["gpu_model_id", "gpu_name", "gpu_is_active"]]
 
    # ---------- 3. Xử lý laptop_benchmark_result: 1 laptop có thể có
    # nhiều lần test -> chỉ lấy bản ghi mới nhất theo created_on ----------
    benchmark = benchmark.sort_values("created_on").drop_duplicates(
        subset="laptop_model_id", keep="last"
    )
    benchmark = benchmark.rename(columns={
        "id": "benchmark_id",
        "note": "benchmark_note",
        "is_active": "benchmark_is_active",
    })
    # Chỉ giữ các cột thật sự cần cho scoring/optimization, bỏ metadata thừa
    benchmark_cols = [
        "laptop_model_id",
        "office_battery_result_minutes",
        "gaming_battery_result_minutes",
        "geekbench_6_compute_gpu_plugged_in",
        "geekbench_6_compute_gpu_battery",
        "geekbench_6_cpu_single_core_plugged_in",
        "geekbench_6_cpu_single_core_battery",
        "geekbench_6_cpu_multi_core_plugged_in",
        "geekbench_6_cpu_multi_core_battery",
        "foldable_opening_battery_result_minutes",
        "review_video_url",
    ]
    benchmark = benchmark[[c for c in benchmark_cols if c in benchmark.columns]]
 
    # ---------- 4. Merge tất cả lại theo laptop_model làm gốc ----------
    df = (
        laptop_model
        .merge(brand, on="brand_id", how="left")
        .merge(cpu_model, on="cpu_model_id", how="left")
        .merge(gpu_model, on="gpu_model_id", how="left")
        .merge(benchmark, on="laptop_model_id", how="left")
    )
 
    # ---------- 5. Kiểm tra sanity trước khi lưu ----------
    n_before = len(laptop_model)
    n_after = len(df)
    df['laptop_weight'] = df['laptop_weight'].apply(lambda x: x / 1000 if x > 20 else x)
    df['price'] = df['price'].apply(lambda x: x * 1000000)
    if n_before != n_after:
        print(
            f"CẢNH BÁO: số dòng trước merge ({n_before}) khác sau merge "
            f"({n_after}) — có thể do JOIN bị nhân bản dòng. Kiểm tra lại "
            f"khóa brand_id/cpu_model_id/gpu_model_id có bị trùng không."
        )
    else:
        print(f" Merge thành công, giữ nguyên {n_after} dòng (1 dòng = 1 laptop).")
 
    # ---------- 6. Xuất file ----------
    out_path = OUT_DIR / "laptop_dataset.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f" Đã lưu: {out_path} ({len(df)} dòng, {len(df.columns)} cột)")
    
    return df
 
 
def build_engagement_features():
    """
    Xử lý RIÊNG cho user_event_tracking (khác grain, không gộp vào file trên).
    Bóc JSON từ cột event_data/device, rồi tổng hợp thành feature theo
    laptop_model_id (mapped từ device_id) để dùng cho LightGBM.

    Ghi chú cấu trúc thực tế của event_data:
      - pageview/load_more: {"page_name": ..., "device_id": ..., "url": ..., "referrer": ...}
      - search: {"keyword": ...}
      - compare: {"device_id": ...}
    """
    import json

    events = load_csv("user_event_tracking")

    def safe_json_get(raw, key):
        try:
            obj = json.loads(raw) if isinstance(raw, str) else {}
            return obj.get(key)
        except (json.JSONDecodeError, TypeError):
            return None

    # Key thực tế trong event_data là "device_id", không phải "laptop_model_id"
    events["laptop_model_id"] = events["event_data"].apply(
        lambda x: safe_json_get(x, "device_id")
    )
    events["search_keyword"] = events["event_data"].apply(
        lambda x: safe_json_get(x, "keyword")
    )
    # Key thực tế trong device là "os_name", không phải "os"
    events["device_os"] = events["device"].apply(
        lambda x: safe_json_get(x, "os_name")
    )

    # Chỉ giữ các event có gắn laptop (device_id)
    events = events.dropna(subset=["laptop_model_id"])

    # Tên event_name thực tế: "pageview", "search_for_device",
    # "add_to_comparison", "select_device_for_comparison", "load_more_device_home"
    engagement = (
        events.groupby("laptop_model_id")
        .agg(
            total_pageview=("event_name", lambda s: (s == "pageview").sum()),
            total_load_more=("event_name", lambda s: (s == "load_more_device_home").sum()),
            total_add_to_compare=("event_name", lambda s: (s == "add_to_comparison").sum()),
            total_select_compare=("event_name", lambda s: (s == "select_device_for_comparison").sum()),
            total_search_hit=("event_name", lambda s: (s == "search_for_device").sum()),
            unique_users=("user_psuedo_id", "nunique"),
        )
        .reset_index()
    )
    
    out_path = OUT_DIR / "laptop_engagement_features.csv"
    engagement.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"📁 Đã lưu: {out_path} ({len(engagement)} dòng)")

    return engagement
 
 
if __name__ == "__main__":
    main()
    build_engagement_features()
 
    
    