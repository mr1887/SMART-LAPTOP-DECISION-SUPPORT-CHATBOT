"""
Tầng 2 - LightGBM Scoring.
Train model dự đoán AI_Score (độ hấp dẫn/phù hợp) cho từng laptop, dựa trên
thông số phần cứng + nhãn auto-tag + tín hiệu hành vi (engagement).

QUAN TRỌNG - Định nghĩa target:
Vì không có nhãn "độ hấp dẫn" trực tiếp, ta dùng ENGAGEMENT SCORE (tự tổng
hợp từ user_event_tracking) làm proxy target để train:
    engagement_score = total_pageview + 0.5*total_load_more + 1.5*total_search_hit
                       + 2*total_add_to_compare + 2.5*total_select_compare
Trọng số ưu tiên "compare" cao nhất vì đây là hành vi cân nhắc kỹ trước khi
mua, tín hiệu mạnh hơn click đơn thuần. Laptop không có engagement (xem
mục has_geekbench_data - vấn đề Cold-Start) được gán engagement_score = 0,
KHÔNG loại bỏ khỏi tập train, để model học được cả những máy "ít nổi".

Input:
    - data/processed/laptop_dataset_tagged.csv       (từ clean_and_tag_dataset.py)
    - data/processed/laptop_engagement_features.csv  (từ merge_laptop_tables.py)

Output:
    - backend/app/scoring/model.pkl                  (model đã train)
    - data/processed/laptop_dataset_scored.csv        (toàn bộ laptop kèm AI_Score)


"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------- Cấu hình ----------
CATEGORICAL_FEATURES = ["brand_name", "cpu_name", "gpu_name"]

NUMERIC_FEATURES = [
    "price", "laptop_weight", "battery_capacity_whr",
    "office_battery_minutes_final", "screen_size", "screen_ppi",
    "geekbench_cpu_single", "geekbench_cpu_multi",
]

BOOLEAN_FEATURES = [
    "is_gaming_laptop", "is_workstation", "is_mobile_device",
    "has_geekbench_data",
    "is_gaming_friendly", "is_office_friendly",
    "is_programming_friendly", "is_graphic_friendly",
]

ENGAGEMENT_WEIGHTS = {
    "total_pageview": 1.0,         # xem trang sản phẩm
    "total_load_more": 0.5,        # load thêm thông tin (tín hiệu yếu)
    "total_search_hit": 1.5,       # xuất hiện trong kết quả tìm kiếm
    "total_add_to_compare": 2.0,   # thêm vào so sánh (tín hiệu cân nhắc kỹ)
    "total_select_compare": 2.5,   # chọn so sánh trực tiếp (tín hiệu mạnh nhất)
}


def _normalize_id_column(df: pd.DataFrame, col: str = "laptop_model_id") -> pd.DataFrame:
    """Ép ID về string CHUẨN HÓA - loại bỏ trường hợp '1' vs '1.0' do cột
    gốc từng đi qua float64 (thường do có NaN trong cột trước khi ép kiểu)."""
    # Nếu là số (int/float), ép về int trước rồi mới sang str để tránh đuôi ".0"
    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.notna().all():
        df[col] = numeric.astype("int64").astype(str)
    else:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_data(dataset_path: Path, engagement_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        sys.exit(f"Không tìm thấy {dataset_path}. Chạy clean_and_tag_dataset.py trước.")
    df = pd.read_csv(dataset_path)
    print(f"Đã đọc dataset: {len(df)} laptop, {len(df.columns)} cột")

    df = _normalize_id_column(df)

    # Phát hiện + loại trùng laptop_model_id trong chính dataset - nếu không
    # xử lý, merge phía dưới sẽ nhân bản dòng (cartesian) như đã gặp.
    n_dup = df["laptop_model_id"].duplicated().sum()
    if n_dup > 0:
        print(
            f"CẢNH BÁO: {n_dup} laptop_model_id bị TRÙNG trong dataset gốc "
            f"- giữ lại dòng đầu tiên, loại bỏ phần trùng lặp."
        )
        df = df.drop_duplicates(subset="laptop_model_id", keep="first")

    if engagement_path.exists():
        eng = pd.read_csv(engagement_path)
        eng = _normalize_id_column(eng)

        # Gộp (sum) nếu 1 laptop_model_id có nhiều dòng engagement - tránh
        # nhân bản dòng khi merge, đồng thời không mất dữ liệu (cộng dồn
        # đúng ý nghĩa "tổng số click/compare" thay vì loại bỏ).
        eng_dup = eng["laptop_model_id"].duplicated().sum()
        if eng_dup > 0:
            print(
                f"CẢNH BÁO: {eng_dup} laptop_model_id bị TRÙNG trong "
                f"engagement - gộp (sum) các dòng trùng lại thành 1."
            )
            numeric_cols = eng.select_dtypes(include="number").columns.tolist()
            eng = eng.groupby("laptop_model_id", as_index=False)[numeric_cols].sum()

        # Kiểm tra sớm: có bao nhiêu ID thực sự khớp được giữa 2 file -
        # nếu quá thấp, khả năng cao là format ID vẫn lệch nhau.
        matched = df["laptop_model_id"].isin(eng["laptop_model_id"]).sum()
        print(
            f"Kiểm tra khớp ID: {matched}/{len(df)} laptop trong dataset "
            f"tìm thấy engagement tương ứng."
        )
        if matched == 0:
            print(
                "CẢNH BÁO NGHIÊM TRỌNG: 0 ID khớp được. Kiểm tra lại 2 file "
                "CSV - in thử vài giá trị laptop_model_id ở mỗi file để so "
                "sánh format:\n"
                f"  dataset:    {df['laptop_model_id'].head(3).tolist()}\n"
                f"  engagement: {eng['laptop_model_id'].head(3).tolist()}"
            )

        df = df.merge(eng, on="laptop_model_id", how="left")
        print(f"Sau merge: {len(df)} dòng (phải bằng số laptop gốc ở trên)")
    else:
        print(f"CẢNH BÁO: không tìm thấy {engagement_path}, dùng engagement = 0 cho toàn bộ.")
        for col in ENGAGEMENT_WEIGHTS:
            df[col] = 0

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Tính engagement_score thô, rồi chuẩn hóa log + min-max về [0,1]
    làm target AI_Score cho model học theo."""
    missing_cols = [col for col in ENGAGEMENT_WEIGHTS if col not in df.columns]
    if missing_cols:
        available_engagement_like = [
            c for c in df.columns
            if any(k in c.lower() for k in ["click", "compare", "search", "view", "event"])
        ]
        print(
            f"\nCẢNH BÁO NGHIÊM TRỌNG: không tìm thấy cột {missing_cols} trong "
            f"dataset sau merge. target_ai_score SẼ BỊ CỐ ĐỊNH = 0.5 cho toàn "
            f"bộ laptop nếu không sửa việc này.\n"
            f"Các cột trong dataset có vẻ liên quan đến engagement: "
            f"{available_engagement_like or 'KHÔNG có cột nào'}\n"
            f"-> Kiểm tra lại tên cột thật trong laptop_engagement_features.csv "
            f"và sửa ENGAGEMENT_WEIGHTS ở đầu file này cho khớp."
        )

    for col in ENGAGEMENT_WEIGHTS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df["engagement_score_raw"] = sum(
        df[col] * weight for col, weight in ENGAGEMENT_WEIGHTS.items()
    )

    total_engagement = df["engagement_score_raw"].sum()
    n_nonzero = (df["engagement_score_raw"] > 0).sum()
    print(
        f"Tổng engagement_score_raw toàn dataset: {total_engagement:.1f} "
        f"({n_nonzero}/{len(df)} laptop có engagement > 0)"
    )
    if total_engagement == 0:
        print(
            "CẢNH BÁO NGHIÊM TRỌNG: TOÀN BỘ dataset có engagement_score = 0. "
            "target_ai_score sẽ giống hệt nhau ở mọi laptop (0.5), model sẽ "
            "không học được gì (RMSE = 0 nhưng vô nghĩa). Nguyên nhân thường "
            "gặp: (1) sai tên cột như cảnh báo trên, (2) merge không khớp "
            "được laptop_model_id dù đã báo 'matched' ở bước trước (kiểm tra "
            "lại xem cột total_click/... có bị đổi tên sau merge do trùng "
            "tên với cột khác không, VD bị thêm hậu tố _x/_y)."
        )

    # Log transform vì phân phối click/compare thường lệch phải mạnh
    # (vài laptop hot áp đảo, đa số ít tương tác)
    log_score = np.log1p(df["engagement_score_raw"])

    score_min, score_max = log_score.min(), log_score.max()
    if score_max > score_min:
        df["target_ai_score"] = (log_score - score_min) / (score_max - score_min)
    else:
        df["target_ai_score"] = 0.5  # toàn bộ engagement bằng nhau (VD tất cả = 0)

    n_zero_engagement = (df["engagement_score_raw"] == 0).sum()
    print(
        f"Target đã tạo: {n_zero_engagement}/{len(df)} laptop không có engagement "
        f"(target_ai_score = {df.loc[df['engagement_score_raw']==0, 'target_ai_score'].iloc[0]:.3f} "
        f"cho nhóm này)"
    )
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Chuẩn bị ma trận feature cho LightGBM. Cột categorical được ép
    dtype 'category' để LightGBM xử lý native, không cần one-hot encode.
    Giữ nguyên NaN ở numeric features - LightGBM tự học cách split."""
    feature_cols = []

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
            feature_cols.append(col)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
            feature_cols.append(col)

    missing = set(NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES) - set(feature_cols)
    if missing:
        print(f"CẢNH BÁO: các cột feature kỳ vọng nhưng không có trong dataset: {missing}")

    return df, feature_cols


def train_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple[lgb.LGBMRegressor, dict]:
    X = df[feature_cols]
    y = df["target_ai_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,       # dataset nhỏ (156 dòng) - tránh overfit, cây nông
        min_child_samples=5,  # tương ứng dataset nhỏ, mặc định 20 sẽ quá lớn
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=categorical_cols,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # Baseline so sánh: dự đoán bằng trung bình toàn tập train (mô hình "ngây thơ")
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5

    metrics = {
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "improvement_pct": (baseline_rmse - rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0,
    }

    print(f"\n--- Kết quả train ---")
    print(f"Train: {metrics['n_train']} laptop, Test: {metrics['n_test']} laptop")
    print(f"RMSE model:    {rmse:.4f}")
    print(f"RMSE baseline: {baseline_rmse:.4f} (dự đoán bằng trung bình)")
    print(f"Cải thiện so với baseline: {metrics['improvement_pct']:.1f}%")

    if metrics["improvement_pct"] < 5:
        print(
            "CẢNH BÁO: model cải thiện rất ít so với baseline ngây thơ. "
            "Với dataset chỉ 156 dòng, cân nhắc: (1) thêm dữ liệu, "
            "(2) đơn giản hóa model (giảm num_leaves), "
            "(3) hoặc dùng feature importance bên dưới để kiểm tra "
            "feature nào thực sự có ích."
        )

    print("\n--- Feature importance (top 10) ---")
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print(importance.sort_values(ascending=False).head(10).to_string())

    return model, metrics


def score_all_laptops(model: lgb.LGBMRegressor, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Dự đoán AI_Score cho TOÀN BỘ laptop (kể cả những máy dùng để train) -
    đây là điểm cuối cùng gắn vào từng laptop cho Gurobi/PuLP sử dụng."""
    raw_pred = model.predict(df[feature_cols])
    # Ép về đúng khoảng [0,1] vì regression có thể dự đoán vượt biên nhẹ
    df["AI_Score"] = np.clip(raw_pred, 0, 1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/laptop_dataset_tagged.csv"))
    parser.add_argument("--engagement", type=Path, default=Path("data/processed/laptop_engagement_features.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/app/scoring/model.pkl"))
    parser.add_argument("--scored-out", type=Path, default=Path("data/processed/laptop_dataset_scored.csv"))
    args = parser.parse_args()

    df = load_data(args.dataset, args.engagement)
    df = build_target(df)
    df, feature_cols = prepare_features(df)

    print(f"\nSố feature dùng để train: {len(feature_cols)}")
    print(f"Danh sách: {feature_cols}")

    model, metrics = train_model(df, feature_cols)

    df = score_all_laptops(model, df, feature_cols)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"\nĐã lưu model: {args.model_out}")

    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.scored_out, index=False, encoding="utf-8-sig")
    print(f"Đã lưu dataset kèm AI_Score: {args.scored_out}")

    print(f"\n--- Top 5 laptop AI_Score cao nhất ---")
    top5 = df.nlargest(10, "AI_Score")[["laptop_model_id", "laptop_name", "AI_Score"]]
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()