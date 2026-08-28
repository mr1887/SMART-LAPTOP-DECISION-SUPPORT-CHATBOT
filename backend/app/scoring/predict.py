"""
Tầng 2 - LightGBM Scoring: Dự đoán AI_Score cho từng laptop.

Load model đã train từ model.pkl, nhận DataFrame laptop, trả về DataFrame
có thêm cột AI_Score trong khoảng [0, 1].
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Đường dẫn mặc định tới model (tính từ file này)
_MODEL_PATH = Path(__file__).parent / "model.pkl"

_model = None
_feature_cols: list[str] = []


def _load_model():
    """Lazy-load model từ file pkl, cache lại để không load lại nhiều lần."""
    global _model, _feature_cols
    if _model is not None:
        return

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {_MODEL_PATH}. "
            "Hãy chạy `python backend/app/scoring/train_lgbm.py` trước."
        )

    with open(_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    _model = payload["model"]
    _feature_cols = payload["feature_cols"]


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Dự đoán AI_Score cho toàn bộ laptop trong DataFrame.

    Args:
        df: DataFrame laptop có đủ cột feature (price, laptop_weight, ...)
            Các cột thiếu sẽ được điền 0/False để tránh lỗi.

    Returns:
        DataFrame gốc với cột AI_Score được thêm/cập nhật (giá trị [0, 1]).
    """
    _load_model()

    df = df.copy()

    # Đảm bảo tất cả feature cols tồn tại (điền 0 nếu thiếu)
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Ép categorical về đúng dtype
    from app.scoring.train_lgbm import CATEGORICAL_FEATURES
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    raw_pred = _model.predict(df[_feature_cols])
    df["AI_Score"] = np.clip(raw_pred, 0, 1)

    return df


def get_feature_cols() -> list[str]:
    """Trả về danh sách feature columns model đang dùng."""
    _load_model()
    return list(_feature_cols)
