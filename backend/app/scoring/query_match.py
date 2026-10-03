"""
Mô-đun tính điểm tương thích truy vấn (Query Match Score).
Tính toán mức độ phù hợp tức thời của từng laptop với câu hỏi/tiêu chí người dùng:
- Ràng buộc mềm về giá (Price soft fit)
- Sở thích về trọng lượng (Weight preference)
- Ràng buộc mềm về pin (Battery soft fit)
- Nhãn nhu cầu (Required tags match)
- Sở thích/từ khóa về GPU (GPU preference)
"""

import re
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from app.nlp.schema import Constraint, Preference, RequirementSet


def _is_discrete_gpu(gpu_name: Any) -> bool:
    """Xác định GPU có phải là card đồ họa rời (Discrete GPU) hay không."""
    if not gpu_name or not isinstance(gpu_name, str):
        return False
    name_lower = gpu_name.lower()
    if any(k in name_lower for k in ["tích hợp", "integrated", "adreno", "uhd", "iris", "intel graphics", "apple m"]):
        return False
    if any(k in name_lower for k in ["rtx", "gtx", "geforce", "nvidia", "radeon rx", "arc b", "discrete"]):
        return True
    return False


def _matches_gpu_keyword(gpu_name: Any, keyword: Optional[str]) -> bool:
    """Kiểm tra GPU có khớp với từ khóa/dòng GPU yêu cầu hay không."""
    if not keyword:
        return True
    if not gpu_name or not isinstance(gpu_name, str):
        return False
    kw_tokens = re.sub(r"\s+", " ", str(keyword).lower().strip()).split()
    gpu_clean = re.sub(r"\s+", " ", str(gpu_name).lower().strip())
    return all(tok in gpu_clean for tok in kw_tokens)


def compute_query_match_score(
    df: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any], Any]
) -> pd.Series:
    """
    Tính điểm Query Match Score trong khoảng [0, 1] cho từng laptop:
    Đánh giá mức độ phù hợp đối với các tiêu chí mềm và sở thích trong truy vấn hiện tại.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    scores: list[pd.Series] = []
    weights: list[float] = []

    # Parse requirements
    constraints: list[Union[Constraint, dict[str, Any]]] = []
    preferences: list[Union[Preference, dict[str, Any]]] = []
    required_tags: list[str] = []

    if isinstance(requirements, RequirementSet):
        constraints = requirements.constraints
        preferences = requirements.preferences
        required_tags = requirements.required_tags
    elif isinstance(requirements, dict):
        if "constraints" in requirements:
            constraints = requirements.get("constraints", [])
            preferences = requirements.get("preferences", [])
            required_tags = requirements.get("required_tags", [])
        else:
            from app.nlp.nl2constraint import convert_legacy_regex_to_requirement_set
            req_set = convert_legacy_regex_to_requirement_set(requirements)
            constraints = req_set.get("constraints", [])
            preferences = req_set.get("preferences", [])
            required_tags = req_set.get("required_tags", [])

    # 1. Price soft fit
    if "price" in df.columns:
        price_col = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median() if not df["price"].empty else 20000000)
        soft_price_constraints = [
            c for c in constraints
            if (isinstance(c, Constraint) and c.field == "price" and c.type == "soft")
            or (isinstance(c, dict) and c.get("field") == "price" and c.get("type") == "soft")
        ]
        price_pref = [
            p for p in preferences
            if (isinstance(p, Preference) and p.field == "price")
            or (isinstance(p, dict) and p.get("field") == "price")
        ]

        if soft_price_constraints:
            for c in soft_price_constraints:
                val = float(c.value if isinstance(c, Constraint) else c["value"])
                op = c.operator if isinstance(c, Constraint) else c.get("operator", "<=")
                if op == "<=":
                    # Nếu giá <= budget: 1.0; nếu vượt quá: giảm tuyến tính trong biên độ 30%
                    p_score = np.where(
                        price_col <= val,
                        1.0,
                        np.clip(1.0 - (price_col - val) / (0.3 * val + 1e-5), 0.0, 1.0)
                    )
                    scores.append(pd.Series(p_score, index=df.index))
                    weights.append(1.0)
                elif op == ">=":
                    p_score = np.where(
                        price_col >= val,
                        1.0,
                        np.clip(1.0 - (val - price_col) / (0.3 * val + 1e-5), 0.0, 1.0)
                    )
                    scores.append(pd.Series(p_score, index=df.index))
                    weights.append(1.0)
        elif price_pref:
            direction = price_pref[0].direction if isinstance(price_pref[0], Preference) else price_pref[0].get("direction", "minimize")
            p_min, p_max = price_col.min(), price_col.max()
            if p_max > p_min:
                if direction == "minimize":
                    p_score = (p_max - price_col) / (p_max - p_min)
                else:
                    p_score = (price_col - p_min) / (p_max - p_min)
                scores.append(pd.Series(p_score, index=df.index))
                weights.append(1.0)

    # 2. Weight preference & soft fit
    if "laptop_weight" in df.columns:
        weight_col = pd.to_numeric(df["laptop_weight"], errors="coerce").fillna(2.0)
        soft_weight_constraints = [
            c for c in constraints
            if (isinstance(c, Constraint) and c.field == "weight_kg" and c.type == "soft")
            or (isinstance(c, dict) and c.get("field") == "weight_kg" and c.get("type") == "soft")
        ]
        weight_pref = [
            p for p in preferences
            if (isinstance(p, Preference) and p.field == "weight_kg")
            or (isinstance(p, dict) and p.get("field") == "weight_kg")
        ]

        if weight_pref or soft_weight_constraints:
            # Máy càng nhẹ điểm càng cao (thang đo chuẩn [1.0kg, 2.6kg])
            w_score = np.clip((2.6 - weight_col) / (2.6 - 1.0), 0.0, 1.0)
            scores.append(pd.Series(w_score, index=df.index))
            weights.append(1.0)

    # 3. Battery soft fit & preference
    battery_col_name = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else ("office_battery_result_minutes" if "office_battery_result_minutes" in df.columns else None)
    if battery_col_name:
        bat_col = pd.to_numeric(df[battery_col_name], errors="coerce").fillna(300.0)
        soft_bat_constraints = [
            c for c in constraints
            if (isinstance(c, Constraint) and c.field == "battery_minutes" and c.type == "soft")
            or (isinstance(c, dict) and c.get("field") == "battery_minutes" and c.get("type") == "soft")
        ]
        bat_pref = [
            p for p in preferences
            if (isinstance(p, Preference) and p.field == "battery_minutes")
            or (isinstance(p, dict) and p.get("field") == "battery_minutes")
        ]

        if soft_bat_constraints or bat_pref:
            # Pin càng lâu điểm càng cao (thang đo chuẩn [180 phút, 600 phút])
            b_score = np.clip((bat_col - 180.0) / (600.0 - 180.0), 0.0, 1.0)
            scores.append(pd.Series(b_score, index=df.index))
            weights.append(1.0)

    # 4. Required tags match
    if required_tags:
        matching_tag_cols = [tag for tag in required_tags if tag in df.columns]
        if matching_tag_cols:
            tag_matches = df[matching_tag_cols].fillna(False).astype(int).sum(axis=1)
            t_score = tag_matches / len(required_tags)
            scores.append(pd.Series(t_score, index=df.index))
            weights.append(1.2)

    # 5. GPU preference
    if "gpu_name" in df.columns:
        gpu_prefs = [
            p for p in preferences
            if (isinstance(p, Preference) and p.field in ("gpu_discrete", "gpu_keyword"))
            or (isinstance(p, dict) and p.get("field") in ("gpu_discrete", "gpu_keyword"))
        ]
        soft_gpu_constraints = [
            c for c in constraints
            if (isinstance(c, Constraint) and c.field in ("gpu_discrete", "gpu_keyword") and c.type == "soft")
            or (isinstance(c, dict) and c.get("field") in ("gpu_discrete", "gpu_keyword") and c.get("type") == "soft")
        ]

        if gpu_prefs or soft_gpu_constraints:
            def _gpu_score(row: Any) -> float:
                g_name = row.get("gpu_name", "")
                disc = _is_discrete_gpu(g_name)
                return 1.0 if disc else 0.2

            g_score = df.apply(_gpu_score, axis=1)
            scores.append(pd.Series(g_score, index=df.index))
            weights.append(1.0)

    # Tính điểm tổng hợp (weighted average)
    if not scores:
        return pd.Series(1.0, index=df.index, dtype=float)

    total_weight = sum(weights)
    final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return final_score.clip(0.0, 1.0)


def combine_scores(
    global_relevance: Union[pd.Series, np.ndarray, float],
    query_match: Union[pd.Series, np.ndarray, float],
    alpha: float = 0.6,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Kết hợp Global Relevance Score (từ mô hình LightGBM) và Query Match Score:
        final_relevance_score = alpha * global_relevance + (1 - alpha) * query_match

    Args:
        global_relevance: Điểm liên quan toàn cục (offline engagement relevance).
        query_match: Điểm tương thích ngữ cảnh truy vấn (query match).
        alpha: Trọng số cân bằng trong đoạn [0, 1] (mặc định 0.6).

    Returns:
        Điểm tổng hợp đã được clamp về khoảng [0, 1].
    """
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    combined = alpha_clamped * global_relevance + (1.0 - alpha_clamped) * query_match

    if isinstance(combined, (pd.Series, pd.DataFrame)):
        return combined.clip(0.0, 1.0)
    elif isinstance(combined, np.ndarray):
        return np.clip(combined, 0.0, 1.0)
    else:
        return float(np.clip(combined, 0.0, 1.0))

