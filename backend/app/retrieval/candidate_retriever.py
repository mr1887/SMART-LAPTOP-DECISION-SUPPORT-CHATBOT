"""
Tầng Retrieval: Tiền lọc các ứng viên (Candidate Retrieval) an toàn trước khi đưa vào mô hình tối ưu.
Chỉ lọc các ràng buộc CỨNG (hard constraints), bỏ qua các ràng buộc mềm và sở thích.
"""

import re
from typing import Any, Optional, Union
import pandas as pd

from app.nlp.schema import Constraint, RequirementSet


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


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Tìm cột đầu tiên tồn tại trong DataFrame từ danh sách các tên cột tiềm năng."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def retrieve_candidates(
    df: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any], Any]
) -> pd.DataFrame:
    """
    Lọc các ứng viên thỏa mãn các ràng buộc CỨNG (hard constraints) an toàn trước khi tối ưu:
    - Chỉ filter hard constraints (price, ram_gb, storage_gb, gpu_discrete, gpu_keyword).
    - Không filter soft constraints hay preferences.
    - Bỏ qua constraint và ghi warning nếu cột cần thiết không tồn tại trong DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    filtered_df = df.copy()

    # Chuẩn hóa requirements thành danh sách constraints
    constraints_list: list[Union[Constraint, dict[str, Any]]] = []
    if isinstance(requirements, RequirementSet):
        constraints_list = requirements.constraints
    elif isinstance(requirements, dict):
        if "constraints" in requirements and isinstance(requirements["constraints"], list):
            constraints_list = requirements["constraints"]
        else:
            from app.nlp.nl2constraint import convert_legacy_regex_to_requirement_set
            req_set = convert_legacy_regex_to_requirement_set(requirements)
            constraints_list = req_set.get("constraints", [])

    for c in constraints_list:
        if isinstance(c, Constraint):
            c_field = c.field
            c_op = c.operator
            c_val = c.value
            c_type = c.type
        elif isinstance(c, dict):
            c_field = c.get("field")
            c_op = c.get("operator")
            c_val = c.get("value")
            c_type = c.get("type", "hard")
        else:
            continue

        # Chỉ lọc ràng buộc cứng (hard)
        if c_type != "hard":
            continue

        if c_field == "price":
            col = _find_column(filtered_df, ["price", "price_vnd", "laptop_price"])
            if col is None:
                print("[CandidateRetriever Warning] Column 'price' not found in DataFrame, skipping constraint.")
                continue
            try:
                val_num = float(c_val)
                if c_op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val_num]
                elif c_op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val_num]
                elif c_op == "=":
                    filtered_df = filtered_df[filtered_df[col] == val_num]
            except Exception as e:
                print(f"[CandidateRetriever Warning] Error filtering price ({e}), skipping constraint.")

        elif c_field == "ram_gb":
            col = _find_column(filtered_df, ["ram_gb", "ram", "ram_capacity", "ram_size"])
            if col is None:
                print("[CandidateRetriever Warning] Column for 'ram_gb' not found in DataFrame, skipping constraint.")
                continue
            try:
                val_num = float(c_val)
                if c_op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val_num]
                elif c_op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val_num]
                elif c_op == "=":
                    filtered_df = filtered_df[filtered_df[col] == val_num]
            except Exception as e:
                print(f"[CandidateRetriever Warning] Error filtering ram_gb ({e}), skipping constraint.")

        elif c_field == "storage_gb":
            col = _find_column(filtered_df, ["storage_gb", "storage", "ssd_gb", "ssd_capacity", "storage_capacity"])
            if col is None:
                print("[CandidateRetriever Warning] Column for 'storage_gb' not found in DataFrame, skipping constraint.")
                continue
            try:
                val_num = float(c_val)
                if c_op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val_num]
                elif c_op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val_num]
                elif c_op == "=":
                    filtered_df = filtered_df[filtered_df[col] == val_num]
            except Exception as e:
                print(f"[CandidateRetriever Warning] Error filtering storage_gb ({e}), skipping constraint.")

        elif c_field == "weight_kg":
            col = _find_column(filtered_df, ["laptop_weight", "weight_kg", "weight"])
            if col is None:
                print("[CandidateRetriever Warning] Column for 'weight_kg' not found in DataFrame, skipping constraint.")
                continue
            try:
                val_num = float(c_val)
                if c_op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val_num]
                elif c_op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val_num]
                elif c_op == "=":
                    filtered_df = filtered_df[filtered_df[col] == val_num]
            except Exception as e:
                print(f"[CandidateRetriever Warning] Error filtering weight_kg ({e}), skipping constraint.")

        elif c_field == "battery_minutes":
            col = _find_column(filtered_df, ["office_battery_minutes_final", "office_battery_result_minutes", "battery_minutes"])
            if col is None:
                print("[CandidateRetriever Warning] Column for 'battery_minutes' not found in DataFrame, skipping constraint.")
                continue
            try:
                val_num = float(c_val)
                if c_op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val_num]
                elif c_op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val_num]
                elif c_op == "=":
                    filtered_df = filtered_df[filtered_df[col] == val_num]
            except Exception as e:
                print(f"[CandidateRetriever Warning] Error filtering battery_minutes ({e}), skipping constraint.")

        elif c_field == "gpu_discrete":
            disc_col = _find_column(filtered_df, ["gpu_discrete", "is_discrete_gpu"])
            gpu_name_col = _find_column(filtered_df, ["gpu_name", "gpu"])
            req_val = bool(c_val)

            if disc_col is not None:
                filtered_df = filtered_df[filtered_df[disc_col] == req_val]
            elif gpu_name_col is not None:
                mask = filtered_df[gpu_name_col].apply(
                    lambda g: _is_discrete_gpu(g) if req_val else not _is_discrete_gpu(g)
                )
                filtered_df = filtered_df[mask]
            else:
                print("[CandidateRetriever Warning] Column for GPU not found in DataFrame, skipping 'gpu_discrete' constraint.")

        elif c_field == "gpu_keyword":
            gpu_name_col = _find_column(filtered_df, ["gpu_name", "gpu"])
            if gpu_name_col is None:
                print("[CandidateRetriever Warning] Column 'gpu_name' not found in DataFrame, skipping 'gpu_keyword' constraint.")
                continue
            kw_str = str(c_val).strip()
            mask = filtered_df[gpu_name_col].apply(lambda g: _matches_gpu_keyword(g, kw_str))
            filtered_df = filtered_df[mask]

    return filtered_df
