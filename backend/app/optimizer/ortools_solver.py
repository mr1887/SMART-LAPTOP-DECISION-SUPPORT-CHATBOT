"""
Bộ tối ưu hóa lựa chọn Laptop sử dụng Google OR-Tools CP-SAT Solver.
Hỗ trợ:
- Ràng buộc CỨNG (Hard Constraints): Tuyệt đối không có slack, bắt buộc thỏa mãn 100%.
- Ràng buộc MỀM (Soft Constraints): Cho phép vi phạm thông qua các biến bù (Slack variables) với hàm phạt theo đơn vị chuẩn hóa.
"""

import re
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from app.nlp.schema import Constraint, RequirementSet

# ==============================================================================
# 1. Định nghĩa đơn vị vi phạm chuẩn hóa (Integer Violation Units)
# Tránh phạt trực tiếp theo raw VND/kg/phút; mỗi vi phạm được scale về đơn vị chuẩn:
# ==============================================================================
PRICE_VIOLATION_UNIT_VND = 100_000          # 1 unit vi phạm = 100,000 VNĐ
WEIGHT_VIOLATION_UNIT_GRAMS = 100           # 1 unit vi phạm = 100 grams (0.1 kg)
BATTERY_VIOLATION_UNIT_MINUTES = 10         # 1 unit vi phạm = 10 phút

# ==============================================================================
# 2. Trọng số phạt trên mỗi đơn vị vi phạm (Penalty per Violation Unit)
# Thang điểm relevance_score được scale = 10,000 (tương ứng [0.0, 1.0] -> [0, 10,000])
# ==============================================================================
PENALTY_PER_UNIT = {
    "price": 20,       # Mỗi 100,000 VNĐ vượt ngân sách -> trừ 20 điểm (0.2% relevance)
    "weight": 100,     # Mỗi 100g (0.1 kg) vượt cân nặng  -> trừ 100 điểm (1.0% relevance)
    "battery": 20,     # Mỗi 10 phút thiếu pin           -> trừ 20 điểm (0.2% relevance)
}


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


def solve_hard_constraints(
    candidates: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any], Any],
    score_scale: int = 10000,
) -> dict[str, Any]:
    """
    Alias tương thích ngược cho solve().
    """
    return solve(candidates, requirements, score_scale=score_scale)


def solve(
    candidates: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any], Any],
    score_scale: int = 10000,
) -> dict[str, Any]:
    """
    Giải bài toán tối ưu hóa lựa chọn 1 laptop tốt nhất:
    - Ràng buộc CỨNG (hard): Không dùng slack, bắt buộc thỏa mãn tuyệt đối.
    - Ràng buộc MỀM (soft): Dùng slack variables nguyên hóa theo Integer Violation Units.
    - Hàm mục tiêu: Maximize (scaled_relevance_score - weighted_soft_penalties).

    Args:
        candidates: DataFrame tập ứng viên laptop.
        requirements: Bộ yêu cầu người dùng (RequirementSet hoặc dict).
        score_scale: Hệ số nhân chuyển điểm relevance float sang integer cho CP-SAT (mặc định 10000).

    Returns:
        dict định dạng kết quả:
        {
            "laptop_id": ...,
            "status": "OPTIMAL" | "FEASIBLE" | "INFEASIBLE",
            "is_feasible": bool,
            "relevance_score": float | None,
            "hard_violations": list[str],
            "has_soft_violation": bool,
            "soft_violations": list[dict]
        }
    """
    if candidates is None or candidates.empty:
        return {
            "laptop_id": None,
            "status": "INFEASIBLE",
            "is_feasible": False,
            "relevance_score": None,
            "hard_violations": ["Danh sách ứng viên rỗng."],
            "has_soft_violation": False,
            "soft_violations": [],
        }

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

    # Khởi tạo mô hình CP-SAT
    model = cp_model.CpModel()
    indices = list(candidates.index)

    # Biến quyết định: x[i] = 1 nếu laptop i được chọn, ngược lại = 0
    x = {i: model.NewBoolVar(f"x_{i}") for i in indices}

    # Ràng buộc cơ sở: chọn chính xác duy nhất 1 laptop
    model.Add(sum(x[i] for i in indices) == 1)

    soft_penalties = []
    soft_trackers = []

    # Áp dụng các ràng buộc (Hard & Soft)
    for c_idx, c in enumerate(constraints_list):
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

        is_hard = (c_type == "hard")

        # 1. Price constraint
        if c_field == "price":
            col = _find_column(candidates, ["price", "price_vnd", "laptop_price"])
            if col:
                target_val = int(float(c_val))
                price_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]

                if is_hard:
                    # Hard: Tuyệt đối không dùng slack
                    if c_op == "<=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val)
                    elif c_op == ">=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) >= target_val)
                    elif c_op == "=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) == target_val)
                else:
                    # Soft: Biến bù slack_p tính theo số đơn vị PRICE_VIOLATION_UNIT_VND
                    slack_p = model.NewIntVar(0, 100_000, f"slack_price_{c_idx}")
                    if c_op == "<=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + PRICE_VIOLATION_UNIT_VND * slack_p)
                    elif c_op == ">=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) + PRICE_VIOLATION_UNIT_VND * slack_p >= target_val)
                    elif c_op == "=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + PRICE_VIOLATION_UNIT_VND * slack_p)
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) + PRICE_VIOLATION_UNIT_VND * slack_p >= target_val)

                    soft_penalties.append(slack_p * PENALTY_PER_UNIT["price"])
                    soft_trackers.append({"type": "price", "var": slack_p, "col": col, "op": c_op, "target": target_val})

        # 2. RAM constraint (mặc định hard)
        elif c_field == "ram_gb":
            col = _find_column(candidates, ["ram_gb", "ram", "ram_capacity", "ram_size"])
            if col:
                target_val = int(float(c_val))
                ram_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                if is_hard:
                    if c_op == ">=":
                        model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val)
                    elif c_op == "=":
                        model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) == target_val)
                else:
                    # Limitation: Soft constraint cho RAM hiện tại chưa hỗ trợ slack variable & penalty trong solver.
                    # Bỏ qua an toàn để không crash và không tự động biến thành hard constraint.
                    pass

        # 3. Storage constraint (mặc định hard)
        elif c_field == "storage_gb":
            col = _find_column(candidates, ["storage_gb", "storage", "ssd_gb", "ssd_capacity", "storage_capacity"])
            if col:
                target_val = int(float(c_val))
                storage_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                if is_hard:
                    if c_op == ">=":
                        model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val)
                    elif c_op == "=":
                        model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) == target_val)
                else:
                    # Limitation: Soft constraint cho Storage hiện tại chưa hỗ trợ slack variable & penalty trong solver.
                    # Bỏ qua an toàn để không crash và không tự động biến thành hard constraint.
                    pass

        # 4. Weight constraint
        elif c_field == "weight_kg":
            col = _find_column(candidates, ["laptop_weight", "weight_kg", "weight"])
            if col:
                target_grams = int(float(c_val) * 1000)
                weight_grams = [int(float(candidates.loc[i, col]) * 1000) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]

                if is_hard:
                    if c_op == "<=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) <= target_grams)
                    elif c_op == ">=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) >= target_grams)
                    elif c_op == "=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) == target_grams)
                else:
                    # Soft: Biến bù slack_w tính theo số đơn vị WEIGHT_VIOLATION_UNIT_GRAMS
                    slack_w = model.NewIntVar(0, 10_000, f"slack_weight_{c_idx}")
                    if c_op == "<=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) <= target_grams + WEIGHT_VIOLATION_UNIT_GRAMS * slack_w)
                    elif c_op == ">=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) + WEIGHT_VIOLATION_UNIT_GRAMS * slack_w >= target_grams)

                    soft_penalties.append(slack_w * PENALTY_PER_UNIT["weight"])
                    soft_trackers.append({"type": "weight_kg", "var": slack_w, "col": col, "op": c_op, "target": float(c_val)})

        # 5. Battery constraint
        elif c_field == "battery_minutes":
            col = _find_column(candidates, ["office_battery_minutes_final", "office_battery_result_minutes", "battery_minutes"])
            if col:
                target_val = int(float(c_val))
                battery_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]

                if is_hard:
                    if c_op == ">=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val)
                    elif c_op == "=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) == target_val)
                else:
                    # Soft: Biến bù slack_b tính theo số đơn vị BATTERY_VIOLATION_UNIT_MINUTES
                    slack_b = model.NewIntVar(0, 10_000, f"slack_battery_{c_idx}")
                    if c_op == ">=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) + BATTERY_VIOLATION_UNIT_MINUTES * slack_b >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + BATTERY_VIOLATION_UNIT_MINUTES * slack_b)

                    soft_penalties.append(slack_b * PENALTY_PER_UNIT["battery"])
                    soft_trackers.append({"type": "battery_minutes", "var": slack_b, "col": col, "op": c_op, "target": target_val})

        # 6. GPU discrete (mặc định hard)
        elif c_field == "gpu_discrete":
            if is_hard:
                disc_col = _find_column(candidates, ["gpu_discrete", "is_discrete_gpu"])
                gpu_name_col = _find_column(candidates, ["gpu_name", "gpu"])
                req_disc = bool(c_val)

                disc_flags = []
                for i in indices:
                    if disc_col and pd.notna(candidates.loc[i, disc_col]):
                        is_d = bool(candidates.loc[i, disc_col])
                    elif gpu_name_col:
                        is_d = _is_discrete_gpu(candidates.loc[i, gpu_name_col])
                    else:
                        is_d = False
                    disc_flags.append(1 if (is_d == req_disc) else 0)

                model.Add(sum(disc_flags[idx] * x[i] for idx, i in enumerate(indices)) == 1)
            else:
                # Limitation: Soft constraint cho GPU Discrete hiện tại chưa hỗ trợ slack penalty.
                # Bỏ qua an toàn để không crash và không tự động biến thành hard constraint.
                pass

        # 7. GPU keyword (mặc định hard)
        elif c_field == "gpu_keyword":
            if is_hard:
                gpu_name_col = _find_column(candidates, ["gpu_name", "gpu"])
                kw = str(c_val).strip()
                if gpu_name_col and kw:
                    kw_flags = [
                        1 if _matches_gpu_keyword(candidates.loc[i, gpu_name_col], kw) else 0
                        for i in indices
                    ]
                    model.Add(sum(kw_flags[idx] * x[i] for idx, i in enumerate(indices)) == 1)
            else:
                # Limitation: Soft constraint cho GPU Keyword hiện tại chưa hỗ trợ slack penalty.
                # Bỏ qua an toàn để không crash và không tự động biến thành hard constraint.
                pass

    # Xác định cột điểm relevance (ưu tiên final_relevance_score -> relevance_score -> AI_Score)
    score_col = _find_column(candidates, ["final_relevance_score", "relevance_score", "AI_Score"])

    if score_col:
        scaled_scores = [
            int(float(candidates.loc[i, score_col]) * score_scale) if pd.notna(candidates.loc[i, score_col]) else 0
            for i in indices
        ]
        total_relevance = sum(scaled_scores[idx] * x[i] for idx, i in enumerate(indices))
    else:
        total_relevance = 0

    # Hàm mục tiêu: Maximize (scaled_relevance_score - weighted_soft_penalties)
    if soft_penalties:
        model.Maximize(total_relevance - sum(soft_penalties))
    else:
        model.Maximize(total_relevance)

    # Tiến hành giải bài toán
    solver = cp_model.CpSolver()
    solver_status = solver.Solve(model)

    if solver_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        chosen_idx = None
        for i in indices:
            if solver.BooleanValue(x[i]):
                chosen_idx = i
                break

        if chosen_idx is not None:
            chosen_row = candidates.loc[chosen_idx]
            laptop_id = chosen_row.get("laptop_model_id", chosen_row.get("laptop_id", chosen_idx))
            relevance_val = float(chosen_row[score_col]) if (score_col and pd.notna(chosen_row[score_col])) else None

            # Phân tích các vi phạm ràng buộc mềm (nếu có)
            soft_violations = []
            for tracker in soft_trackers:
                field_type = tracker["type"]
                target = tracker["target"]
                col = tracker["col"]
                op = tracker["op"]
                actual_raw = chosen_row.get(col)

                if field_type == "price":
                    actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
                    if op == "<=" and actual > target:
                        diff = actual - target
                        soft_violations.append({
                            "field": "price",
                            "target": target,
                            "actual": actual,
                            "violation": f"Vượt ngân sách {diff:,.0f}đ",
                        })
                    elif op == ">=" and actual < target:
                        diff = target - actual
                        soft_violations.append({
                            "field": "price",
                            "target": target,
                            "actual": actual,
                            "violation": f"Thấp hơn ngân sách tối thiểu {diff:,.0f}đ",
                        })
                elif field_type == "weight_kg":
                    actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
                    if op == "<=" and actual > target:
                        diff = actual - target
                        soft_violations.append({
                            "field": "weight_kg",
                            "target": target,
                            "actual": actual,
                            "violation": f"Nặng hơn yêu cầu {diff:.2f}kg",
                        })
                    elif op == ">=" and actual < target:
                        diff = target - actual
                        soft_violations.append({
                            "field": "weight_kg",
                            "target": target,
                            "actual": actual,
                            "violation": f"Nhẹ hơn yêu cầu {diff:.2f}kg",
                        })
                elif field_type == "battery_minutes":
                    actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
                    if op == ">=" and actual < target:
                        diff = target - actual
                        soft_violations.append({
                            "field": "battery_minutes",
                            "target": target,
                            "actual": actual,
                            "violation": f"Thời lượng pin thấp hơn yêu cầu {diff:.0f} phút",
                        })
                    elif op == "<=" and actual > target:
                        diff = actual - target
                        soft_violations.append({
                            "field": "battery_minutes",
                            "target": target,
                            "actual": actual,
                            "violation": f"Thời lượng pin cao hơn yêu cầu {diff:.0f} phút",
                        })

            return {
                "laptop_id": laptop_id,
                "status": "OPTIMAL" if solver_status == cp_model.OPTIMAL else "FEASIBLE",
                "is_feasible": True,
                "relevance_score": relevance_val,
                "hard_violations": [],
                "has_soft_violation": len(soft_violations) > 0,
                "soft_violations": soft_violations,
            }

    return {
        "laptop_id": None,
        "status": "INFEASIBLE",
        "is_feasible": False,
        "relevance_score": None,
        "hard_violations": ["No laptop satisfies all hard constraints."],
        "has_soft_violation": False,
        "soft_violations": [],
    }
