"""
Bộ tối ưu hóa lựa chọn Laptop sử dụng Google OR-Tools CP-SAT Solver.
Hỗ trợ:
- Ràng buộc CỨNG (Hard Constraints): Tuyệt đối không có slack, bắt buộc thỏa mãn 100%.
- Ràng buộc MỀM (Soft Constraints): Cho phép vi phạm thông qua các biến bù (Slack variables) với hàm phạt theo đơn vị chuẩn hóa.
"""

import re
from typing import Any, Optional, Union, cast
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
    requirements: Union[RequirementSet, dict[str, Any]],
    score_scale: int = 10000,
) -> dict[str, Any]:
    """
    Alias tương thích ngược cho solve().
    """
    return solve(candidates, requirements, score_scale=score_scale)


def solve(
    candidates: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any]],
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
        constraints_list = list(requirements.constraints)
    elif isinstance(requirements, dict):
        raw_constraints = requirements.get("constraints")
        if isinstance(raw_constraints, list):
            constraints_list = raw_constraints
        else:
            from app.nlp.nl2constraint import convert_legacy_regex_to_requirement_set
            req_set = convert_legacy_regex_to_requirement_set(requirements)
            if isinstance(req_set, dict):
                req_constraints = req_set.get("constraints")
                if isinstance(req_constraints, list):
                    constraints_list = req_constraints

    # Khởi tạo mô hình CP-SAT
    model: Any = cp_model.CpModel()
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
            c_field: str = str(c.field)
            c_op: str = str(c.operator)
            c_val: Any = c.value
            c_type: str = str(c.type)
        elif isinstance(c, dict):
            c_field = str(c.get("field", ""))
            c_op = str(c.get("operator", ""))
            c_val = c.get("value")
            c_type = str(c.get("type", "hard"))
        else:
            continue

        if c_val is None:
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
    solver: Any = cp_model.CpSolver()
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
            soft_violations = _extract_soft_violations(chosen_row, soft_trackers)



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


# ==============================================================================
# 3. Helpers & Top-3 Recommendation Optimization
# ==============================================================================

def compute_performance_scores(df: pd.DataFrame) -> pd.Series:
    """
    Tính điểm hiệu năng CPU chuẩn hóa [0.0, 1.0] dựa trên Geekbench 6:
    - single_norm = (geekbench_cpu_single - min_single) / (max_single - min_single)
    - multi_norm = (geekbench_cpu_multi - min_multi) / (max_multi - min_multi)
    - performance_score = 0.35 * single_norm + 0.65 * multi_norm

    Nếu has_geekbench_data=False hoặc benchmark bị thiếu -> fallback 0.0 (an toàn, không crash).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    single_col = _find_column(df, ["geekbench_cpu_single", "geekbench_6_cpu_single_core_plugged_in"])
    multi_col = _find_column(df, ["geekbench_cpu_multi", "geekbench_6_cpu_multi_core_plugged_in"])

    if single_col and single_col in df.columns:
        s_single = pd.to_numeric(df[single_col], errors="coerce")
    else:
        s_single = pd.Series(np.nan, index=df.index)

    if multi_col and multi_col in df.columns:
        s_multi = pd.to_numeric(df[multi_col], errors="coerce")
    else:
        s_multi = pd.Series(np.nan, index=df.index)

    has_gb_col = _find_column(df, ["has_geekbench_data"])
    if has_gb_col and has_gb_col in df.columns:
        has_gb_mask = df[has_gb_col].fillna(False).astype(bool) & (s_single.notna() | s_multi.notna())
    else:
        has_gb_mask = s_single.notna() | s_multi.notna()

    valid_single = s_single[has_gb_mask & s_single.notna()]
    valid_multi = s_multi[has_gb_mask & s_multi.notna()]

    min_single = float(valid_single.min()) if not valid_single.empty else 0.0
    max_single = float(valid_single.max()) if not valid_single.empty else 0.0

    min_multi = float(valid_multi.min()) if not valid_multi.empty else 0.0
    max_multi = float(valid_multi.max()) if not valid_multi.empty else 0.0

    if max_single > min_single:
        single_norm = (s_single - min_single) / (max_single - min_single)
    else:
        single_norm = pd.Series(1.0 if not valid_single.empty else 0.0, index=df.index)

    if max_multi > min_multi:
        multi_norm = (s_multi - min_multi) / (max_multi - min_multi)
    else:
        multi_norm = pd.Series(1.0 if not valid_multi.empty else 0.0, index=df.index)

    perf = 0.35 * single_norm.fillna(0.0) + 0.65 * multi_norm.fillna(0.0)
    perf = perf.where(has_gb_mask, 0.0).fillna(0.0)
    return perf.round(4)


def _extract_soft_violations(chosen_row: pd.Series, soft_trackers: list[dict]) -> list[dict[str, Any]]:
    """Trích xuất chi tiết các vi phạm ràng buộc mềm của 1 dòng laptop đã chọn.
    Chỉ append violation khi thực sự vi phạm (không tạo violation âm).
    """
    soft_violations = []
    for tracker in soft_trackers:
        field_type = tracker["type"]
        target: float = float(tracker["target"])
        col = tracker["col"]
        op = tracker["op"]
        actual_raw = chosen_row.get(col)

        if field_type == "price":
            actual: float = float(actual_raw) if pd.notna(actual_raw) else 0.0
            if (op == "<=" or op == "<") and actual > target:
                diff: float = actual - target
                soft_violations.append({
                    "field": "price",
                    "target": target,
                    "actual": actual,
                    "violation": f"Vượt ngân sách {diff:,.0f}đ",
                })
            elif (op == ">=" or op == ">") and actual < target:
                diff: float = target - actual
                soft_violations.append({
                    "field": "price",
                    "target": target,
                    "actual": actual,
                    "violation": f"Thấp hơn ngân sách tối thiểu {diff:,.0f}đ",
                })
        elif field_type == "weight_kg":
            actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
            if (op == "<=" or op == "<") and actual > target:
                diff = actual - target
                soft_violations.append({
                    "field": "weight_kg",
                    "target": target,
                    "actual": actual,
                    "violation": f"Nặng hơn yêu cầu {diff:.2f}kg",
                })
            elif (op == ">=" or op == ">") and actual < target:
                diff = target - actual
                soft_violations.append({
                    "field": "weight_kg",
                    "target": target,
                    "actual": actual,
                    "violation": f"Nhẹ hơn yêu cầu {diff:.2f}kg",
                })
        elif field_type == "battery_minutes":
            actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
            if (op == ">=" or op == ">") and actual < target:
                diff = target - actual
                soft_violations.append({
                    "field": "battery_minutes",
                    "target": target,
                    "actual": actual,
                    "violation": f"Thời lượng pin thấp hơn yêu cầu {diff:.0f} phút",
                })
            elif (op == "<=" or op == "<") and actual > target:
                diff = actual - target
                soft_violations.append({
                    "field": "battery_minutes",
                    "target": target,
                    "actual": actual,
                    "violation": f"Thời lượng pin cao hơn yêu cầu {diff:.0f} phút",
                })
    return soft_violations


def _build_optimization_step_model(
    candidates: pd.DataFrame,
    allowed_indices: list[Any],
    constraints_list: list[Union[Constraint, dict[str, Any]]],
    score_col: Optional[str],
    score_scale: int,
) -> tuple[cp_model.CpModel, dict[Any, Any], Any, list[dict[str, Any]]]:
    """
    Xây dựng CP-SAT model cho 1 lượt chọn (cho phép chọn đúng 1 laptop từ allowed_indices):
    - Ràng buộc Cứng (Hard): price, ram_gb, storage_gb, weight_kg, battery_minutes, gpu_discrete, gpu_keyword.
    - Ràng buộc Mềm (Soft): price, weight_kg, battery_minutes với slack variables và penalties.
    - Biến utility_var = total_relevance - sum(soft_penalties).
    """
    model = cp_model.CpModel()
    x = {i: model.NewBoolVar(f"x_{i}") for i in allowed_indices}
    model.Add(sum(x[i] for i in allowed_indices) == 1)

    soft_penalties = []
    soft_trackers = []

    for c_idx, c in enumerate(constraints_list):
        if isinstance(c, Constraint):
            c_field: str = str(c.field)
            c_op: str = str(c.operator)
            c_val: Any = c.value
            c_type: str = str(c.type)
        elif isinstance(c, dict):
            c_field = str(c.get("field", ""))
            c_op = str(c.get("operator", ""))
            c_val = c.get("value")
            c_type = str(c.get("type", "hard"))
        else:
            continue

        if c_val is None:
            continue

        is_hard = (c_type == "hard")

        # 1. Price constraint
        if c_field == "price":
            col = _find_column(candidates, ["price", "price_vnd", "laptop_price"])
            if col:
                target_val = int(float(c_val))
                price_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in allowed_indices]

                if is_hard:
                    if c_op == "<=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val)
                    elif c_op == ">=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) >= target_val)
                    elif c_op == "=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) == target_val)
                else:
                    slack_p = model.NewIntVar(0, 100_000, f"slack_price_{c_idx}")
                    if c_op == "<=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val + PRICE_VIOLATION_UNIT_VND * slack_p)
                    elif c_op == ">=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) + PRICE_VIOLATION_UNIT_VND * slack_p >= target_val)
                    elif c_op == "=":
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val + PRICE_VIOLATION_UNIT_VND * slack_p)
                        model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) + PRICE_VIOLATION_UNIT_VND * slack_p >= target_val)

                    soft_penalties.append(slack_p * PENALTY_PER_UNIT["price"])
                    soft_trackers.append({"type": "price", "var": slack_p, "col": col, "op": c_op, "target": target_val})

        # 2. RAM constraint
        elif c_field == "ram_gb":
            col = _find_column(candidates, ["ram_gb", "ram", "ram_capacity", "ram_size"])
            if col:
                target_val = int(float(c_val))
                ram_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in allowed_indices]
                if is_hard:
                    if c_op == ">=":
                        model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val)
                    elif c_op == "=":
                        model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) == target_val)

        # 3. Storage constraint
        elif c_field == "storage_gb":
            col = _find_column(candidates, ["storage_gb", "storage", "ssd_gb", "ssd_capacity", "storage_capacity"])
            if col:
                target_val = int(float(c_val))
                storage_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in allowed_indices]
                if is_hard:
                    if c_op == ">=":
                        model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val)
                    elif c_op == "=":
                        model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) == target_val)

        # 4. Weight constraint
        elif c_field == "weight_kg":
            col = _find_column(candidates, ["laptop_weight", "weight_kg", "weight"])
            if col:
                target_grams = int(float(c_val) * 1000)
                weight_grams = [int(float(candidates.loc[i, col]) * 1000) if pd.notna(candidates.loc[i, col]) else 0 for i in allowed_indices]

                if is_hard:
                    if c_op == "<=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_grams)
                    elif c_op == ">=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(allowed_indices)) >= target_grams)
                    elif c_op == "=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(allowed_indices)) == target_grams)
                else:
                    slack_w = model.NewIntVar(0, 10_000, f"slack_weight_{c_idx}")
                    if c_op == "<=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_grams + WEIGHT_VIOLATION_UNIT_GRAMS * slack_w)
                    elif c_op == ">=":
                        model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(allowed_indices)) + WEIGHT_VIOLATION_UNIT_GRAMS * slack_w >= target_grams)

                    soft_penalties.append(slack_w * PENALTY_PER_UNIT["weight"])
                    soft_trackers.append({"type": "weight_kg", "var": slack_w, "col": col, "op": c_op, "target": float(c_val)})

        # 5. Battery constraint
        elif c_field == "battery_minutes":
            col = _find_column(candidates, ["office_battery_minutes_final", "office_battery_result_minutes", "battery_minutes"])
            if col:
                target_val = int(float(c_val))
                battery_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in allowed_indices]

                if is_hard:
                    if c_op == ">=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val)
                    elif c_op == "=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) == target_val)
                else:
                    slack_b = model.NewIntVar(0, 10_000, f"slack_battery_{c_idx}")
                    if c_op == ">=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) + BATTERY_VIOLATION_UNIT_MINUTES * slack_b >= target_val)
                    elif c_op == "<=":
                        model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(allowed_indices)) <= target_val + BATTERY_VIOLATION_UNIT_MINUTES * slack_b)

                    soft_penalties.append(slack_b * PENALTY_PER_UNIT["battery"])
                    soft_trackers.append({"type": "battery_minutes", "var": slack_b, "col": col, "op": c_op, "target": target_val})

        # 6. GPU discrete
        elif c_field == "gpu_discrete":
            if is_hard:
                disc_col = _find_column(candidates, ["gpu_discrete", "is_discrete_gpu"])
                gpu_name_col = _find_column(candidates, ["gpu_name", "gpu"])
                req_disc = bool(c_val)

                disc_flags = []
                for i in allowed_indices:
                    if disc_col and pd.notna(candidates.loc[i, disc_col]):
                        is_d = bool(candidates.loc[i, disc_col])
                    elif gpu_name_col:
                        is_d = _is_discrete_gpu(candidates.loc[i, gpu_name_col])
                    else:
                        is_d = False
                    disc_flags.append(1 if (is_d == req_disc) else 0)

                model.Add(sum(disc_flags[idx] * x[i] for idx, i in enumerate(allowed_indices)) == 1)

        # 7. GPU keyword
        elif c_field == "gpu_keyword":
            if is_hard:
                gpu_name_col = _find_column(candidates, ["gpu_name", "gpu"])
                kw = str(c_val).strip()
                if gpu_name_col and kw:
                    kw_flags = [
                        1 if _matches_gpu_keyword(candidates.loc[i, gpu_name_col], kw) else 0
                        for i in allowed_indices
                    ]
                    model.Add(sum(kw_flags[idx] * x[i] for idx, i in enumerate(allowed_indices)) == 1)

    if score_col:
        scaled_scores = [
            int(float(candidates.loc[i, score_col]) * score_scale) if pd.notna(candidates.loc[i, score_col]) else 0
            for i in allowed_indices
        ]
        total_relevance = sum(scaled_scores[idx] * x[i] for idx, i in enumerate(allowed_indices))
    else:
        total_relevance = 0

    utility_var = model.NewIntVar(-100_000_000, 100_000_000, "utility")
    if soft_penalties:
        model.Add(utility_var == total_relevance - sum(soft_penalties))
    else:
        model.Add(utility_var == total_relevance)

    return model, x, utility_var, soft_trackers


def solve_top3(
    candidates: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any]],
    score_scale: int = 10000,
) -> dict[str, Any]:
    """
    Tối ưu hóa lựa chọn Top-3 Laptop đa mục tiêu:
    - Top 1 (best_match): Maximize utility (scaled_relevance - soft_penalties).
    - Top 2 (budget_alternative): Loại Top 1, utility >= 90% utility Top 1, Minimize price.
    - Top 3 (performance_alternative): Loại Top 1 và Top 2, utility >= 80% utility Top 1, Maximize performance_score.

    Nếu không đủ ứng viên, trả về 1 hoặc 2 recommendations an toàn, không crash.

    Returns:
        dict định dạng kết quả:
        {
            "status": "OPTIMAL" | "FEASIBLE" | "INFEASIBLE",
            "is_feasible": bool,
            "recommendations": [
                {
                    "rank": 1,
                    "type": "best_match",
                    "laptop_id": ...,
                    "relevance_score": ...,
                    "utility_score": ...
                },
                ...
            ]
        }
    """
    if candidates is None or candidates.empty:
        return {
            "status": "INFEASIBLE",
            "is_feasible": False,
            "laptop_id": None,
            "relevance_score": None,
            "hard_violations": ["Danh sách ứng viên rỗng."],
            "has_soft_violation": False,
            "soft_violations": [],
            "recommendations": [],
        }

    # Chuẩn hóa requirements thành danh sách constraints
    constraints_list: list[Union[Constraint, dict[str, Any]]] = []
    if isinstance(requirements, RequirementSet):
        constraints_list = list(requirements.constraints)
    elif isinstance(requirements, dict):
        raw_constraints = requirements.get("constraints")
        if isinstance(raw_constraints, list):
            constraints_list = raw_constraints
        else:
            from app.nlp.nl2constraint import convert_legacy_regex_to_requirement_set
            req_set = convert_legacy_regex_to_requirement_set(requirements)
            if isinstance(req_set, dict):
                req_constraints = req_set.get("constraints")
                if isinstance(req_constraints, list):
                    constraints_list = req_constraints

    score_col = _find_column(candidates, ["final_relevance_score", "relevance_score", "AI_Score"])
    price_col = _find_column(candidates, ["price", "price_vnd", "laptop_price"])
    perf_scores = compute_performance_scores(candidates)

    recommendations: list[dict[str, Any]] = []
    chosen_indices: set[Any] = set()

    # --------------------------------------------------------------------------
    # Rank 1: best_match (Maximize utility)
    # --------------------------------------------------------------------------
    allowed_1 = list(candidates.index)
    model_1, x_1, utility_1, trackers_1 = _build_optimization_step_model(
        candidates, allowed_1, constraints_list, score_col, score_scale
    )
    model_1.Maximize(utility_1)

    solver_1: Any = cp_model.CpSolver()
    status_1 = solver_1.Solve(model_1)

    if status_1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "INFEASIBLE",
            "is_feasible": False,
            "laptop_id": None,
            "relevance_score": None,
            "hard_violations": ["No laptop satisfies all hard constraints."],
            "has_soft_violation": False,
            "soft_violations": [],
            "recommendations": [],
        }

    chosen_idx_1 = None
    for i in allowed_1:
        if solver_1.BooleanValue(x_1[i]):
            chosen_idx_1 = i
            break

    if chosen_idx_1 is None:
        return {
            "status": "INFEASIBLE",
            "is_feasible": False,
            "laptop_id": None,
            "relevance_score": None,
            "hard_violations": ["No laptop satisfies all hard constraints."],
            "has_soft_violation": False,
            "soft_violations": [],
            "recommendations": [],
        }

    chosen_indices.add(chosen_idx_1)
    u1_val = int(solver_1.Value(utility_1))
    row_1 = candidates.loc[chosen_idx_1]
    soft_v_1 = _extract_soft_violations(row_1, trackers_1)
    relevance_1 = float(row_1[score_col]) if (score_col and pd.notna(row_1.get(score_col))) else None

    recommendations.append({
        "rank": 1,
        "type": "best_match",
        "laptop_id": row_1.get("laptop_model_id", row_1.get("laptop_id", chosen_idx_1)),
        "relevance_score": relevance_1,
        "utility_score": round(u1_val / score_scale, 4),
        "price": float(row_1[price_col]) if price_col and pd.notna(row_1.get(price_col)) else None,
        "performance_score": float(perf_scores.loc[chosen_idx_1]) if chosen_idx_1 in perf_scores.index else 0.0,
        "has_soft_violation": len(soft_v_1) > 0,
        "soft_violations": soft_v_1,
    })

    # --------------------------------------------------------------------------
    # Rank 2: budget_alternative (utility >= 90% u1, Minimize price)
    # --------------------------------------------------------------------------
    allowed_2 = [i for i in candidates.index if i not in chosen_indices]
    if allowed_2:
        model_2, x_2, utility_2, trackers_2 = _build_optimization_step_model(
            candidates, allowed_2, constraints_list, score_col, score_scale
        )
        min_u2 = int(np.floor(0.90 * u1_val))
        model_2.Add(utility_2 >= min_u2)

        if price_col:
            price_vals_2 = [
                int(float(candidates.loc[i, price_col])) if pd.notna(candidates.loc[i, price_col]) else 0
                for i in allowed_2
            ]
            total_price_2 = sum(price_vals_2[idx] * x_2[i] for idx, i in enumerate(allowed_2))
            model_2.Minimize(total_price_2)
        else:
            model_2.Maximize(utility_2)

        solver_2: Any = cp_model.CpSolver()
        status_2 = solver_2.Solve(model_2)

        if status_2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen_idx_2 = None
            for i in allowed_2:
                if solver_2.BooleanValue(x_2[i]):
                    chosen_idx_2 = i
                    break

            if chosen_idx_2 is not None:
                chosen_indices.add(chosen_idx_2)
                u2_val = int(solver_2.Value(utility_2))
                row_2 = candidates.loc[chosen_idx_2]
                soft_v_2 = _extract_soft_violations(row_2, trackers_2)
                relevance_2 = float(row_2[score_col]) if (score_col and pd.notna(row_2.get(score_col))) else None

                recommendations.append({
                    "rank": 2,
                    "type": "budget_alternative",
                    "laptop_id": row_2.get("laptop_model_id", row_2.get("laptop_id", chosen_idx_2)),
                    "relevance_score": relevance_2,
                    "utility_score": round(u2_val / score_scale, 4),
                    "price": float(row_2[price_col]) if price_col and pd.notna(row_2.get(price_col)) else None,
                    "performance_score": float(perf_scores.loc[chosen_idx_2]) if chosen_idx_2 in perf_scores.index else 0.0,
                    "has_soft_violation": len(soft_v_2) > 0,
                    "soft_violations": soft_v_2,
                })

    # --------------------------------------------------------------------------
    # Rank 3: performance_alternative (utility >= 80% u1, Maximize performance_score)
    # --------------------------------------------------------------------------
    allowed_3 = [i for i in candidates.index if i not in chosen_indices]
    if allowed_3:
        model_3, x_3, utility_3, trackers_3 = _build_optimization_step_model(
            candidates, allowed_3, constraints_list, score_col, score_scale
        )
        min_u3 = int(np.floor(0.80 * u1_val))
        model_3.Add(utility_3 >= min_u3)

        perf_vals_3 = [
            int(float(perf_scores.loc[i]) * score_scale) if i in perf_scores.index else 0
            for i in allowed_3
        ]
        total_perf_3 = sum(perf_vals_3[idx] * x_3[i] for idx, i in enumerate(allowed_3))
        model_3.Maximize(total_perf_3)

        solver_3: Any = cp_model.CpSolver()
        status_3 = solver_3.Solve(model_3)

        if status_3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen_idx_3 = None
            for i in allowed_3:
                if solver_3.BooleanValue(x_3[i]):
                    chosen_idx_3 = i
                    break

            if chosen_idx_3 is not None:
                chosen_indices.add(chosen_idx_3)
                u3_val = int(solver_3.Value(utility_3))
                row_3 = candidates.loc[chosen_idx_3]
                soft_v_3 = _extract_soft_violations(row_3, trackers_3)
                relevance_3 = float(row_3[score_col]) if (score_col and pd.notna(row_3.get(score_col))) else None

                recommendations.append({
                    "rank": 3,
                    "type": "performance_alternative",
                    "laptop_id": row_3.get("laptop_model_id", row_3.get("laptop_id", chosen_idx_3)),
                    "relevance_score": relevance_3,
                    "utility_score": round(u3_val / score_scale, 4),
                    "price": float(row_3[price_col]) if price_col and pd.notna(row_3.get(price_col)) else None,
                    "performance_score": float(perf_scores.loc[chosen_idx_3]) if chosen_idx_3 in perf_scores.index else 0.0,
                    "has_soft_violation": len(soft_v_3) > 0,
                    "soft_violations": soft_v_3,
                })

    overall_status = "OPTIMAL" if status_1 == cp_model.OPTIMAL else "FEASIBLE"
    top1 = recommendations[0] if recommendations else None
    return {
        "status": overall_status,
        "is_feasible": True,
        "laptop_id": top1["laptop_id"] if top1 else None,
        "relevance_score": top1["relevance_score"] if top1 else None,
        "hard_violations": [],
        "has_soft_violation": top1["has_soft_violation"] if top1 else False,
        "soft_violations": top1["soft_violations"] if top1 else [],
        "recommendations": recommendations,
    }


def solve_nearest_alternative(
    candidates: pd.DataFrame,
    requirements: Union[RequirementSet, dict[str, Any]],
    score_scale: int = 10000,
) -> dict[str, Any]:
    """
    Tìm 1 laptop gần nhất (Nearest Alternative) từ dataset khi tập ràng buộc cứng vô nghiệm:
    - Ràng buộc: Chọn chính xác 1 laptop.
    - Mọi ràng buộc cứng/mềm được chuyển thành biến vi phạm bù (slack).
    - Hàm mục tiêu: Minimize (M * total_weighted_violation - scaled_relevance).
      -> Ưu tiên 1: Tổng vi phạm nhỏ nhất.
      -> Ưu tiên 2 (khi bằng nhau): Relevance score cao nhất.

    Returns:
        dict định dạng kết quả:
        {
            "status": "RELAXED",
            "is_feasible": False,
            "laptop_id": ...,
            "relevance_score": ...,
            "hard_violations": [...],
            "violations": [...],
            "recommendations": [
                {
                    "rank": 1,
                    "type": "nearest_alternative",
                    "laptop_id": ...,
                    "violations": [...]
                }
            ]
        }
    """
    if candidates is None or candidates.empty:
        return {
            "status": "INFEASIBLE",
            "is_feasible": False,
            "laptop_id": None,
            "relevance_score": None,
            "hard_violations": ["Cơ sở dữ liệu ứng viên rỗng."],
            "has_soft_violation": False,
            "soft_violations": [],
            "violations": [],
            "recommendations": [],
        }

    constraints_list: list[Union[Constraint, dict[str, Any]]] = []
    if isinstance(requirements, RequirementSet):
        constraints_list = list(requirements.constraints)
    elif isinstance(requirements, dict):
        raw_constraints = requirements.get("constraints")
        if isinstance(raw_constraints, list):
            constraints_list = raw_constraints
        else:
            from app.nlp.nl2constraint import convert_legacy_regex_to_requirement_set
            req_set = convert_legacy_regex_to_requirement_set(requirements)
            if isinstance(req_set, dict):
                req_constraints = req_set.get("constraints")
                if isinstance(req_constraints, list):
                    constraints_list = req_constraints

    model = cp_model.CpModel()
    indices = list(candidates.index)
    x = {i: model.NewBoolVar(f"x_{i}") for i in indices}
    model.Add(sum(x[i] for i in indices) == 1)

    weighted_penalties = []
    trackers = []

    for c_idx, c in enumerate(constraints_list):
        if isinstance(c, Constraint):
            c_field: str = str(c.field)
            c_op: str = str(c.operator)
            c_val: Any = c.value
            c_type: str = str(c.type).lower() if hasattr(c, "type") and c.type else "hard"
        elif isinstance(c, dict):
            c_field = str(c.get("field", ""))
            c_op = str(c.get("operator", c.get("op", "")))
            c_val = c.get("value")
            c_type = str(c.get("type", "hard")).lower()
        else:
            continue

        if c_type != "hard":
            continue

        if c_val is None:
            continue

        # 1. Price
        if c_field == "price":
            col = _find_column(candidates, ["price", "price_vnd", "laptop_price"])
            if col:
                target_val = int(float(c_val))
                price_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                slack_p = model.NewIntVar(0, 100_000, f"slack_price_fb_{c_idx}")
                if c_op == "<=":
                    model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + PRICE_VIOLATION_UNIT_VND * slack_p)
                elif c_op == ">=":
                    model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) + PRICE_VIOLATION_UNIT_VND * slack_p >= target_val)
                elif c_op == "=":
                    model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + PRICE_VIOLATION_UNIT_VND * slack_p)
                    model.Add(sum(price_vals[idx] * x[i] for idx, i in enumerate(indices)) + PRICE_VIOLATION_UNIT_VND * slack_p >= target_val)

                weighted_penalties.append(slack_p * 20)
                trackers.append({"type": "price", "var": slack_p, "col": col, "op": c_op, "target": target_val})

        # 2. RAM
        elif c_field == "ram_gb":
            col = _find_column(candidates, ["ram_gb", "ram", "ram_capacity", "ram_size"])
            if col:
                target_val = int(float(c_val))
                ram_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                slack_ram = model.NewIntVar(0, 1000, f"slack_ram_fb_{c_idx}")
                if c_op == ">=":
                    model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) + slack_ram >= target_val)
                elif c_op == "<=":
                    model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + slack_ram)
                elif c_op == "=":
                    model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) + slack_ram >= target_val)
                    model.Add(sum(ram_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + slack_ram)

                weighted_penalties.append(slack_ram * 500)
                trackers.append({"type": "ram_gb", "var": slack_ram, "col": col, "op": c_op, "target": target_val})

        # 3. Storage
        elif c_field == "storage_gb":
            col = _find_column(candidates, ["storage_gb", "storage", "ssd_gb", "ssd_capacity", "storage_capacity"])
            if col:
                target_val = int(float(c_val))
                storage_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                slack_st = model.NewIntVar(0, 10_000, f"slack_st_fb_{c_idx}")
                if c_op == ">=":
                    model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) + 128 * slack_st >= target_val)
                elif c_op == "<=":
                    model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + 128 * slack_st)
                elif c_op == "=":
                    model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) + 128 * slack_st >= target_val)
                    model.Add(sum(storage_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + 128 * slack_st)

                weighted_penalties.append(slack_st * 200)
                trackers.append({"type": "storage_gb", "var": slack_st, "col": col, "op": c_op, "target": target_val})

        # 4. Weight
        elif c_field == "weight_kg":
            col = _find_column(candidates, ["laptop_weight", "weight_kg", "weight"])
            if col:
                target_grams = int(float(c_val) * 1000)
                weight_grams = [int(float(candidates.loc[i, col]) * 1000) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                slack_w = model.NewIntVar(0, 10_000, f"slack_w_fb_{c_idx}")
                if c_op == "<=":
                    model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) <= target_grams + WEIGHT_VIOLATION_UNIT_GRAMS * slack_w)
                elif c_op == ">=":
                    model.Add(sum(weight_grams[idx] * x[i] for idx, i in enumerate(indices)) + WEIGHT_VIOLATION_UNIT_GRAMS * slack_w >= target_grams)

                weighted_penalties.append(slack_w * 100)
                trackers.append({"type": "weight_kg", "var": slack_w, "col": col, "op": c_op, "target": float(c_val)})

        # 5. Battery
        elif c_field == "battery_minutes":
            col = _find_column(candidates, ["office_battery_minutes_final", "office_battery_result_minutes", "battery_minutes"])
            if col:
                target_val = int(float(c_val))
                battery_vals = [int(float(candidates.loc[i, col])) if pd.notna(candidates.loc[i, col]) else 0 for i in indices]
                slack_b = model.NewIntVar(0, 10_000, f"slack_b_fb_{c_idx}")
                if c_op == ">=":
                    model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) + BATTERY_VIOLATION_UNIT_MINUTES * slack_b >= target_val)
                elif c_op == "<=":
                    model.Add(sum(battery_vals[idx] * x[i] for idx, i in enumerate(indices)) <= target_val + BATTERY_VIOLATION_UNIT_MINUTES * slack_b)

                weighted_penalties.append(slack_b * 20)
                trackers.append({"type": "battery_minutes", "var": slack_b, "col": col, "op": c_op, "target": target_val})

        # 6. GPU discrete
        elif c_field == "gpu_discrete":
            disc_col = _find_column(candidates, ["gpu_discrete", "is_discrete_gpu"])
            gpu_name_col = _find_column(candidates, ["gpu_name", "gpu"])
            req_disc = bool(c_val)

            mismatch_flags = []
            for i in indices:
                if disc_col and pd.notna(candidates.loc[i, disc_col]):
                    is_d = bool(candidates.loc[i, disc_col])
                elif gpu_name_col:
                    is_d = _is_discrete_gpu(candidates.loc[i, gpu_name_col])
                else:
                    is_d = False
                mismatch_flags.append(1 if (is_d != req_disc) else 0)

            slack_disc = model.NewIntVar(0, 1, f"slack_disc_fb_{c_idx}")
            model.Add(sum(mismatch_flags[idx] * x[i] for idx, i in enumerate(indices)) == slack_disc)
            weighted_penalties.append(slack_disc * 5000)
            trackers.append({"type": "gpu_discrete", "var": slack_disc, "col": gpu_name_col or disc_col, "op": "=", "target": req_disc})

        # 7. GPU keyword
        elif c_field == "gpu_keyword":
            gpu_name_col = _find_column(candidates, ["gpu_name", "gpu"])
            kw = str(c_val).strip()
            if kw:
                mismatch_flags = []
                for i in indices:
                    is_m = _matches_gpu_keyword(candidates.loc[i, gpu_name_col], kw) if gpu_name_col else False
                    mismatch_flags.append(0 if is_m else 1)

                slack_kw = model.NewIntVar(0, 1, f"slack_kw_fb_{c_idx}")
                model.Add(sum(mismatch_flags[idx] * x[i] for idx, i in enumerate(indices)) == slack_kw)
                weighted_penalties.append(slack_kw * 5000)
                trackers.append({"type": "gpu_keyword", "var": slack_kw, "col": gpu_name_col, "op": "=", "target": kw})

    score_col = _find_column(candidates, ["final_relevance_score", "relevance_score", "AI_Score"])
    if score_col:
        scaled_scores = [
            int(float(candidates.loc[i, score_col]) * score_scale) if pd.notna(candidates.loc[i, score_col]) else 0
            for i in indices
        ]
        total_relevance = sum(scaled_scores[idx] * x[i] for idx, i in enumerate(indices))
    else:
        total_relevance = 0

    # Multiplier M = 10,000 để penalty luôn áp đảo relevance, relevance chỉ làm tie-breaker
    M = 10_000
    if weighted_penalties:
        total_violation = sum(weighted_penalties)
        model.Minimize(M * total_violation - total_relevance)
    else:
        model.Maximize(total_relevance)

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
            relevance_val = float(chosen_row[score_col]) if (score_col and pd.notna(chosen_row.get(score_col))) else None

            # Trích xuất chi tiết vi phạm
            violations = []
            for tracker in trackers:
                t_type = tracker["type"]
                target = tracker["target"]
                col = tracker["col"]
                op = tracker["op"]
                actual_raw = chosen_row.get(col) if col else None

                if t_type == "price":
                    actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
                    target_num = float(target)
                    if (op == "<=" or op == "<") and actual > target_num:
                        diff = actual - target_num
                        violations.append({
                            "field": "price",
                            "target": target_num,
                            "actual": actual,
                            "violation": f"Vượt ngân sách {diff:,.0f}đ",
                        })
                    elif (op == ">=" or op == ">") and actual < target_num:
                        diff = target_num - actual
                        violations.append({
                            "field": "price",
                            "target": target_num,
                            "actual": actual,
                            "violation": f"Thấp hơn ngân sách tối thiểu {diff:,.0f}đ",
                        })
                elif t_type == "ram_gb":
                    actual = int(float(actual_raw)) if pd.notna(actual_raw) else 0
                    target_num = int(float(target))
                    if op == ">=" and actual < target_num:
                        diff = target_num - actual
                        violations.append({
                            "field": "ram_gb",
                            "target": target_num,
                            "actual": actual,
                            "violation": f"Thiếu {diff}GB RAM so với yêu cầu {target_num}GB",
                        })
                elif t_type == "storage_gb":
                    actual = int(float(actual_raw)) if pd.notna(actual_raw) else 0
                    target_num = int(float(target))
                    if op == ">=" and actual < target_num:
                        diff = target_num - actual
                        violations.append({
                            "field": "storage_gb",
                            "target": target_num,
                            "actual": actual,
                            "violation": f"Thiếu {diff}GB ổ cứng so với yêu cầu {target_num}GB",
                        })
                elif t_type == "weight_kg":
                    actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
                    target_num = float(target)
                    if op == "<=" and actual > target_num:
                        diff = actual - target_num
                        violations.append({
                            "field": "weight_kg",
                            "target": target_num,
                            "actual": actual,
                            "violation": f"Nặng hơn yêu cầu {diff:.2f}kg",
                        })
                elif t_type == "battery_minutes":
                    actual = float(actual_raw) if pd.notna(actual_raw) else 0.0
                    target_num = float(target)
                    if op == ">=" and actual < target_num:
                        diff = target_num - actual
                        violations.append({
                            "field": "battery_minutes",
                            "target": target_num,
                            "actual": actual,
                            "violation": f"Thời lượng pin thấp hơn yêu cầu {diff:.0f} phút",
                        })
                elif t_type == "gpu_discrete":
                    gpu_name = str(actual_raw) if pd.notna(actual_raw) else "Không xác định"
                    is_d = _is_discrete_gpu(actual_raw)
                    if bool(target) and not is_d:
                        violations.append({
                            "field": "gpu_discrete",
                            "target": True,
                            "actual": gpu_name,
                            "violation": f"Chỉ có GPU tích hợp ({gpu_name}), không có card rời",
                        })
                elif t_type == "gpu_keyword":
                    gpu_name = str(actual_raw) if pd.notna(actual_raw) else "Không xác định"
                    if not _matches_gpu_keyword(actual_raw, str(target)):
                        violations.append({
                            "field": "gpu_keyword",
                            "target": str(target),
                            "actual": gpu_name,
                            "violation": f"GPU ({gpu_name}) không khớp từ khóa yêu cầu '{target}'",
                        })

            hard_violations_text = [v["violation"] for v in violations]
            if not hard_violations_text:
                hard_violations_text = ["Không thỏa mãn toàn bộ ràng buộc cứng ban đầu."]

            return {
                "status": "RELAXED",
                "is_feasible": False,
                "laptop_id": laptop_id,
                "relevance_score": relevance_val,
                "hard_violations": hard_violations_text,
                "has_soft_violation": False,
                "soft_violations": [],
                "violations": violations,
                "recommendations": [
                    {
                        "rank": 1,
                        "type": "nearest_alternative",
                        "laptop_id": laptop_id,
                        "relevance_score": relevance_val,
                        "violations": violations,
                    }
                ],
            }

    return {
        "status": "INFEASIBLE",
        "is_feasible": False,
        "laptop_id": None,
        "relevance_score": None,
        "hard_violations": ["Không tìm được phương án gần nhất khả thi."],
        "has_soft_violation": False,
        "soft_violations": [],
        "violations": [],
        "recommendations": [],
    }

