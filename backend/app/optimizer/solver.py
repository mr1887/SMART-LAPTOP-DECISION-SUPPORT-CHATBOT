"""
Tầng 3 - Decision Support System (Gurobi/PuLP Optimizer).

Nhận: bộ ràng buộc (từ Tầng 1 NLP) + AI_Score cho từng laptop (từ Tầng 2
LightGBM), trả về 1 laptop tối ưu tuân thủ 100% ràng buộc cứng, hoặc giải
quyết bài toán vô nghiệm bằng phương pháp nới lỏng ràng buộc Soft Constraint (BIP Relaxation)
để luôn tìm ra nghiệm tiệm cận tối ưu nhất kèm giải thích chi tiết mức vi phạm.

Backend mặc định: Gurobi (nếu có license) hoặc PuLP + CBC solver (miễn phí).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import pulp

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# ---------- Xác thực Gurobi ----------
_GUROBI_ENV = None


def _create_gurobi_model(name: str) -> "gp.Model":
    """Tạo Gurobi Model - ưu tiên dò license tự động hoặc WLS params từ env."""
    global _GUROBI_ENV

    if _GUROBI_ENV is not None:
        return gp.Model(name, env=_GUROBI_ENV)

    try:
        model = gp.Model(name)
        return model
    except gp.GurobiError as e:
        license_related = "license" in str(e).lower() or "wls" in str(e).lower()
        if not license_related:
            raise

        access_id = os.environ.get("GUROBI_WLSACCESSID")
        secret = os.environ.get("GUROBI_WLSSECRET")
        license_id = os.environ.get("GUROBI_LICENSEID")

        if not (access_id and secret and license_id):
            raise RuntimeError("Thiếu thông tin license Gurobi WLS.") from e

        _GUROBI_ENV = gp.Env(params={
            "WLSAccessID": access_id,
            "WLSSecret": secret,
            "LicenseID": int(license_id),
        })
        return gp.Model(name, env=_GUROBI_ENV)


# Trọng số phạt khi phải nới lỏng ràng buộc (Soft Constraint)
# Chuẩn hóa theo thang AI_Score [0, 1]
RELAX_PENALTY = {
    "price": 1e-7,          # phạt theo VNĐ vượt ngân sách tối đa (10M VNĐ = -1.0)
    "min_price": 1e-7,      # phạt theo VNĐ thiếu so với ngân sách tối thiểu
    "weight": 0.3,          # phạt theo kg vượt cân nặng
    "battery": 0.002,       # phạt theo phút thiếu pin
    "tag": 0.6,             # phạt nếu không đạt nhãn nhu cầu (office, gaming,...)
    "gpu_discrete": 0.5,    # phạt nếu vi phạm yêu cầu card rời / tích hợp
    "gpu_keyword": 0.5,     # phạt nếu không khớp đúng dòng GPU mong muốn
}


def _drop_unusable_rows(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Loại các laptop bị thiếu (NaN) ở các cột bắt buộc tối thiểu (price, AI_Score)."""
    required_cols = ["AI_Score", "price"]
    clean_df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    return clean_df


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


def _solve_pulp(df: pd.DataFrame, constraints: dict) -> dict:
    """Giải bài toán bằng PuLP:
    - Vòng 1: Tìm nghiệm tối ưu thỏa 100% ràng buộc cứng.
    - Vòng 2: Nếu vô nghiệm, giải bài toán Soft BIP với biến bù (Slack variables)
              để tìm nghiệm gần tối ưu nhất và đo đạc mức vi phạm chính xác.
    """
    battery_col = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else "office_battery_result_minutes"
    indices = list(df.index)

    # -------------------------------------------------------------
    # VÒNG 1: THỬ GIẢI VỚI RÀNG BUỘC CỨNG (STRICT BIP)
    # -------------------------------------------------------------
    prob = pulp.LpProblem("laptop_selection_strict", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in indices}

    # Hàm mục tiêu: Maximize tổng AI_Score
    prob += pulp.lpSum(df.loc[i, "AI_Score"] * x[i] for i in indices)
    # Ràng buộc chỉ chọn duy nhất 1 laptop
    prob += pulp.lpSum(x[i] for i in indices) == 1

    # Ràng buộc giá
    if constraints.get("max_price") is not None:
        prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in indices) <= constraints["max_price"]
    if constraints.get("min_price") is not None:
        prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in indices) >= constraints["min_price"]

    # Ràng buộc cân nặng & pin
    if constraints.get("max_weight") is not None and "laptop_weight" in df.columns:
        prob += pulp.lpSum(df.loc[i, "laptop_weight"] * x[i] for i in indices) <= constraints["max_weight"]
    if constraints.get("min_battery") is not None and battery_col in df.columns:
        prob += pulp.lpSum(df.loc[i, battery_col] * x[i] for i in indices) >= constraints["min_battery"]

    # Ràng buộc Tags nhu cầu
    for tag in constraints.get("required_tags") or []:
        if tag in df.columns:
            tag_vals = [1 if bool(df.loc[i, tag]) else 0 for i in indices]
            prob += pulp.lpSum(tag_vals[i] * x[i] for i in indices) >= 1

    # Ràng buộc GPU (Card rời / Tích hợp)
    if "gpu_name" in df.columns:
        if constraints.get("require_discrete_gpu") is True:
            disc_vals = [1 if _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            prob += pulp.lpSum(disc_vals[i] * x[i] for i in indices) >= 1
        elif constraints.get("require_discrete_gpu") is False:
            integ_vals = [1 if not _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            prob += pulp.lpSum(integ_vals[i] * x[i] for i in indices) >= 1

        if constraints.get("gpu_keyword"):
            kw = constraints["gpu_keyword"]
            kw_vals = [1 if _matches_gpu_keyword(df.loc[i, "gpu_name"], kw) else 0 for i in indices]
            prob += pulp.lpSum(kw_vals[i] * x[i] for i in indices) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        chosen_indices = [i for i in indices if x[i].value() and x[i].value() > 0.5]
        if chosen_indices:
            chosen = chosen_indices[0]
            row = df.loc[chosen]
            score_val = float(row["AI_Score"])
            return {
                "laptop_id": int(row["laptop_model_id"]),
                "is_feasible": True,
                "is_relaxed": False,
                "ai_score": score_val,
                "violations": [],
                "violation_text": "",
                "explanation": (
                    f"Đề xuất tối ưu: '{row.get('laptop_name', row['laptop_model_id'])}'. "
                    f"Mẫu máy này thỏa mãn 100% các tiêu chí của bạn với điểm đánh giá tối ưu đạt "
                    f"{score_val * 10:.1f}/10 ⭐."
                ),
            }

    # -------------------------------------------------------------
    # VÒNG 2: VÔ NGHIỆM -> GIẢI SOFT BIP RELAXATION (NGHIỆM GẦN TỐI ƯU)
    # -------------------------------------------------------------
    relax_prob = pulp.LpProblem("laptop_selection_relaxed", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"rx_{i}", cat="Binary") for i in indices}

    slack_price = pulp.LpVariable("slack_price", lowBound=0)
    slack_min_price = pulp.LpVariable("slack_min_price", lowBound=0)
    slack_weight = pulp.LpVariable("slack_weight", lowBound=0)
    slack_battery = pulp.LpVariable("slack_battery", lowBound=0)
    slack_gpu_disc = pulp.LpVariable("slack_gpu_disc", lowBound=0)
    slack_gpu_kw = pulp.LpVariable("slack_gpu_kw", lowBound=0)
    slack_tags = {
        tag: pulp.LpVariable(f"slack_tag_{tag}", lowBound=0)
        for tag in constraints.get("required_tags") or []
        if tag in df.columns
    }

    # Hàm mục tiêu nới lỏng có phạt
    obj = pulp.lpSum(df.loc[i, "AI_Score"] * x[i] for i in indices)
    if constraints.get("max_price") is not None:
        obj -= RELAX_PENALTY["price"] * slack_price
    if constraints.get("min_price") is not None:
        obj -= RELAX_PENALTY["min_price"] * slack_min_price
    if constraints.get("max_weight") is not None and "laptop_weight" in df.columns:
        obj -= RELAX_PENALTY["weight"] * slack_weight
    if constraints.get("min_battery") is not None and battery_col in df.columns:
        obj -= RELAX_PENALTY["battery"] * slack_battery
    if constraints.get("require_discrete_gpu") is not None and "gpu_name" in df.columns:
        obj -= RELAX_PENALTY["gpu_discrete"] * slack_gpu_disc
    if constraints.get("gpu_keyword") and "gpu_name" in df.columns:
        obj -= RELAX_PENALTY["gpu_keyword"] * slack_gpu_kw
    for tag, s_var in slack_tags.items():
        obj -= RELAX_PENALTY["tag"] * s_var

    relax_prob += obj
    relax_prob += pulp.lpSum(x[i] for i in indices) == 1

    if constraints.get("max_price") is not None:
        relax_prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in indices) <= constraints["max_price"] + slack_price
    if constraints.get("min_price") is not None:
        relax_prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in indices) >= constraints["min_price"] - slack_min_price
    if constraints.get("max_weight") is not None and "laptop_weight" in df.columns:
        relax_prob += pulp.lpSum(df.loc[i, "laptop_weight"] * x[i] for i in indices) <= constraints["max_weight"] + slack_weight
    if constraints.get("min_battery") is not None and battery_col in df.columns:
        relax_prob += pulp.lpSum(df.loc[i, battery_col] * x[i] for i in indices) >= constraints["min_battery"] - slack_battery

    for tag, s_var in slack_tags.items():
        tag_vals = [1 if bool(df.loc[i, tag]) else 0 for i in indices]
        relax_prob += pulp.lpSum(tag_vals[i] * x[i] for i in indices) + s_var >= 1

    if "gpu_name" in df.columns:
        if constraints.get("require_discrete_gpu") is True:
            disc_vals = [1 if _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            relax_prob += pulp.lpSum(disc_vals[i] * x[i] for i in indices) + slack_gpu_disc >= 1
        elif constraints.get("require_discrete_gpu") is False:
            integ_vals = [1 if not _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            relax_prob += pulp.lpSum(integ_vals[i] * x[i] for i in indices) + slack_gpu_disc >= 1

        if constraints.get("gpu_keyword"):
            kw = constraints["gpu_keyword"]
            kw_vals = [1 if _matches_gpu_keyword(df.loc[i, "gpu_name"], kw) else 0 for i in indices]
            relax_prob += pulp.lpSum(kw_vals[i] * x[i] for i in indices) + slack_gpu_kw >= 1

    relax_prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[relax_prob.status] == "Optimal":
        chosen_indices = [i for i in indices if x[i].value() and x[i].value() > 0.5]
        if chosen_indices:
            chosen = chosen_indices[0]
            row = df.loc[chosen]
            score_val = float(row["AI_Score"])

            violations = []
            if constraints.get("max_price") is not None and slack_price.value() and slack_price.value() > 100:
                violations.append(f"vượt ngân sách tối đa {slack_price.value():,.0f}đ")
            if constraints.get("min_price") is not None and slack_min_price.value() and slack_min_price.value() > 100:
                violations.append(f"thấp hơn ngân sách tối thiểu {slack_min_price.value():,.0f}đ")
            if constraints.get("max_weight") is not None and slack_weight.value() and slack_weight.value() > 0.05:
                violations.append(f"nặng hơn {slack_weight.value():.2f}kg so với yêu cầu")
            if constraints.get("min_battery") is not None and slack_battery.value() and slack_battery.value() > 5:
                violations.append(f"thiếu {slack_battery.value():.0f} phút pin so với yêu cầu")
            if constraints.get("require_discrete_gpu") is True and slack_gpu_disc.value() and slack_gpu_disc.value() > 0.5:
                violations.append("chưa trang bị card đồ họa rời (dùng card tích hợp)")
            elif constraints.get("require_discrete_gpu") is False and slack_gpu_disc.value() and slack_gpu_disc.value() > 0.5:
                violations.append("trang bị card đồ họa rời thay vì card tích hợp")
            if constraints.get("gpu_keyword") and slack_gpu_kw.value() and slack_gpu_kw.value() > 0.5:
                violations.append(f"không trang bị đúng dòng GPU '{constraints['gpu_keyword']}'")
            for tag, s_var in slack_tags.items():
                if s_var.value() and s_var.value() > 0.5:
                    tag_name_vi = {
                        "is_gaming_friendly": "Gaming",
                        "is_office_friendly": "Văn phòng",
                        "is_graphic_friendly": "Đồ họa",
                        "is_programming_friendly": "Lập trình",
                    }.get(tag, tag)
                    violations.append(f"chưa đạt chuẩn tối ưu riêng cho nhu cầu {tag_name_vi}")

            violation_text = ", ".join(violations) if violations else "chưa thỏa mãn đồng thời tất cả các tiêu chí"

            return {
                "laptop_id": int(row["laptop_model_id"]),
                "is_feasible": False,
                "is_relaxed": True,
                "ai_score": score_val,
                "violations": violations,
                "violation_text": violation_text,
                "explanation": (
                    f"Gợi ý gần đạt chuẩn nhất: '{row.get('laptop_name', row['laptop_model_id'])}'. "
                    f"Hiện chưa có mẫu máy thỏa mãn tuyệt đối 100% các tiêu chí ({violation_text}), "
                    f"nhưng đây là lựa chọn cân đối và phù hợp nhất với điểm đánh giá đạt {score_val * 10:.1f}/10 ⭐."
                ),
            }

    # Trường hợp dự phòng cực đoan: Lấy laptop có AI_Score cao nhất
    best_row = df.sort_values(by="AI_Score", ascending=False).iloc[0]
    return {
        "laptop_id": int(best_row["laptop_model_id"]),
        "is_feasible": False,
        "is_relaxed": True,
        "ai_score": float(best_row["AI_Score"]),
        "violations": ["ràng buộc quá khắt khe"],
        "violation_text": "ràng buộc quá khắt khe",
        "explanation": f"Gợi ý mẫu laptop nổi bật nhất: '{best_row.get('laptop_name', best_row['laptop_model_id'])}'.",
    }


def _solve_gurobi(df: pd.DataFrame, constraints: dict) -> dict:
    """Giải bài toán bằng Gurobi:
    - Vòng 1: Strict BIP.
    - Vòng 2: Soft BIP Relaxation (Nghiệm gần tối ưu).
    """
    battery_col = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else "office_battery_result_minutes"
    indices = list(df.index)

    # -------------------------------------------------------------
    # VÒNG 1: THỬ GIẢI RÀNG BUỘC CỨNG (STRICT)
    # -------------------------------------------------------------
    model = _create_gurobi_model("laptop_selection_strict")
    model.setParam("OutputFlag", 0)

    x = model.addVars(indices, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(df.loc[i, "AI_Score"] * x[i] for i in indices), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(x[i] for i in indices) == 1, name="choose_one")

    if constraints.get("max_price") is not None:
        model.addConstr(gp.quicksum(df.loc[i, "price"] * x[i] for i in indices) <= constraints["max_price"], name="budget_max")
    if constraints.get("min_price") is not None:
        model.addConstr(gp.quicksum(df.loc[i, "price"] * x[i] for i in indices) >= constraints["min_price"], name="budget_min")
    if constraints.get("max_weight") is not None and "laptop_weight" in df.columns:
        model.addConstr(gp.quicksum(df.loc[i, "laptop_weight"] * x[i] for i in indices) <= constraints["max_weight"], name="weight")
    if constraints.get("min_battery") is not None and battery_col in df.columns:
        model.addConstr(gp.quicksum(df.loc[i, battery_col] * x[i] for i in indices) >= constraints["min_battery"], name="battery")

    for tag in constraints.get("required_tags") or []:
        if tag in df.columns:
            tag_vals = [1 if bool(df.loc[i, tag]) else 0 for i in indices]
            model.addConstr(gp.quicksum(tag_vals[i] * x[i] for i in indices) >= 1, name=f"tag_{tag}")

    if "gpu_name" in df.columns:
        if constraints.get("require_discrete_gpu") is True:
            disc_vals = [1 if _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            model.addConstr(gp.quicksum(disc_vals[i] * x[i] for i in indices) >= 1, name="gpu_disc")
        elif constraints.get("require_discrete_gpu") is False:
            integ_vals = [1 if not _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            model.addConstr(gp.quicksum(integ_vals[i] * x[i] for i in indices) >= 1, name="gpu_integ")

        if constraints.get("gpu_keyword"):
            kw = constraints["gpu_keyword"]
            kw_vals = [1 if _matches_gpu_keyword(df.loc[i, "gpu_name"], kw) else 0 for i in indices]
            model.addConstr(gp.quicksum(kw_vals[i] * x[i] for i in indices) >= 1, name="gpu_kw")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        chosen_indices = [i for i in indices if x[i].X > 0.5]
        if chosen_indices:
            chosen = chosen_indices[0]
            row = df.loc[chosen]
            score_val = float(row["AI_Score"])
            return {
                "laptop_id": int(row["laptop_model_id"]),
                "is_feasible": True,
                "is_relaxed": False,
                "ai_score": score_val,
                "violations": [],
                "violation_text": "",
                "explanation": (
                    f"Đề xuất tối ưu: '{row.get('laptop_name', row['laptop_model_id'])}'. "
                    f"Mẫu máy này thỏa mãn 100% các tiêu chí của bạn với điểm đánh giá tối ưu đạt "
                    f"{score_val * 10:.1f}/10 ⭐."
                ),
            }

    # -------------------------------------------------------------
    # VÒNG 2: VÔ NGHIỆM -> GIẢI SOFT BIP RELAXATION BẰNG GUROBI
    # -------------------------------------------------------------
    relax_model = _create_gurobi_model("laptop_selection_relaxed")
    relax_model.setParam("OutputFlag", 0)

    x = relax_model.addVars(indices, vtype=GRB.BINARY, name="x")
    slack_price = relax_model.addVar(lb=0, name="slack_price")
    slack_min_price = relax_model.addVar(lb=0, name="slack_min_price")
    slack_weight = relax_model.addVar(lb=0, name="slack_weight")
    slack_battery = relax_model.addVar(lb=0, name="slack_battery")
    slack_gpu_disc = relax_model.addVar(lb=0, name="slack_gpu_disc")
    slack_gpu_kw = relax_model.addVar(lb=0, name="slack_gpu_kw")
    slack_tags = {
        tag: relax_model.addVar(lb=0, name=f"slack_tag_{tag}")
        for tag in constraints.get("required_tags") or []
        if tag in df.columns
    }

    obj_expr = gp.quicksum(df.loc[i, "AI_Score"] * x[i] for i in indices)
    if constraints.get("max_price") is not None:
        obj_expr -= RELAX_PENALTY["price"] * slack_price
    if constraints.get("min_price") is not None:
        obj_expr -= RELAX_PENALTY["min_price"] * slack_min_price
    if constraints.get("max_weight") is not None and "laptop_weight" in df.columns:
        obj_expr -= RELAX_PENALTY["weight"] * slack_weight
    if constraints.get("min_battery") is not None and battery_col in df.columns:
        obj_expr -= RELAX_PENALTY["battery"] * slack_battery
    if constraints.get("require_discrete_gpu") is not None and "gpu_name" in df.columns:
        obj_expr -= RELAX_PENALTY["gpu_discrete"] * slack_gpu_disc
    if constraints.get("gpu_keyword") and "gpu_name" in df.columns:
        obj_expr -= RELAX_PENALTY["gpu_keyword"] * slack_gpu_kw
    for tag, s_var in slack_tags.items():
        obj_expr -= RELAX_PENALTY["tag"] * s_var

    relax_model.setObjective(obj_expr, GRB.MAXIMIZE)
    relax_model.addConstr(gp.quicksum(x[i] for i in indices) == 1)

    if constraints.get("max_price") is not None:
        relax_model.addConstr(gp.quicksum(df.loc[i, "price"] * x[i] for i in indices) <= constraints["max_price"] + slack_price)
    if constraints.get("min_price") is not None:
        relax_model.addConstr(gp.quicksum(df.loc[i, "price"] * x[i] for i in indices) >= constraints["min_price"] - slack_min_price)
    if constraints.get("max_weight") is not None and "laptop_weight" in df.columns:
        relax_model.addConstr(gp.quicksum(df.loc[i, "laptop_weight"] * x[i] for i in indices) <= constraints["max_weight"] + slack_weight)
    if constraints.get("min_battery") is not None and battery_col in df.columns:
        relax_model.addConstr(gp.quicksum(df.loc[i, battery_col] * x[i] for i in indices) >= constraints["min_battery"] - slack_battery)

    for tag, s_var in slack_tags.items():
        tag_vals = [1 if bool(df.loc[i, tag]) else 0 for i in indices]
        relax_model.addConstr(gp.quicksum(tag_vals[i] * x[i] for i in indices) + s_var >= 1)

    if "gpu_name" in df.columns:
        if constraints.get("require_discrete_gpu") is True:
            disc_vals = [1 if _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            relax_model.addConstr(gp.quicksum(disc_vals[i] * x[i] for i in indices) + slack_gpu_disc >= 1)
        elif constraints.get("require_discrete_gpu") is False:
            integ_vals = [1 if not _is_discrete_gpu(df.loc[i, "gpu_name"]) else 0 for i in indices]
            relax_model.addConstr(gp.quicksum(integ_vals[i] * x[i] for i in indices) + slack_gpu_disc >= 1)

        if constraints.get("gpu_keyword"):
            kw = constraints["gpu_keyword"]
            kw_vals = [1 if _matches_gpu_keyword(df.loc[i, "gpu_name"], kw) else 0 for i in indices]
            relax_model.addConstr(gp.quicksum(kw_vals[i] * x[i] for i in indices) + slack_gpu_kw >= 1)

    relax_model.optimize()

    if relax_model.status == GRB.OPTIMAL:
        chosen_indices = [i for i in indices if x[i].X > 0.5]
        if chosen_indices:
            chosen = chosen_indices[0]
            row = df.loc[chosen]
            score_val = float(row["AI_Score"])

            violations = []
            if constraints.get("max_price") is not None and slack_price.X > 100:
                violations.append(f"vượt ngân sách tối đa {slack_price.X:,.0f}đ")
            if constraints.get("min_price") is not None and slack_min_price.X > 100:
                violations.append(f"thấp hơn ngân sách tối thiểu {slack_min_price.X:,.0f}đ")
            if constraints.get("max_weight") is not None and slack_weight.X > 0.05:
                violations.append(f"nặng hơn {slack_weight.X:.2f}kg so với yêu cầu")
            if constraints.get("min_battery") is not None and slack_battery.X > 5:
                violations.append(f"thiếu {slack_battery.X:.0f} phút pin so với yêu cầu")
            if constraints.get("require_discrete_gpu") is True and slack_gpu_disc.X > 0.5:
                violations.append("chưa trang bị card đồ họa rời (dùng card tích hợp)")
            elif constraints.get("require_discrete_gpu") is False and slack_gpu_disc.X > 0.5:
                violations.append("trang bị card đồ họa rời thay vì card tích hợp")
            if constraints.get("gpu_keyword") and slack_gpu_kw.X > 0.5:
                violations.append(f"không trang bị đúng dòng GPU '{constraints['gpu_keyword']}'")
            for tag, s_var in slack_tags.items():
                if s_var.X > 0.5:
                    tag_name_vi = {
                        "is_gaming_friendly": "Gaming",
                        "is_office_friendly": "Văn phòng",
                        "is_graphic_friendly": "Đồ họa",
                        "is_programming_friendly": "Lập trình",
                    }.get(tag, tag)
                    violations.append(f"chưa đạt chuẩn tối ưu riêng cho nhu cầu {tag_name_vi}")

            violation_text = ", ".join(violations) if violations else "chưa thỏa mãn đồng thời tất cả các tiêu chí"

            return {
                "laptop_id": int(row["laptop_model_id"]),
                "is_feasible": False,
                "is_relaxed": True,
                "ai_score": score_val,
                "violations": violations,
                "violation_text": violation_text,
                "explanation": (
                    f"Gợi ý gần đạt chuẩn nhất: '{row.get('laptop_name', row['laptop_model_id'])}'. "
                    f"Hiện chưa có mẫu máy thỏa mãn tuyệt đối 100% các tiêu chí ({violation_text}), "
                    f"nhưng đây là lựa chọn cân đối và phù hợp nhất với điểm đánh giá đạt {score_val * 10:.1f}/10 ⭐."
                ),
            }

    # Fallback sang PuLP nếu Gurobi gặp vấn đề
    return _solve_pulp(df, constraints)


def solve(constraints: dict, scored_laptops: pd.DataFrame, backend: str = "gurobi") -> dict:
    """Interface chính giải bài toán tối ưu và nới lỏng ràng buộc Soft BIP."""
    if scored_laptops.empty:
        return {
            "laptop_id": None,
            "is_feasible": False,
            "is_relaxed": False,
            "explanation": "Chưa có dữ liệu laptop đã chấm điểm (AI_Score)."
        }

    clean_df = _drop_unusable_rows(scored_laptops, constraints)
    if clean_df.empty:
        clean_df = scored_laptops

    if backend == "gurobi" and GUROBI_AVAILABLE:
        try:
            return _solve_gurobi(clean_df, constraints)
        except Exception as e:
            print(f"[Solver Warning] Gurobi gặp sự cố ({e}), chuyển sang PuLP solver.")
            return _solve_pulp(clean_df, constraints)

    return _solve_pulp(clean_df, constraints)


def main():
    """Test nhanh solver độc lập."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, default=Path("data/processed/laptop_dataset_scored.csv"))
    parser.add_argument("--backend", choices=["pulp", "gurobi"], default="gurobi")
    args = parser.parse_args()

    df = pd.read_csv(args.scored)

    test_cases = [
        {"max_price": 15_000_000, "required_tags": ["is_office_friendly"]},
        {"max_price": 25_000_000, "gpu_keyword": "RTX 4060", "required_tags": ["is_gaming_friendly"]},
        {"max_price": 8_000_000, "max_weight": 1.0, "min_battery": 600, "required_tags": []},
    ]

    for i, constraints in enumerate(test_cases, 1):
        print(f"\n=== Test case {i}: {constraints} ===")
        result = solve(constraints, df, backend=args.backend)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()