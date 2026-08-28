"""
Tầng 3 - Decision Support System (Gurobi/PuLP Optimizer).

Nhận: bộ ràng buộc (từ Tầng 1 Regex) + AI_Score cho từng laptop (từ Tầng 2
LightGBM), trả về 1 laptop tối ưu tuân thủ 100% ràng buộc cứng, hoặc giải
thích rõ vì sao Infeasible kèm phương án gần đạt chuẩn nhất.

Backend mặc định: PuLP + CBC solver (miễn phí, không cần license).
Nếu máy có cài gurobipy VÀ có license hợp lệ, có thể đổi backend="gurobi"
để dùng Gurobi thật (xem hàm _solve_gurobi ở cuối file) - giữ 2 backend
sau cùng 1 interface solve() để không phụ thuộc vào việc có license hay
không khi demo/báo cáo.

Cách dùng độc lập (test nhanh không cần chạy cả app):
    python solver.py --scored data/processed/laptop_dataset_scored.csv
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

import os

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# ---------- Xác thực Gurobi ----------
# CÓ 2 CÁCH license được nhận diện, tùy vào máy bạn đã setup thế nào:
#
# Cách 1 (đơn giản, TỰ ĐỘNG): nếu bạn đã chạy `grbgetkey <license-key>`
# lúc cài đặt, Gurobi đã tự tải sẵn file gurobi.lic chứa WLSACCESSID/
# WLSSECRET/LICENSEID vào thư mục mặc định trên máy. Lúc này chỉ cần gọi
# gp.Model() TRƠN, không cần khai báo gì - Gurobi tự tìm thấy file đó.
# Đây là cách hầu hết người dùng WLS trên máy cá nhân gặp phải.
#
# Cách 2 (dự phòng, môi trường không có file sẵn - VD Docker container
# mới chưa từng chạy grbgetkey): phải khai báo credentials tường minh
# qua gp.Env(params=...), đọc từ biến môi trường GUROBI_WLSACCESSID/
# GUROBI_WLSSECRET/GUROBI_LICENSEID (xem .env.example).
#
# Code dưới đây THỬ CÁCH 1 TRƯỚC (đúng như Gurobi hoạt động mặc định),
# chỉ rơi xuống Cách 2 nếu Cách 1 báo lỗi liên quan tới license.
_GUROBI_ENV = None  # cache Env nếu phải dùng Cách 2, tránh xác thực lại mỗi lần


def _create_gurobi_model(name: str) -> "gp.Model":
    """Tạo Gurobi Model - ưu tiên để Gurobi TỰ TÌM license file trên máy
    (Cách 1), chỉ khai báo credentials tường minh (Cách 2) nếu Cách 1
    thất bại. In rõ đang dùng cách nào để bạn biết máy mình thuộc dạng nào."""
    global _GUROBI_ENV

    if _GUROBI_ENV is not None:
        return gp.Model(name, env=_GUROBI_ENV)

    try:
        # Cách 1: không truyền env gì cả - Gurobi tự dò file gurobi.lic
        model = gp.Model(name)
        return model
    except gp.GurobiError as e:
        license_related = "license" in str(e).lower() or "wls" in str(e).lower()
        if not license_related:
            raise  # lỗi khác, không phải do thiếu license - không nên nuốt lỗi

        print(
            f"Không tìm thấy license file tự động trên máy ({e}). "
            f"Chuyển sang khai báo WLS credentials tường minh từ biến môi trường..."
        )
        access_id = os.environ.get("GUROBI_WLSACCESSID")
        secret = os.environ.get("GUROBI_WLSSECRET")
        license_id = os.environ.get("GUROBI_LICENSEID")

        missing = [
            n for n, v in [("GUROBI_WLSACCESSID", access_id),
                            ("GUROBI_WLSSECRET", secret),
                            ("GUROBI_LICENSEID", license_id)] if not v
        ]
        if missing:
            raise RuntimeError(
                f"Gurobi không tự tìm được license trên máy, và cũng thiếu "
                f"biến môi trường {missing} để xác thực thủ công.\n"
                f"-> Cách sửa NHANH NHẤT (khuyến nghị): chạy lệnh sau trong "
                f"terminal (chỉ cần làm 1 LẦN DUY NHẤT):\n"
                f"     grbgetkey <license-key-của-bạn>\n"
                f"   Sau đó gp.Model() sẽ tự động chạy được, không cần sửa "
                f"code gì thêm.\n"
                f"-> Hoặc nếu chạy trong Docker/môi trường không lưu được "
                f"file: điền GUROBI_WLSACCESSID/GUROBI_WLSSECRET/"
                f"GUROBI_LICENSEID vào file .env."
            ) from e

        _GUROBI_ENV = gp.Env(params={
            "WLSAccessID": access_id,
            "WLSSecret": secret,
            "LicenseID": int(license_id),
        })
        print("Xác thực WLS bằng biến môi trường thành công.")
        return gp.Model(name, env=_GUROBI_ENV)


# Trọng số phạt khi phải nới lỏng ràng buộc (Soft Constraint) - càng lớn
# càng "ép" solver ưu tiên tuân thủ ràng buộc đó hơn là chọn AI_Score cao.
# Đơn vị của mỗi lambda phải quy đổi tương đối so với thang AI_Score [0,1].
RELAX_PENALTY = {
    "price": 1e-8,          # phạt theo VNĐ vượt ngân sách tối đa (max_price)
    "min_price": 1e-8,      # phạt theo VNĐ thiếu so với ngân sách tối thiểu (min_price)
    "weight": 0.3,           # phạt theo kg vượt cân nặng
    "battery": 0.002,        # phạt theo phút thiếu pin
}


def _drop_unusable_rows(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Loại các laptop bị thiếu (NaN) đúng ở cột ràng buộc đang được áp dụng -
    PuLP/Gurobi không xử lý được NaN trong hệ số ràng buộc. Chỉ laptop nào
    có đủ dữ liệu cho MỌI ràng buộc đang bật mới được đưa vào solver."""
    battery_col = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else "office_battery_result_minutes"

    required_cols = ["AI_Score"]
    if constraints.get("max_price") is not None or constraints.get("min_price") is not None:
        required_cols.append("price")
    if constraints.get("max_weight") is not None:
        required_cols.append("laptop_weight")
    if constraints.get("min_battery") is not None:
        required_cols.append(battery_col)

    before = len(df)
    df_clean = df.dropna(subset=required_cols)
    dropped = before - len(df_clean)
    if dropped > 0:
        print(
            f"Loại {dropped}/{before} laptop khỏi solver vì thiếu dữ liệu ở "
            f"cột ràng buộc đang xét ({required_cols})."
        )
    return df_clean


def _filter_by_tags(df: pd.DataFrame, required_tags: list[str]) -> tuple[pd.DataFrame, Optional[str]]:
    """Lọc cứng theo nhãn nhu cầu (is_gaming_friendly...). Trả về dataframe
    đã lọc + tên tag đầu tiên làm rỗng tập ứng viên (nếu có), để giải
    thích rõ nguyên nhân Infeasible ngay từ bước này thay vì để solver
    chạy trên tập rỗng."""
    candidates = df.copy()
    for tag in required_tags or []:
        if tag not in candidates.columns:
            continue
        before = len(candidates)
        candidates = candidates[candidates[tag] == True]  # noqa: E712
        if len(candidates) == 0 and before > 0:
            return candidates, tag
    return candidates, None


def _diagnose_infeasible(df: pd.DataFrame, constraints: dict) -> dict:
    """Chẩn đoán thủ công kiểu 'IIS' cho PuLP (PuLP không có computeIIS
    sẵn như Gurobi) - kiểm tra TỪNG ràng buộc riêng lẻ xem có bao nhiêu
    laptop thỏa mãn, để biết ràng buộc nào đang 'siết' quá chặt."""
    diagnosis = {}

    if constraints.get("max_price") is not None:
        n_ok = (df["price"] <= constraints["max_price"]).sum()
        diagnosis["max_price"] = f"{n_ok}/{len(df)} laptop trong ngân sách tối đa"

    if constraints.get("min_price") is not None:
        n_ok = (df["price"] >= constraints["min_price"]).sum()
        diagnosis["min_price"] = f"{n_ok}/{len(df)} laptop đạt ngân sách tối thiểu"

    if constraints.get("max_weight") is not None:
        n_ok = (df["laptop_weight"] <= constraints["max_weight"]).sum()
        diagnosis["max_weight"] = f"{n_ok}/{len(df)} laptop đủ nhẹ"

    if constraints.get("min_battery") is not None:
        battery_col = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else "office_battery_result_minutes"
        n_ok = (df[battery_col] >= constraints["min_battery"]).sum()
        diagnosis["min_battery"] = f"{n_ok}/{len(df)} laptop đủ pin"

    return diagnosis


def _solve_pulp(df: pd.DataFrame, constraints: dict) -> dict:
    """Bài toán BIP chính: max AI_Score, ràng buộc cứng giá/cân nặng/pin.
    Nếu infeasible, tự động chuyển sang bài toán nới lỏng (Soft Constraint)
    để tìm laptop 'gần đạt chuẩn nhất'."""

    battery_col = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else "office_battery_result_minutes"

    # ---------- Vòng 1: thử giải với ràng buộc cứng tuyệt đối ----------
    prob = pulp.LpProblem("laptop_selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(df.loc[i, "AI_Score"] * x[i] for i in df.index)
    prob += pulp.lpSum(x[i] for i in df.index) == 1

    if constraints.get("max_price") is not None:
        prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in df.index) <= constraints["max_price"]
    if constraints.get("min_price") is not None:
        prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in df.index) >= constraints["min_price"]
    if constraints.get("max_weight") is not None:
        prob += pulp.lpSum(df.loc[i, "laptop_weight"] * x[i] for i in df.index) <= constraints["max_weight"]
    if constraints.get("min_battery") is not None:
        prob += pulp.lpSum(df.loc[i, battery_col] * x[i] for i in df.index) >= constraints["min_battery"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        chosen = [i for i in df.index if x[i].value() == 1][0]
        row = df.loc[chosen]
        return {
            "laptop_id": int(row["laptop_model_id"]),
            "is_feasible": True,
            "is_relaxed": False,
            "ai_score": float(row["AI_Score"]),
            "explanation": (
                f"Chọn '{row.get('laptop_name', row['laptop_model_id'])}' - "
                f"thỏa mãn 100% ràng buộc, AI_Score cao nhất trong tập ứng viên "
                f"({row['AI_Score']:.3f})."
            ),
        }

    # ---------- Vòng 2: Infeasible - chẩn đoán + nới lỏng (Soft Constraint) ----------
    diagnosis = _diagnose_infeasible(df, constraints)

    relax_prob = pulp.LpProblem("laptop_selection_relaxed", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}
    slack_price = pulp.LpVariable("slack_price", lowBound=0)
    slack_min_price = pulp.LpVariable("slack_min_price", lowBound=0)
    slack_weight = pulp.LpVariable("slack_weight", lowBound=0)
    slack_battery = pulp.LpVariable("slack_battery", lowBound=0)

    relax_prob += (
        pulp.lpSum(df.loc[i, "AI_Score"] * x[i] for i in df.index)
        - RELAX_PENALTY["price"] * slack_price
        - RELAX_PENALTY["min_price"] * slack_min_price
        - RELAX_PENALTY["weight"] * slack_weight
        - RELAX_PENALTY["battery"] * slack_battery
    )
    relax_prob += pulp.lpSum(x[i] for i in df.index) == 1

    if constraints.get("max_price") is not None:
        relax_prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in df.index) <= constraints["max_price"] + slack_price
    if constraints.get("min_price") is not None:
        relax_prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in df.index) >= constraints["min_price"] - slack_min_price
    if constraints.get("max_weight") is not None:
        relax_prob += pulp.lpSum(df.loc[i, "laptop_weight"] * x[i] for i in df.index) <= constraints["max_weight"] + slack_weight
    if constraints.get("min_battery") is not None:
        relax_prob += pulp.lpSum(df.loc[i, battery_col] * x[i] for i in df.index) >= constraints["min_battery"] - slack_battery

    relax_prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[relax_prob.status] != "Optimal":
        # Trường hợp hiếm: ngay cả bản nới lỏng cũng không giải được
        # (thường do required_tags làm rỗng tập ứng viên từ đầu)
        return {
            "laptop_id": None,
            "is_feasible": False,
            "is_relaxed": False,
            "ai_score": None,
            "explanation": (
                "Không tìm được laptop nào phù hợp, kể cả khi nới lỏng ràng buộc. "
                "Có thể do các nhãn nhu cầu (ngành học/mục đích) đã lọc hết ứng viên. "
                f"Chẩn đoán từng ràng buộc: {json.dumps(diagnosis, ensure_ascii=False)}"
            ),
            "diagnosis": diagnosis,
        }

    chosen = [i for i in df.index if x[i].value() == 1][0]
    row = df.loc[chosen]

    violations = []
    if slack_price.value() and slack_price.value() > 1:
        violations.append(f"vượt ngân sách tối đa {slack_price.value():,.0f}đ")
    if slack_min_price.value() and slack_min_price.value() > 1:
        violations.append(f"thấp hơn ngân sách tối thiểu {slack_min_price.value():,.0f}đ")
    if slack_weight.value() and slack_weight.value() > 0.01:
        violations.append(f"nặng hơn {slack_weight.value():.2f}kg so với yêu cầu")
    if slack_battery.value() and slack_battery.value() > 1:
        violations.append(f"thiếu {slack_battery.value():.0f} phút pin so với yêu cầu")

    violation_text = ", ".join(violations) if violations else "không xác định được mức chênh lệch cụ thể"

    return {
        "laptop_id": int(row["laptop_model_id"]),
        "is_feasible": False,
        "is_relaxed": True,
        "ai_score": float(row["AI_Score"]),
        "explanation": (
            f"Không có laptop nào thỏa mãn 100% yêu cầu. Gợi ý gần đạt chuẩn nhất: "
            f"'{row.get('laptop_name', row['laptop_model_id'])}' ({violation_text}). "
            f"Chẩn đoán: {json.dumps(diagnosis, ensure_ascii=False)}"
        ),
        "diagnosis": diagnosis,
    }


def _solve_gurobi(df: pd.DataFrame, constraints: dict) -> dict:
    """Backend Gurobi thật - dùng computeIIS() khi Infeasible, và tự viết
    bài toán nới lỏng (Soft Constraint) bằng Gurobi luôn, không mượn PuLP,
    để giữ đúng 1 solver xuyên suốt khi bạn đã có license."""
    battery_col = "office_battery_minutes_final" if "office_battery_minutes_final" in df.columns else "office_battery_result_minutes"

    # ---------- Vòng 1: ràng buộc cứng tuyệt đối ----------
    model = _create_gurobi_model("laptop_selection")
    model.setParam("OutputFlag", 0)
    x = model.addVars(df.index, vtype=GRB.BINARY, name="x")

    model.setObjective(
        gp.quicksum(df.loc[i, "AI_Score"] * x[i] for i in df.index), GRB.MAXIMIZE
    )
    model.addConstr(gp.quicksum(x[i] for i in df.index) == 1, name="select_one")

    if constraints.get("max_price") is not None:
        model.addConstr(
            gp.quicksum(df.loc[i, "price"] * x[i] for i in df.index) <= constraints["max_price"],
            name="budget_max",
        )
    if constraints.get("min_price") is not None:
        model.addConstr(
            gp.quicksum(df.loc[i, "price"] * x[i] for i in df.index) >= constraints["min_price"],
            name="budget_min",
        )
    if constraints.get("max_weight") is not None:
        model.addConstr(
            gp.quicksum(df.loc[i, "laptop_weight"] * x[i] for i in df.index) <= constraints["max_weight"],
            name="weight",
        )
    if constraints.get("min_battery") is not None:
        model.addConstr(
            gp.quicksum(df.loc[i, battery_col] * x[i] for i in df.index) >= constraints["min_battery"],
            name="battery",
        )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        chosen = [i for i in df.index if x[i].X > 0.5][0]
        row = df.loc[chosen]
        return {
            "laptop_id": int(row["laptop_model_id"]),
            "is_feasible": True,
            "is_relaxed": False,
            "ai_score": float(row["AI_Score"]),
            "explanation": (
                f"[Gurobi] Chọn '{row.get('laptop_name', row['laptop_model_id'])}' - "
                f"thỏa mãn 100% ràng buộc, AI_Score cao nhất trong tập ứng viên "
                f"({row['AI_Score']:.3f})."
            ),
        }

    # ---------- Vòng 2: Infeasible - computeIIS() để biết ràng buộc nào xung đột ----------
    model.computeIIS()
    conflicting = [c.constrName for c in model.getConstrs() if c.IISConstr]
    diagnosis = _diagnose_infeasible(df, constraints)

    # ---------- Vòng 3: nới lỏng bằng Soft Constraint (Slack Variable) ----------
    relax_model = _create_gurobi_model("laptop_selection_relaxed")
    relax_model.setParam("OutputFlag", 0)
    x = relax_model.addVars(df.index, vtype=GRB.BINARY, name="x")
    slack_price = relax_model.addVar(lb=0, name="slack_price")
    slack_min_price = relax_model.addVar(lb=0, name="slack_min_price")
    slack_weight = relax_model.addVar(lb=0, name="slack_weight")
    slack_battery = relax_model.addVar(lb=0, name="slack_battery")

    relax_model.setObjective(
        gp.quicksum(df.loc[i, "AI_Score"] * x[i] for i in df.index)
        - RELAX_PENALTY["price"] * slack_price
        - RELAX_PENALTY["min_price"] * slack_min_price
        - RELAX_PENALTY["weight"] * slack_weight
        - RELAX_PENALTY["battery"] * slack_battery,
        GRB.MAXIMIZE,
    )
    relax_model.addConstr(gp.quicksum(x[i] for i in df.index) == 1)

    if constraints.get("max_price") is not None:
        relax_model.addConstr(
            gp.quicksum(df.loc[i, "price"] * x[i] for i in df.index) <= constraints["max_price"] + slack_price
        )
    if constraints.get("min_price") is not None:
        relax_model.addConstr(
            gp.quicksum(df.loc[i, "price"] * x[i] for i in df.index) >= constraints["min_price"] - slack_min_price
        )
    if constraints.get("max_weight") is not None:
        relax_model.addConstr(
            gp.quicksum(df.loc[i, "laptop_weight"] * x[i] for i in df.index) <= constraints["max_weight"] + slack_weight
        )
    if constraints.get("min_battery") is not None:
        relax_model.addConstr(
            gp.quicksum(df.loc[i, battery_col] * x[i] for i in df.index) >= constraints["min_battery"] - slack_battery
        )

    relax_model.optimize()

    if relax_model.status != GRB.OPTIMAL:
        return {
            "laptop_id": None,
            "is_feasible": False,
            "is_relaxed": False,
            "ai_score": None,
            "explanation": (
                f"[Gurobi] Không tìm được laptop nào phù hợp, kể cả khi nới lỏng. "
                f"Ràng buộc xung đột (IIS): {conflicting}. "
                f"Chẩn đoán: {json.dumps(diagnosis, ensure_ascii=False)}"
            ),
            "diagnosis": diagnosis,
        }

    chosen = [i for i in df.index if x[i].X > 0.5][0]
    row = df.loc[chosen]

    violations = []
    if slack_price.X > 1:
        violations.append(f"vượt ngân sách tối đa {slack_price.X:,.0f}đ")
    if slack_min_price.X > 1:
        violations.append(f"thấp hơn ngân sách tối thiểu {slack_min_price.X:,.0f}đ")
    if slack_weight.X > 0.01:
        violations.append(f"nặng hơn {slack_weight.X:.2f}kg so với yêu cầu")
    if slack_battery.X > 1:
        violations.append(f"thiếu {slack_battery.X:.0f} phút pin so với yêu cầu")
    violation_text = ", ".join(violations) if violations else "không xác định được mức chênh lệch cụ thể"

    return {
        "laptop_id": int(row["laptop_model_id"]),
        "is_feasible": False,
        "is_relaxed": True,
        "ai_score": float(row["AI_Score"]),
        "explanation": (
            f"[Gurobi] Không có laptop nào thỏa mãn 100% yêu cầu (ràng buộc xung đột: "
            f"{conflicting}). Gợi ý gần đạt chuẩn nhất: "
            f"'{row.get('laptop_name', row['laptop_model_id'])}' ({violation_text}). "
            f"Chẩn đoán: {json.dumps(diagnosis, ensure_ascii=False)}"
        ),
        "diagnosis": diagnosis,
    }


def solve(constraints: dict, scored_laptops: pd.DataFrame, backend: str = "gurobi") -> dict:
    """Interface chính - dùng trong backend/app/api/routes/chat.py.

    constraints: dict dạng {"max_price": ..., "min_price": ..., "max_weight": ...,
                             "min_battery": ..., "required_tags": [...]}
    scored_laptops: DataFrame có cột laptop_model_id, price, laptop_weight,
                     office_battery_minutes_final, AI_Score, và các cột is_*_friendly
    backend: "pulp" (mặc định, không cần license) hoặc "gurobi" (cần license)
    """
    if scored_laptops.empty:
        return {"laptop_id": None, "is_feasible": False, "is_relaxed": False,
                "explanation": "Chưa có dữ liệu laptop đã chấm điểm (AI_Score)."}

    required_tags = constraints.get("required_tags", [])
    clean_df = _drop_unusable_rows(scored_laptops, constraints)

    if clean_df.empty:
        return {
            "laptop_id": None,
            "is_feasible": False,
            "is_relaxed": False,
            "explanation": (
                "Không còn laptop nào đủ dữ liệu cho các ràng buộc đang xét "
                "(quá nhiều giá trị thiếu ở price/laptop_weight/pin)."
            ),
        }

    candidates, empty_tag = _filter_by_tags(clean_df, required_tags)

    if empty_tag is not None:
        return {
            "laptop_id": None,
            "is_feasible": False,
            "is_relaxed": False,
            "explanation": (
                f"Không có laptop nào thỏa nhãn nhu cầu '{empty_tag}' trong toàn bộ "
                f"danh mục hiện có. Cần bỏ bớt yêu cầu về nhu cầu chuyên biệt này."
            ),
        }

    if backend == "gurobi" and GUROBI_AVAILABLE:
        try:
            return _solve_gurobi(candidates, constraints)
        except RuntimeError as e:
            # Thiếu biến môi trường WLS - lỗi cấu hình rõ ràng, nên dừng
            # hẳn để người phát triển sửa .env, KHÔNG âm thầm chuyển PuLP
            # (tránh việc tưởng đang chạy Gurobi nhưng thực ra không phải).
            raise
        except gp.GurobiError as e:  # type: ignore[union-attr]
            # Lỗi kết nối/xác thực WLS (VD mạng chập chờn, license hết hạn)
            # - đây là lỗi RUNTIME ngoài ý muốn khi đang chạy thật, nên
            # chuyển sang PuLP để chatbot vẫn trả lời được cho khách,
            # đồng thời log rõ để biết mà kiểm tra license sau.
            print(
                f"CẢNH BÁO: Gurobi WLS lỗi khi đang chạy ({e}) - tự động "
                f"chuyển sang PuLP cho lượt này. Cần kiểm tra lại license/mạng."
            )
            return _solve_pulp(candidates, constraints)

    if backend == "gurobi" and not GUROBI_AVAILABLE:
        print("CẢNH BÁO: backend='gurobi' được chọn nhưng gurobipy chưa cài/license "
              "không hợp lệ - tự động chuyển sang PuLP.")

    return _solve_pulp(candidates, constraints)


def main():
    """Test nhanh solver độc lập, không cần chạy backend/frontend."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, default=Path("data/processed/laptop_dataset_scored.csv"))
    parser.add_argument("--backend", choices=["pulp", "gurobi"], default="gurobi")
    args = parser.parse_args()

    df = pd.read_csv(args.scored)

    test_cases = [
        {"max_price": 20_000_000, "max_weight": 3, "min_battery": 360, "required_tags": []},
        {"max_price": 8_000_000, "max_weight": 1.0, "min_battery": 600, "required_tags": []},  # cố tình phi thực tế
        {"min_price": 50_000_000, "max_price": 100_000_000, "required_tags": []},  # có cả min và max
    ]

    for i, constraints in enumerate(test_cases, 1):
        print(f"\n=== Test case {i}: {constraints} ===")
        result = solve(constraints, df, backend=args.backend)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()