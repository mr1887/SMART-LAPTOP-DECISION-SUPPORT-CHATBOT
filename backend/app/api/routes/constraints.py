"""
Route /api/constraints - Endpoint để test trực tiếp solver với ràng buộc tùy chỉnh.

Hữu ích khi debug hoặc khi frontend muốn gọi solver trực tiếp
mà không qua NLP parsing.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.nlp.nl2constraint import convert_legacy_regex_to_requirement_set
from app.optimizer.ortools_solver import solve as solve_ortools

router = APIRouter()

_df_cache: pd.DataFrame | None = None


def _find_scored_csv() -> Path:
    candidates = [
        Path.cwd() / "data" / "processed" / "laptop_dataset_scored.csv",
        Path("/app/data/processed/laptop_dataset_scored.csv"),
        Path(__file__).resolve().parents[4] / "data" / "processed" / "laptop_dataset_scored.csv",
        Path(__file__).resolve().parents[3] / "data" / "processed" / "laptop_dataset_scored.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_scored_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    csv_path = _find_scored_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {_SCORED_CSV}. "
            "Hãy chạy `python backend/app/scoring/train_lgbm.py` trước."
        )
    _df_cache = pd.read_csv(csv_path)
    return _df_cache


# ---------- Schema ----------

class ConstraintRequest(BaseModel):
    max_price: float | None = None
    min_price: float | None = None
    max_weight: float | None = None
    min_battery: float | None = None
    require_discrete_gpu: bool | None = None
    gpu_keyword: str | None = None
    required_tags: list[str] = []
    backend: str = "ortools"


# ---------- Endpoint ----------

@router.post("")
def solve_with_constraints(req: ConstraintRequest):
    """Gọi Google OR-Tools solver trực tiếp với bộ ràng buộc được cung cấp, trả về câu trả lời và thông tin chi tiết đầy đủ."""
    constraints = {
        "max_price":            req.max_price,
        "min_price":            req.min_price,
        "max_weight":           req.max_weight,
        "min_battery":          req.min_battery,
        "require_discrete_gpu": req.require_discrete_gpu,
        "gpu_keyword":          req.gpu_keyword,
        "required_tags":        req.required_tags,
    }

    try:
        df = _load_scored_df()
        req_set = convert_legacy_regex_to_requirement_set(constraints)
        opt_res = solve_ortools(df, req_set)

        has_soft_v = bool(opt_res.get("has_soft_violation", False))
        soft_vs = opt_res.get("soft_violations", [])
        violations = [v["violation"] for v in soft_vs] if soft_vs else []
        violation_text = ", ".join(violations) if violations else ""

        result = {
            "laptop_id": opt_res.get("laptop_id"),
            "is_feasible": bool(opt_res.get("is_feasible")) and not has_soft_v,
            "is_relaxed": has_soft_v,
            "ai_score": opt_res.get("relevance_score"),
            "relevance_score": opt_res.get("relevance_score"),
            "violations": violations,
            "violation_text": violation_text,
            "status": opt_res.get("status"),
            "hard_violations": opt_res.get("hard_violations", []),
            "soft_violations": soft_vs,
        }
        
        # Import helper từ route chat để sinh câu trả lời đầy đủ & laptop_details
        from app.api.routes.chat import _get_laptop_details, _build_reply
        laptop_details = _get_laptop_details(df, result.get("laptop_id"))
        
        reply = None
        if result is not None:
            try:
                from app.ai.gemini_service import generate_gemini_consultation
                reply = generate_gemini_consultation(
                    user_message="Tìm kiếm laptop theo bộ lọc",
                    constraints=constraints,
                    result=result,
                    laptop_details=laptop_details,
                )
            except Exception:
                reply = None

        if reply is None:
            reply = _build_reply(constraints, result, laptop_details)

        return {
            "constraints": constraints,
            "result": result,
            "laptop_details": laptop_details,
            "reply": reply,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi solver: {e}")


@router.get("/tags")
def get_available_tags():
    """Trả về danh sách các tag nhu cầu hợp lệ."""
    return {
        "tags": [
            "is_gaming_friendly",
            "is_office_friendly",
            "is_programming_friendly",
            "is_graphic_friendly",
        ]
    }
