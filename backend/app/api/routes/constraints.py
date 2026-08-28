"""
Route /api/constraints - Endpoint để test trực tiếp solver với ràng buộc tùy chỉnh.

Hữu ích khi debug hoặc khi frontend muốn gọi solver trực tiếp
mà không qua NLP parsing.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.optimizer import solver

router = APIRouter()

_SCORED_CSV = Path(__file__).parents[4] / "data" / "processed" / "laptop_dataset_scored.csv"

_df_cache: pd.DataFrame | None = None


def _load_scored_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    if not _SCORED_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {_SCORED_CSV}. "
            "Hãy chạy `python backend/app/scoring/train_lgbm.py` trước."
        )
    _df_cache = pd.read_csv(_SCORED_CSV)
    return _df_cache


# ---------- Schema ----------

class ConstraintRequest(BaseModel):
    max_price: float | None = None
    min_price: float | None = None
    max_weight: float | None = None
    min_battery: float | None = None
    required_tags: list[str] = []
    backend: str = "gurobi"


# ---------- Endpoint ----------

@router.post("")
def solve_with_constraints(req: ConstraintRequest):
    """Gọi solver trực tiếp với bộ ràng buộc được cung cấp."""
    constraints = {
        "max_price":    req.max_price,
        "min_price":    req.min_price,
        "max_weight":   req.max_weight,
        "min_battery":  req.min_battery,
        "required_tags": req.required_tags,
    }

    try:
        df = _load_scored_df()
        result = solver.solve(constraints, df, backend=req.backend)
        return {"constraints": constraints, "result": result}
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
