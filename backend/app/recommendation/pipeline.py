"""
Pipeline tư vấn & tối ưu hóa lựa chọn Laptop thuần túy (Experiment-ready Pure Pipeline).
Thực hiện toàn bộ chu trình tính toán:
1. NLU Parsing (Gemini / Regex Fallback)
2. Semantic Validation & Normalization
3. Hard-filter Candidate Retrieval
4. Global Relevance Prediction (LightGBM)
5. Query Match Scoring
6. Hybrid Relevance Score Combination
7. Google OR-Tools CP-SAT Optimization
8. Structured Result Formatting

Hoàn toàn độc lập với Web Framework (FastAPI), Database (Firestore) và LLM Explanation.
"""

from pathlib import Path
from typing import Any, Optional, Union
import pandas as pd

from app.nlp.nl2constraint import parse as parse_nlu, classify_context_action
from app.nlp.schema import RequirementSet
from app.nlp.validator import validate_requirement_set
from app.optimizer.ortools_solver import (
    _find_column,
    solve as solve_optimization,
    solve_nearest_alternative,
    solve_top3,
)
from app.retrieval.candidate_retriever import retrieve_candidates
from app.scoring.predict import predict as predict_global_relevance
from app.scoring.query_match import combine_scores, compute_query_match_score

_df_cache: Optional[pd.DataFrame] = None


def _find_dataset_path() -> Path:
    """Tự động tìm kiếm file dữ liệu laptop đã xử lý."""
    candidates = [
        Path.cwd() / "data" / "processed" / "laptop_dataset_scored.csv",
        Path.cwd() / "data" / "processed" / "laptop_dataset_tagged.csv",
        Path(__file__).resolve().parents[3] / "data" / "processed" / "laptop_dataset_scored.csv",
        Path(__file__).resolve().parents[3] / "data" / "processed" / "laptop_dataset_tagged.csv",
        Path(__file__).resolve().parents[4] / "data" / "processed" / "laptop_dataset_scored.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_dataset() -> pd.DataFrame:
    """Lazy load và cache dataset mặc định cho pipeline."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    csv_path = _find_dataset_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset tại {csv_path}.")
    _df_cache = pd.read_csv(csv_path)
    return _df_cache


def _merge_requirements(
    base: Union[dict[str, Any], RequirementSet],
    new_req: Union[dict[str, Any], RequirementSet]
) -> dict[str, Any]:
    """Hợp nhất các ràng buộc và sở thích qua các lượt hội thoại."""
    base_dict = base.model_dump() if isinstance(base, RequirementSet) else (base or {})
    new_dict = new_req.model_dump() if isinstance(new_req, RequirementSet) else (new_req or {})

    # Gộp constraints theo trường (cập nhật nếu có ràng buộc mới)
    constraint_map = {}
    for c in base_dict.get("constraints", []):
        field = c.get("field") if isinstance(c, dict) else getattr(c, "field", None)
        if field:
            constraint_map[field] = c
    for c in new_dict.get("constraints", []):
        field = c.get("field") if isinstance(c, dict) else getattr(c, "field", None)
        if field:
            constraint_map[field] = c

    # Gộp preferences theo trường
    pref_map = {}
    for p in base_dict.get("preferences", []):
        field = p.get("field") if isinstance(p, dict) else getattr(p, "field", None)
        if field:
            pref_map[field] = p
    for p in new_dict.get("preferences", []):
        field = p.get("field") if isinstance(p, dict) else getattr(p, "field", None)
        if field:
            pref_map[field] = p

    # Required tags biểu diễn "nhu cầu chính". Khi lượt mới nêu rõ một nhu cầu
    # mới (ví dụ gaming -> văn phòng), thay thế tag cũ thay vì union vô hạn.
    base_tags = list(base_dict.get("required_tags", []))
    new_tags = list(new_dict.get("required_tags", []))
    merged_tags = new_tags if new_tags else base_tags

    # Nếu người dùng chuyển sang nhu cầu văn phòng/học tập thuần túy mà lượt mới
    # không nhắc GPU cụ thể, loại bỏ ràng buộc GPU cũ để tránh "kẹt" RTX/gaming
    # từ truy vấn trước.
    new_constraint_fields = {
        c.get("field") if isinstance(c, dict) else getattr(c, "field", None)
        for c in new_dict.get("constraints", [])
    }
    switched_to_office = (
        "is_office_friendly" in new_tags
        and "is_gaming_friendly" not in new_tags
        and set(new_tags) != set(base_tags)
    )
    if switched_to_office:
        if "gpu_keyword" not in new_constraint_fields:
            constraint_map.pop("gpu_keyword", None)
        if "gpu_discrete" not in new_constraint_fields:
            constraint_map.pop("gpu_discrete", None)

    return {
        "constraints": list(constraint_map.values()),
        "preferences": list(pref_map.values()),
        "required_tags": merged_tags,
    }


def _is_fresh_search_request(parsed: Union[dict[str, Any], RequirementSet]) -> bool:
    """Một truy vấn mới đủ đầy đủ (có nhu cầu + ngân sách) nên thay context cũ.

    Mục tiêu: tránh trường hợp user chuyển từ "gaming RTX 4060 25tr" sang
    "văn phòng 15tr" nhưng session vẫn giữ GPU/tag cũ. Các follow-up ngắn
    như "rẻ hơn", "nhẹ hơn" vẫn tiếp tục merge với context hiện tại.
    """
    data = parsed.model_dump() if isinstance(parsed, RequirementSet) else (parsed or {})
    tags = data.get("required_tags", []) or []
    constraints = data.get("constraints", []) or []

    has_price = False
    for c in constraints:
        field = c.get("field") if isinstance(c, dict) else getattr(c, "field", None)
        if field == "price":
            has_price = True
            break

    return bool(tags) and has_price


def recommend(
    query: str,
    current_constraints: Optional[Union[dict[str, Any], RequirementSet]] = None,
    df: Optional[pd.DataFrame] = None,
    alpha: float = 0.6,
    use_gemini_nlu: bool = True,
) -> dict[str, Any]:
    """
    Thực hiện trọn vẹn pipeline gợi ý laptop:
    1. parse NLU
    2. validate
    3. retrieve candidates
    4. predict global relevance
    5. compute query match
    6. combine final relevance
    7. OR-Tools optimization
    8. return structured result

    Args:
        query: Câu hỏi/truy vấn của người dùng.
        current_constraints: Bộ ràng buộc đã tích lũy từ các lượt trước (nếu có).
        df: DataFrame dữ liệu laptop (tự động load dataset nếu để None).
        alpha: Trọng số kết hợp giữa global relevance (LightGBM) và query match score.
        use_gemini_nlu: Có ưu tiên sử dụng Gemini để trích xuất NLU hay không.

    Returns:
        dict kết quả có cấu trúc hoàn chỉnh.
    """
    # 1. Parse NLU
    raw_nlu = parse_nlu(query, use_gemini=use_gemini_nlu)

    # Dialogue-context action:
    # ADD     -> giữ context cũ và bổ sung field mới.
    # UPDATE  -> merge theo field; field mới ghi đè field cũ cùng tên.
    # REPLACE -> bắt đầu một yêu cầu tìm kiếm mới, bỏ context cũ.
    context_action = classify_context_action(
        query,
        parsed_requirements=raw_nlu,
        current_requirements=(
            current_constraints.model_dump()
            if isinstance(current_constraints, RequirementSet)
            else (current_constraints or {})
        ),
    )

    if current_constraints and context_action in ("ADD", "UPDATE"):
        merged_raw = _merge_requirements(current_constraints, raw_nlu)
    else:
        merged_raw = raw_nlu

    # 2. Validate & Normalize
    validated_req = validate_requirement_set(merged_raw)

    # 3. Retrieve Candidates (Lọc an toàn các ràng buộc cứng)
    source_df = df if df is not None else load_dataset()
    candidates = retrieve_candidates(source_df, validated_req)

    # Nếu không còn ứng viên nào thỏa mãn các ràng buộc cứng ban đầu -> Fallback Nearest Alternative
    if candidates.empty:
        if source_df is None or source_df.empty:
            opt_infeasible = {
                "laptop_id": None,
                "status": "INFEASIBLE",
                "is_feasible": False,
                "relevance_score": None,
                "hard_violations": ["Không có laptop nào trong cơ sở dữ liệu."],
                "has_soft_violation": False,
                "soft_violations": [],
                "violations": [],
                "recommendations": [],
            }
            return {
                "query": query,
                "context_action": context_action,
                "requirements": validated_req.model_dump(),
                "candidates_count": 0,
                "optimization": opt_infeasible,
                "recommendations": [],
                "recommended_laptops": [],
                "recommended_laptop": None,
            }

        # Tính điểm relevance cho toàn bộ dataset nguồn để làm tie-breaker
        fallback_scored = source_df.copy()
        if "relevance_score" not in fallback_scored.columns and "AI_Score" not in fallback_scored.columns:
            fallback_scored = predict_global_relevance(fallback_scored)
        else:
            if "relevance_score" not in fallback_scored.columns:
                fallback_scored["relevance_score"] = fallback_scored["AI_Score"]
            if "AI_Score" not in fallback_scored.columns:
                fallback_scored["AI_Score"] = fallback_scored["relevance_score"]

        q_match = compute_query_match_score(fallback_scored, validated_req)
        fallback_scored["query_match_score"] = q_match
        fallback_scored["final_relevance_score"] = combine_scores(
            fallback_scored["relevance_score"],
            fallback_scored["query_match_score"],
            alpha=alpha
        )

        opt_fallback = solve_nearest_alternative(fallback_scored, validated_req)
        recommendations = opt_fallback.get("recommendations", [])
        recommended_laptops: list[dict[str, Any]] = []

        id_col = _find_column(fallback_scored, ["laptop_model_id", "laptop_id"])
        for rec in recommendations:
            rec_id = rec.get("laptop_id")
            if rec_id is not None:
                if id_col and id_col in fallback_scored.columns:
                    matched_rows = fallback_scored[fallback_scored[id_col] == rec_id]
                else:
                    matched_rows = fallback_scored[fallback_scored.index == rec_id]

                if not matched_rows.empty:
                    recommended_laptops.append(matched_rows.iloc[0].to_dict())

        recommended_laptop = recommended_laptops[0] if recommended_laptops else None

        return {
            "query": query,
            "context_action": context_action,
            "requirements": validated_req.model_dump(),
            "candidates_count": 0,
            "optimization": opt_fallback,
            "recommendations": recommendations,
            "recommended_laptops": recommended_laptops,
            "recommended_laptop": recommended_laptop,
        }

    # 4. Predict Global Relevance (LightGBM engagement proxy regression)
    if "relevance_score" not in candidates.columns and "AI_Score" not in candidates.columns:
        candidates_scored = predict_global_relevance(candidates)
    else:
        candidates_scored = candidates.copy()
        if "relevance_score" not in candidates_scored.columns:
            candidates_scored["relevance_score"] = candidates_scored["AI_Score"]
        if "AI_Score" not in candidates_scored.columns:
            candidates_scored["AI_Score"] = candidates_scored["relevance_score"]

    # 5. Compute Query Match Score
    query_match_scores = compute_query_match_score(candidates_scored, validated_req)
    candidates_scored["query_match_score"] = query_match_scores

    # 6. Combine Final Hybrid Relevance Score
    final_scores = combine_scores(
        candidates_scored["relevance_score"],
        candidates_scored["query_match_score"],
        alpha=alpha
    )
    candidates_scored["final_relevance_score"] = final_scores

    # 7. OR-Tools Optimization (Top-3 CP-SAT Solver)
    print(
        ">>> PIPELINE BEFORE TOP3:",
        "candidates =", len(candidates_scored),
        "requirements =", validated_req.model_dump(),
        flush=True,
    )

    opt_result = solve_top3(candidates_scored, validated_req)

    # 8. Format Structured Result
    recommendations = opt_result.get("recommendations", [])

    print(
        ">>> PIPELINE AFTER TOP3:",
        "recommendations =", len(recommendations),
        [(r.get("rank"), r.get("type"), r.get("laptop_id")) for r in recommendations],
        flush=True,
    )
    recommended_laptops: list[dict[str, Any]] = []

    id_col = _find_column(candidates_scored, ["laptop_model_id", "laptop_id"])

    for rec in recommendations:
        rec_id = rec.get("laptop_id")
        candidate_index = rec.get("candidate_index")

        laptop_detail = None

        # Ưu tiên đúng row mà solver đã chọn. Điều này tránh lấy nhầm biến thể
        # khi nhiều dòng có cùng laptop_model_id nhưng giá/cấu hình khác nhau.
        if candidate_index is not None and candidate_index in candidates_scored.index:
            laptop_detail = candidates_scored.loc[candidate_index].to_dict()
        elif rec_id is not None:
            if id_col and id_col in candidates_scored.columns:
                matched_rows = candidates_scored[candidates_scored[id_col] == rec_id]
            else:
                matched_rows = candidates_scored[candidates_scored.index == rec_id]

            if not matched_rows.empty:
                laptop_detail = matched_rows.iloc[0].to_dict()

        if laptop_detail is not None:
            laptop_detail["rank"] = rec.get("rank")
            laptop_detail["type"] = rec.get("type")
            laptop_detail["utility_score"] = rec.get("utility_score")
            laptop_detail["performance_score"] = rec.get("performance_score")

            # Giá canonical phải là đúng giá mà solver đã dùng để tối ưu.
            # Dataset có thể đồng thời chứa price / price_vnd với giá trị không đồng bộ.
            solver_price = rec.get("price")
            if solver_price is not None:
                laptop_detail["solver_price"] = solver_price
                laptop_detail["price"] = solver_price
                laptop_detail["price_vnd"] = solver_price

            recommended_laptops.append(laptop_detail)

    recommended_laptop = recommended_laptops[0] if recommended_laptops else None

    return {
        "query": query,
        "context_action": context_action,
        "requirements": validated_req.model_dump(),
        "candidates_count": len(candidates_scored),
        "optimization": opt_result,
        "recommendations": recommendations,
        "recommended_laptops": recommended_laptops,
        "recommended_laptop": recommended_laptop,
    }
