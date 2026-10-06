"""
Evaluation Logger Module - Ghi nhận log chi tiết cho các thử nghiệm benchmark / evaluation.

Lưu trữ local dạng JSONL (hoặc xuất CSV/DataFrame), không phụ thuộc Firestore hay external services.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DEFAULT_LOG_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "experiment_logs.jsonl"


@dataclass
class ExperimentRecord:
    """Cấu trúc dữ liệu cho một bản ghi thực nghiệm hệ thống."""
    query_id: str
    system: str  # e.g., "hybrid_pipeline", "llm_only", "baseline_solver"
    parser: str  # e.g., "gemini", "regex", "none"
    raw_query: str
    parsed_requirements: Union[Dict[str, Any], List[Any], str] = field(default_factory=dict)
    candidate_count: int = 0
    selected_product: Optional[Union[Dict[str, Any], str, int]] = None
    status: str = "UNKNOWN"  # e.g., "OPTIMAL", "FEASIBLE", "INFEASIBLE", "ERROR"
    hard_constraint_satisfied: bool = True
    has_soft_violation: bool = False
    soft_violation_count: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    nlu_latency_ms: float = 0.0
    scoring_latency_ms: float = 0.0
    optimization_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationLogger:
    """Logger ghi nhận các bản ghi thực nghiệm vào file JSONL cục bộ."""

    def __init__(self, log_path: Optional[Union[str, Path]] = None):
        if log_path is None:
            self.log_path = DEFAULT_LOG_PATH
        else:
            self.log_path = Path(log_path)
        
        # Đảm bảo thư mục lưu trữ tồn tại
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Union[ExperimentRecord, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ghi một bản ghi thực nghiệm vào file JSONL.
        
        Args:
            record: Instance của ExperimentRecord hoặc dict chứa các trường tương ứng.
        """
        if isinstance(record, ExperimentRecord):
            data = record.to_dict()
        else:
            data = dict(record)

        # Chuyển đổi an toàn sang JSON
        json_line = json.dumps(data, ensure_ascii=False, default=str)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        return data

    def load_records(self) -> List[Dict[str, Any]]:
        """Đọc tất cả bản ghi từ file JSONL."""
        if not self.log_path.exists():
            return []
        
        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def clear(self) -> None:
        """Xóa nội dung file log hiện tại."""
        if self.log_path.exists():
            self.log_path.unlink()


# Singleton / Convenience helper functions
_default_logger = EvaluationLogger()


def log_experiment(record: Union[ExperimentRecord, Dict[str, Any]], log_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Helper ghi nhận nhanh một record thực nghiệm."""
    if log_path:
        logger = EvaluationLogger(log_path)
        return logger.log(record)
    return _default_logger.log(record)


def load_experiments(log_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Helper đọc danh sách các record thực nghiệm."""
    if log_path:
        logger = EvaluationLogger(log_path)
        return logger.load_records()
    return _default_logger.load_records()
