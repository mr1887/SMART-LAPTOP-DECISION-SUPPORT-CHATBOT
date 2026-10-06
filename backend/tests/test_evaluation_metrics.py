"""
Unit tests cho module Evaluation Metrics & Logger (backend/app/evaluation).
Kiểm tra tính độc lập giữa Hard Constraints và Soft Constraints trong đánh giá thực nghiệm.
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.evaluation.logger import EvaluationLogger, ExperimentRecord
from app.evaluation.metrics import (
    average_latency,
    average_llm_calls,
    average_soft_violations,
    average_tokens,
    compute_all_metrics,
    constraint_satisfaction_rate,
    constraint_violation_rate,
    product_hallucination_rate,
    soft_constraint_violation_rate,
)


class TestEvaluationMetrics(unittest.TestCase):

    def test_hard_satisfied_with_soft_violation(self):
        """
        Kiểm tra: Nếu nghiệm thỏa 100% hard constraints nhưng có vi phạm soft constraint,
        hard_constraint_satisfied PHẢI là True và constraint_satisfaction_rate = 1.0 (100%).
        Soft violation phải được đo tách biệt.
        """
        records = [
            ExperimentRecord(
                query_id="Q1",
                system="hybrid_pipeline",
                parser="gemini",
                raw_query="Laptop 20tr có card rời",
                selected_product="Gaming Pro 15",
                status="OPTIMAL",
                hard_constraint_satisfied=True,  # Thỏa toàn bộ hard
                has_soft_violation=True,         # Nhưng vượt nhẹ ngân sách soft
                soft_violation_count=1,
            ),
            ExperimentRecord(
                query_id="Q2",
                system="hybrid_pipeline",
                parser="gemini",
                raw_query="Laptop 15tr học tập",
                selected_product="Office UltraBook 13",
                status="OPTIMAL",
                hard_constraint_satisfied=True,  # Thỏa 100% hard
                has_soft_violation=False,        # Không vi phạm soft
                soft_violation_count=0,
            ),
        ]

        # Hard constraint satisfaction rate phải đạt 100% (1.0)
        csr = constraint_satisfaction_rate(records)
        self.assertEqual(csr, 1.0, "Cả 2 queries đều thỏa hard constraint nên satisfaction rate phải = 1.0")

        cvr = constraint_violation_rate(records)
        self.assertEqual(cvr, 0.0, "Không có query nào vi phạm hard constraint nên violation rate phải = 0.0")

        # Soft constraint violation rate đo riêng (1 trong 2 query có soft violation -> 50%)
        scvr = soft_constraint_violation_rate(records)
        self.assertEqual(scvr, 0.5, "1/2 queries có soft violation nên rate phải = 0.5")

        # Số soft violation trung bình: (1 + 0)/2 = 0.5
        avg_sv = average_soft_violations(records)
        self.assertEqual(avg_sv, 0.5)

    def test_hard_violation_infeasible(self):
        """Kiểm tra khi có query infeasible (vi phạm hard constraint)."""
        records = [
            {
                "query_id": "Q1",
                "hard_constraint_satisfied": True,
                "has_soft_violation": False,
                "soft_violation_count": 0,
            },
            {
                "query_id": "Q2",
                "hard_constraint_satisfied": False,  # INFEASIBLE
                "has_soft_violation": False,
                "soft_violation_count": 0,
            },
        ]

        csr = constraint_satisfaction_rate(records)
        self.assertEqual(csr, 0.5)
        cvr = constraint_violation_rate(records)
        self.assertEqual(cvr, 0.5)

    def test_product_hallucination_rate(self):
        """Kiểm tra tỷ lệ hallucination sản phẩm đối chiếu catalog."""
        catalog = {"Asus TUF Gaming A15", "Dell Inspiron 14", "MacBook Air M2"}
        records = [
            {"selected_product": "Asus TUF Gaming A15"},
            {"selected_product": "Dell Inspiron 14"},
            {"selected_product": "Non-existent Phantom Laptop XYZ"},  # Hallucinated
        ]

        rate = product_hallucination_rate(records, catalog)
        self.assertAlmostEqual(rate, 1 / 3, places=4)

    def test_compute_all_metrics_structure(self):
        """Kiểm tra output của compute_all_metrics có đầy đủ các trường đánh giá chuẩn xác."""
        records = [
            {
                "hard_constraint_satisfied": True,
                "has_soft_violation": True,
                "soft_violation_count": 1,
                "selected_product": "Asus TUF",
                "llm_calls": 1,
                "input_tokens": 150,
                "output_tokens": 50,
                "total_latency_ms": 250.0,
            }
        ]

        metrics = compute_all_metrics(records, catalog_product_names={"Asus TUF"})
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["constraint_satisfaction_rate"], 1.0)
        self.assertEqual(metrics["constraint_violation_rate"], 0.0)
        self.assertEqual(metrics["soft_constraint_violation_rate"], 1.0)
        self.assertEqual(metrics["average_soft_violations"], 1.0)
        self.assertEqual(metrics["product_hallucination_rate"], 0.0)
        self.assertEqual(metrics["average_llm_calls"], 1.0)
        self.assertEqual(metrics["avg_input_tokens"], 150.0)
        self.assertEqual(metrics["avg_output_tokens"], 50.0)
        self.assertEqual(metrics["avg_total_tokens"], 200.0)


if __name__ == "__main__":
    unittest.main()
