"""
Final Verification Test Suite cho hệ thống Top-1 Recommendation:
Google OR-Tools CP-SAT Optimizer (ortools_solver.py) & Pure Pipeline (pipeline.py).

Bao phủ 10 kịch bản kiểm thử:
1. price hard feasible
2. RAM hard feasible
3. GPU hard feasible
4. multiple hard constraints
5. price soft
6. weight soft
7. battery soft
8. hard + soft cùng lúc
9. hard infeasible
10. feasible + soft violation
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd

from app.nlp.schema import Constraint, RequirementSet
from app.optimizer.ortools_solver import solve, solve_hard_constraints
from app.recommendation.pipeline import recommend
from app.retrieval.candidate_retriever import retrieve_candidates


class TestOrToolsTop1Verification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame([
            {
                "laptop_model_id": 1,
                "laptop_name": "Office UltraBook 13",
                "price": 14000000,
                "ram_gb": 8,
                "storage_gb": 256,
                "laptop_weight": 1.1,
                "office_battery_minutes_final": 500,
                "gpu_name": "Intel Iris Xe",
                "relevance_score": 0.85,
                "final_relevance_score": 0.85,
                "AI_Score": 0.85,
            },
            {
                "laptop_model_id": 2,
                "laptop_name": "Gaming Pro 15",
                "price": 28000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 2.3,
                "office_battery_minutes_final": 240,
                "gpu_name": "NVIDIA GeForce RTX 4060",
                "relevance_score": 0.92,
                "final_relevance_score": 0.92,
                "AI_Score": 0.92,
            },
            {
                "laptop_model_id": 3,
                "laptop_name": "All-Rounder 14",
                "price": 19000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.5,
                "office_battery_minutes_final": 420,
                "gpu_name": "Intel Iris Xe",
                "relevance_score": 0.89,
                "final_relevance_score": 0.89,
                "AI_Score": 0.89,
            },
        ])

    def test_01_price_hard_feasible(self):
        """1. Price hard feasible: ngân sách <= 15tr -> chọn Office UltraBook 13 (id 1, 14tr)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=15000000, type="hard")
            ]
        )
        res = solve(self.df, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 1)
        self.assertFalse(res["has_soft_violation"])
        self.assertEqual(res["hard_violations"], [])

    def test_02_ram_hard_feasible(self):
        """2. RAM hard feasible: RAM >= 16GB -> chọn laptop 2 (score 0.92 cao nhất trong tập RAM 16GB)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard")
            ]
        )
        res = solve(self.df, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 2)
        self.assertFalse(res["has_soft_violation"])

    def test_03_gpu_hard_feasible(self):
        """3. GPU hard feasible: yêu cầu card rời -> chọn Gaming Pro 15 (id 2, RTX 4060)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="gpu_discrete", operator="=", value=True, type="hard")
            ]
        )
        res = solve(self.df, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 2)
        self.assertFalse(res["has_soft_violation"])

    def test_04_multiple_hard_constraints(self):
        """4. Multiple hard constraints: Price <= 20tr AND RAM >= 16GB AND Weight <= 1.6kg AND Battery >= 400min."""
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=20000000, type="hard"),
                Constraint(field="ram_gb", operator=">=", value=16, type="hard"),
                Constraint(field="weight_kg", operator="<=", value=1.6, type="hard"),
                Constraint(field="battery_minutes", operator=">=", value=400, type="hard"),
            ]
        )
        # Candidate retriever lọc đúng
        retrieved = retrieve_candidates(self.df, req)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved.iloc[0]["laptop_model_id"], 3)

        res = solve(retrieved, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 3)
        self.assertFalse(res["has_soft_violation"])
        self.assertEqual(res["hard_violations"], [])

    def test_05_price_soft(self):
        """5. Price soft: ngân sách mềm <= 18tr nhưng cần RAM >= 16GB -> chọn laptop 3 (19tr)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard"),
                Constraint(field="price", operator="<=", value=18000000, type="soft"),
            ]
        )
        # Soft constraint không bị retriever lọc
        retrieved = retrieve_candidates(self.df, req)
        self.assertEqual(len(retrieved), 2)

        res = solve(retrieved, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 3)
        self.assertTrue(res["has_soft_violation"])
        self.assertEqual(len(res["soft_violations"]), 1)
        self.assertEqual(res["soft_violations"][0]["field"], "price")
        self.assertIn("1,000,000", res["soft_violations"][0]["violation"])

    def test_06_weight_soft(self):
        """6. Weight soft: cân nặng mềm <= 1.0kg -> chọn laptop 1 (1.1kg) kèm soft violation."""
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=15000000, type="hard"),
                Constraint(field="weight_kg", operator="<=", value=1.0, type="soft"),
            ]
        )
        retrieved = retrieve_candidates(self.df, req)
        self.assertEqual(len(retrieved), 1)

        res = solve(retrieved, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 1)
        self.assertTrue(res["has_soft_violation"])
        self.assertEqual(res["soft_violations"][0]["field"], "weight_kg")
        self.assertIn("0.10kg", res["soft_violations"][0]["violation"])

    def test_07_battery_soft(self):
        """7. Battery soft: pin mềm >= 600 phút -> chọn laptop 1 (500 phút) kèm soft violation."""
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=15000000, type="hard"),
                Constraint(field="battery_minutes", operator=">=", value=600, type="soft"),
            ]
        )
        res = solve(self.df, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 1)
        self.assertTrue(res["has_soft_violation"])
        self.assertEqual(res["soft_violations"][0]["field"], "battery_minutes")
        self.assertIn("100 phút", res["soft_violations"][0]["violation"])

    def test_08_hard_and_soft_combined(self):
        """8. Hard + Soft cùng lúc: Hard GPU Discrete + Hard RAM 16GB + Soft Price <= 25tr -> chọn laptop 2 (28tr)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="gpu_discrete", operator="=", value=True, type="hard"),
                Constraint(field="ram_gb", operator=">=", value=16, type="hard"),
                Constraint(field="price", operator="<=", value=25000000, type="soft"),
            ]
        )
        retrieved = retrieve_candidates(self.df, req)
        self.assertEqual(len(retrieved), 1)  # laptop 2 thỏa hard constraints

        res = solve(retrieved, req)
        self.assertTrue(res["is_feasible"])
        self.assertEqual(res["laptop_id"], 2)
        self.assertTrue(res["has_soft_violation"])
        self.assertEqual(res["soft_violations"][0]["field"], "price")
        self.assertIn("3,000,000", res["soft_violations"][0]["violation"])

    def test_09_hard_infeasible(self):
        """9. Hard infeasible: không có laptop thỏa hard constraints -> Fallback 1 nearest alternative, status RELAXED, is_feasible False."""
        rec_res = recommend(
            query="Cần tìm laptop giá đúng 5 triệu có card rời RTX 4060",
            df=self.df,
            use_gemini_nlu=False,
        )
        self.assertIn("optimization", rec_res)
        opt = rec_res["optimization"]

        self.assertEqual(opt["status"], "RELAXED")
        self.assertFalse(opt["is_feasible"])
        self.assertIsNotNone(opt["laptop_id"])
        self.assertEqual(opt["laptop_id"], 2)  # Laptop 2 có RTX 4060, gần nhất với yêu cầu
        self.assertIsInstance(opt["hard_violations"], list)
        self.assertGreater(len(opt["hard_violations"]), 0)
        self.assertTrue(opt["has_soft_violation"])

        self.assertEqual(len(rec_res["recommendations"]), 1)
        self.assertEqual(rec_res["recommendations"][0]["type"], "nearest_alternative")
        self.assertIsNotNone(rec_res["recommended_laptop"])
        self.assertEqual(rec_res["recommended_laptop"]["laptop_model_id"], 2)

        # Chat API reply an toàn
        from app.api.routes.chat import _build_reply
        reply = _build_reply(rec_res["requirements"], opt, rec_res["recommended_laptop"])
        self.assertIn("Thông báo tiêu chí", reply)
        self.assertIn("Nghiệm nới lỏng ràng buộc", reply)

    def test_10_feasible_with_soft_violation(self):
        """10. Feasible + Soft violation: Thỏa hard constraint nhưng có soft violation -> is_feasible = True, has_soft_violation = True."""
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard"),
                Constraint(field="price", operator="<=", value=16000000, type="soft"),  # Soft: < 16tr
            ]
        )
        res = solve(self.df, req)
        self.assertTrue(res["is_feasible"], "Nghiệm phải feasible vì thỏa 100% hard constraints")
        self.assertEqual(res["laptop_id"], 3)
        self.assertTrue(res["has_soft_violation"], "Phải đánh dấu có soft violation")
        self.assertEqual(len(res["hard_violations"]), 0, "Không được có hard violation")
        self.assertEqual(len(res["soft_violations"]), 1)


if __name__ == "__main__":
    unittest.main()
