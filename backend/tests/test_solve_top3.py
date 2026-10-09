"""
Unit Test Suite cho Top-3 Recommendation Optimization:
Google OR-Tools CP-SAT Solver (solve_top3 & compute_performance_scores trong ortools_solver.py).

Bao phủ các kịch bản kiểm thử:
1. Đủ 3 máy (Top 1: best_match, Top 2: budget_alternative, Top 3: performance_alternative).
2. Chỉ đủ 2 máy (Tập ứng viên chỉ có 2 máy hoặc máy 3 không đạt utility threshold).
3. Chỉ đủ 1 máy (Tập ứng viên chỉ có 1 máy hoặc các máy khác không thỏa hard constraint / utility threshold).
4. Đảm bảo không trùng lặp laptop giữa các rank (rank 1, 2, 3 hoàn toàn khác nhau).
5. Trường hợp không tìm thấy laptop nào (Infeasible).
6. Kiểm thử hàm compute_performance_scores (đầy đủ benchmark, thiếu benchmark, has_geekbench_data=False).
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
import numpy as np

from app.nlp.schema import Constraint, RequirementSet
from app.optimizer.ortools_solver import (
    solve,
    solve_top3,
    compute_performance_scores,
)


class TestSolveTop3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Tạo tập DataFrame phong phú với 5 laptops
        cls.df = pd.DataFrame([
            {
                "laptop_model_id": 1,
                "laptop_name": "Premium Flagship 14",
                "price": 30000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.3,
                "office_battery_minutes_final": 480,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 2200,
                "geekbench_cpu_multi": 9000,
                "has_geekbench_data": True,
                "relevance_score": 0.95,
                "final_relevance_score": 0.95,
            },
            {
                "laptop_model_id": 2,
                "laptop_name": "Budget Value 14",
                "price": 15000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.4,
                "office_battery_minutes_final": 420,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 1800,
                "geekbench_cpu_multi": 7000,
                "has_geekbench_data": True,
                "relevance_score": 0.90,  # 0.90 / 0.95 = 94.7% >= 90%
                "final_relevance_score": 0.90,
            },
            {
                "laptop_model_id": 3,
                "laptop_name": "Powerhouse Monster 16",
                "price": 35000000,
                "ram_gb": 32,
                "storage_gb": 1024,
                "laptop_weight": 2.2,
                "office_battery_minutes_final": 300,
                "gpu_name": "NVIDIA GeForce RTX 4070",
                "geekbench_cpu_single": 2800,
                "geekbench_cpu_multi": 14000,
                "has_geekbench_data": True,
                "relevance_score": 0.85,  # 0.85 / 0.95 = 89.4% >= 80%
                "final_relevance_score": 0.85,
            },
            {
                "laptop_model_id": 4,
                "laptop_name": "Mid-tier Balanced 15",
                "price": 22000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.7,
                "office_battery_minutes_final": 360,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 2000,
                "geekbench_cpu_multi": 8000,
                "has_geekbench_data": True,
                "relevance_score": 0.82,
                "final_relevance_score": 0.82,
            },
            {
                "laptop_model_id": 5,
                "laptop_name": "Ultra Low Cost 14",
                "price": 10000000,
                "ram_gb": 8,
                "storage_gb": 256,
                "laptop_weight": 1.6,
                "office_battery_minutes_final": 300,
                "gpu_name": "Intel UHD Graphics",
                "geekbench_cpu_single": 1200,
                "geekbench_cpu_multi": 4000,
                "has_geekbench_data": True,
                "relevance_score": 0.60,  # 0.60 / 0.95 = 63% < 80%
                "final_relevance_score": 0.60,
            },
        ])

    def test_01_performance_score_calculation(self):
        """Kiểm tra tính điểm performance_score chuẩn hóa: 0.35 * single + 0.65 * multi."""
        perf = compute_performance_scores(self.df)
        self.assertEqual(len(perf), 5)
        # Laptop 3 có single (2800) và multi (14000) cao nhất -> performance_score = 1.0
        self.assertAlmostEqual(perf.iloc[2], 1.0, places=2)
        # Laptop 5 có single (1200) và multi (4000) thấp nhất -> performance_score = 0.0
        self.assertAlmostEqual(perf.iloc[4], 0.0, places=2)

        # Kiểm tra máy không có dữ liệu benchmark
        df_missing = self.df.copy()
        df_missing.loc[0, "has_geekbench_data"] = False
        df_missing.loc[1, "geekbench_cpu_single"] = np.nan
        df_missing.loc[1, "geekbench_cpu_multi"] = np.nan
        perf_missing = compute_performance_scores(df_missing)
        self.assertEqual(perf_missing.iloc[0], 0.0)
        self.assertEqual(perf_missing.iloc[1], 0.0)

    def test_02_solve_top3_full_three_recommendations(self):
        """1. Đủ 3 máy: Top 1 (best utility), Top 2 (min price, utility >= 90%), Top 3 (max perf, utility >= 80%)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard")
            ]
        )
        res = solve_top3(self.df, req)

        self.assertTrue(res["is_feasible"])
        self.assertIn(res["status"], ("OPTIMAL", "FEASIBLE"))
        recs = res["recommendations"]
        self.assertEqual(len(recs), 3)

        # Top 1: best_match -> Laptop 1 (relevance 0.95)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[0]["type"], "best_match")
        self.assertEqual(recs[0]["laptop_id"], 1)
        self.assertEqual(recs[0]["utility_score"], 0.95)

        # Top 2: budget_alternative -> Loại id 1, utility >= 0.95*0.9=0.855 -> id 2 (relevance 0.90, price 15tr)
        self.assertEqual(recs[1]["rank"], 2)
        self.assertEqual(recs[1]["type"], "budget_alternative")
        self.assertEqual(recs[1]["laptop_id"], 2)
        self.assertEqual(recs[1]["price"], 15000000)

        # Top 3: performance_alternative -> Loại id 1, id 2, utility >= 0.95*0.8=0.76 -> id 3 (relevance 0.85, max perf 1.0)
        self.assertEqual(recs[2]["rank"], 3)
        self.assertEqual(recs[2]["type"], "performance_alternative")
        self.assertEqual(recs[2]["laptop_id"], 3)
        self.assertAlmostEqual(recs[2]["performance_score"], 1.0, places=2)

    def test_02b_budget_alternative_is_strictly_cheaper_than_top1(self):
        """Rank 2 chỉ được gọi là budget_alternative khi thực sự rẻ hơn Rank 1."""
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard")
            ]
        )
        res = solve_top3(self.df, req)
        recs = res["recommendations"]

        budget = next((r for r in recs if r["type"] == "budget_alternative"), None)
        self.assertIsNotNone(budget)
        self.assertLess(budget["price"], recs[0]["price"])

    def test_02c_no_misleading_budget_if_all_remaining_are_more_expensive(self):
        """Nếu không có máy rẻ hơn Top 1 thì không được gắn nhãn 'budget_alternative' sai nghĩa."""
        df = pd.DataFrame([
            {
                "laptop_model_id": 1,
                "price": 10000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.3,
                "office_battery_minutes_final": 480,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 2200,
                "geekbench_cpu_multi": 9000,
                "has_geekbench_data": True,
                "relevance_score": 0.95,
                "final_relevance_score": 0.95,
            },
            {
                "laptop_model_id": 2,
                "price": 15000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.4,
                "office_battery_minutes_final": 420,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 1800,
                "geekbench_cpu_multi": 7000,
                "has_geekbench_data": True,
                "relevance_score": 0.90,
                "final_relevance_score": 0.90,
            },
            {
                "laptop_model_id": 3,
                "price": 20000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.5,
                "office_battery_minutes_final": 400,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 2400,
                "geekbench_cpu_multi": 10000,
                "has_geekbench_data": True,
                "relevance_score": 0.85,
                "final_relevance_score": 0.85,
            },
        ])
        res = solve_top3(df, RequirementSet())
        self.assertTrue(res["is_feasible"])
        self.assertFalse(any(r["type"] == "budget_alternative" for r in res["recommendations"]))

    def test_03_no_duplicates_among_ranks(self):
        """4. Không trùng lặp laptop giữa các rank."""
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=40000000, type="hard")
            ]
        )
        res = solve_top3(self.df, req)
        self.assertTrue(res["is_feasible"])
        recs = res["recommendations"]
        self.assertGreaterEqual(len(recs), 2)

        laptop_ids = [r["laptop_id"] for r in recs]
        self.assertEqual(len(laptop_ids), len(set(laptop_ids)), "Các rank không được trùng laptop_id!")

    def test_04_only_two_recommendations_feasible(self):
        """2. Chỉ đủ 2 máy: Tập ứng viên chỉ có 2 máy thỏa yêu cầu."""
        # Chỉ lấy 2 máy
        df_2 = self.df.iloc[:2].copy()
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard")
            ]
        )
        res = solve_top3(df_2, req)
        self.assertTrue(res["is_feasible"])
        recs = res["recommendations"]
        self.assertEqual(len(recs), 2, "Chỉ được trả về đúng 2 recommendations khi tập ứng viên có 2 máy.")
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[1]["rank"], 2)
        self.assertNotEqual(recs[0]["laptop_id"], recs[1]["laptop_id"])

    def test_05_only_one_recommendation_feasible(self):
        """3. Chỉ đủ 1 máy: Tập ứng viên chỉ có 1 máy thỏa yêu cầu."""
        df_1 = self.df.iloc[:1].copy()
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=16, type="hard")
            ]
        )
        res = solve_top3(df_1, req)
        self.assertTrue(res["is_feasible"])
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1, "Chỉ được trả về đúng 1 recommendation khi tập ứng viên có 1 máy.")
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[0]["type"], "best_match")
        self.assertEqual(recs[0]["laptop_id"], 1)

    def test_06_infeasible_empty_or_no_match(self):
        """5. Infeasible: Không có máy nào thỏa mãn hard constraints -> Trả về rỗng, không crash."""
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=64, type="hard")
            ]
        )
        res = solve_top3(self.df, req)
        self.assertFalse(res["is_feasible"])
        self.assertEqual(res["status"], "INFEASIBLE")
        self.assertEqual(res["recommendations"], [])

        # Kiểm tra DataFrame rỗng
        res_empty = solve_top3(pd.DataFrame(), req)
        self.assertFalse(res_empty["is_feasible"])
        self.assertEqual(res_empty["recommendations"], [])

    def test_07_solve_backward_compatibility(self):
        """Kiểm tra hàm solve() gốc vẫn hoạt động 100% bình thường."""
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=20000000, type="hard")
            ]
        )
        res = solve(self.df, req)
        self.assertTrue(res["is_feasible"])
        self.assertIn("laptop_id", res)
        self.assertEqual(res["laptop_id"], 2)

    def test_08_soft_gte_satisfied_no_violation(self):
        """soft >= satisfied -> không violation (không tạo violation âm)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="battery_minutes", operator=">=", value=300, type="soft")
            ]
        )
        res = solve_top3(self.df, req)
        self.assertTrue(res["is_feasible"])
        top1 = res["recommendations"][0]
        # Laptop 1 có pin 480 phút >= 300 phút -> thỏa mãn, không có violation
        self.assertFalse(top1["has_soft_violation"])
        self.assertEqual(top1["soft_violations"], [])

    def test_09_soft_lte_satisfied_no_violation(self):
        """soft <= satisfied -> không violation (không tạo violation âm)."""
        req = RequirementSet(
            constraints=[
                Constraint(field="weight_kg", operator="<=", value=1.5, type="soft")
            ]
        )
        res = solve_top3(self.df, req)
        self.assertTrue(res["is_feasible"])
        top1 = res["recommendations"][0]
        # Laptop 1 nặng 1.3kg <= 1.5kg -> thỏa mãn, không có violation
        self.assertFalse(top1["has_soft_violation"])
        self.assertEqual(top1["soft_violations"], [])

    def test_10_nearest_alternative_only_evaluates_hard_constraints(self):
        """nearest alternative chỉ tính hard constraints, bỏ qua soft constraints."""
        from app.optimizer.ortools_solver import solve_nearest_alternative
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=8000000, type="hard"),
                Constraint(field="weight_kg", operator="<=", value=1.0, type="soft"),
            ]
        )
        res = solve_nearest_alternative(self.df, req)
        self.assertEqual(res["status"], "RELAXED")
        self.assertFalse(res["is_feasible"])
        self.assertFalse(res["has_soft_violation"])
        self.assertEqual(res["soft_violations"], [])
        # Toàn bộ violation ghi nhận chỉ xuất phát từ hard constraint (price)
        violated_fields = [v["field"] for v in res["violations"]]
        self.assertIn("price", violated_fields)
        self.assertNotIn("weight_kg", violated_fields)


class TestRequirementMergeSemantics(unittest.TestCase):
    """Kiểm tra chuyển nhu cầu mới không bị giữ tag/GPU cũ từ session."""

    def test_office_query_replaces_old_gaming_context(self):
        from app.recommendation.pipeline import _merge_requirements

        old = {
            "constraints": [
                {"field": "gpu_discrete", "operator": "=", "value": True, "type": "hard"},
                {"field": "gpu_keyword", "operator": "=", "value": "RTX 4060", "type": "hard"},
                {"field": "price", "operator": "<=", "value": 25000000, "type": "hard"},
            ],
            "preferences": [],
            "required_tags": ["is_gaming_friendly"],
        }
        new = {
            "constraints": [
                {"field": "price", "operator": "<=", "value": 15000000, "type": "soft"},
            ],
            "preferences": [],
            "required_tags": ["is_office_friendly"],
        }

        merged = _merge_requirements(old, new)
        fields = {c["field"] for c in merged["constraints"]}

        self.assertEqual(merged["required_tags"], ["is_office_friendly"])
        self.assertNotIn("gpu_discrete", fields)
        self.assertNotIn("gpu_keyword", fields)

class TestPipelineTop3Integration(unittest.TestCase):
    """
    Kiểm thử tích hợp Top-3 Recommendation vào Pipeline (recommend() trong pipeline.py).
    """

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame([
            {
                "laptop_model_id": 1,
                "laptop_name": "Pro Flagship 14",
                "price": 30000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.3,
                "office_battery_minutes_final": 480,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 2200,
                "geekbench_cpu_multi": 9000,
                "has_geekbench_data": True,
                "relevance_score": 0.95,
                "final_relevance_score": 0.95,
                "AI_Score": 0.95,
            },
            {
                "laptop_model_id": 2,
                "laptop_name": "Economy Work 14",
                "price": 15000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.4,
                "office_battery_minutes_final": 420,
                "gpu_name": "Intel Iris Xe",
                "geekbench_cpu_single": 1800,
                "geekbench_cpu_multi": 7000,
                "has_geekbench_data": True,
                "relevance_score": 0.90,
                "final_relevance_score": 0.90,
                "AI_Score": 0.90,
            },
            {
                "laptop_model_id": 3,
                "laptop_name": "High Perf 16",
                "price": 35000000,
                "ram_gb": 32,
                "storage_gb": 1024,
                "laptop_weight": 2.2,
                "office_battery_minutes_final": 300,
                "gpu_name": "NVIDIA GeForce RTX 4070",
                "geekbench_cpu_single": 2800,
                "geekbench_cpu_multi": 14000,
                "has_geekbench_data": True,
                "relevance_score": 0.85,
                "final_relevance_score": 0.85,
                "AI_Score": 0.85,
            },
        ])

    def test_01_pipeline_returns_three_laptops(self):
        """1. Pipeline trả đủ 3 máy khi có đủ 3 ứng viên thỏa mãn."""
        from app.recommendation.pipeline import recommend
        res = recommend(
            query="Cần tìm laptop RAM 16GB",
            df=self.df,
            use_gemini_nlu=False,
        )

        # Kiểm tra đầy đủ các keys trong schema kết quả
        self.assertIn("query", res)
        self.assertIn("requirements", res)
        self.assertIn("candidates_count", res)
        self.assertIn("optimization", res)
        self.assertIn("recommendations", res)
        self.assertIn("recommended_laptops", res)
        self.assertIn("recommended_laptop", res)

        recs = res["recommendations"]
        rec_laptops = res["recommended_laptops"]

        self.assertEqual(len(recs), 3)
        self.assertEqual(len(rec_laptops), 3)
        self.assertEqual(res["optimization"]["status"], "OPTIMAL")

        # Kiểm tra backward compatibility: recommended_laptop == recommended_laptops[0]
        self.assertIsNotNone(res["recommended_laptop"])
        self.assertEqual(res["recommended_laptop"]["laptop_model_id"], rec_laptops[0]["laptop_model_id"])
        self.assertEqual(res["recommended_laptop"]["laptop_model_id"], recs[0]["laptop_id"])

        # Kiểm tra không trùng lặp laptop_id
        laptop_ids = [l["laptop_model_id"] for l in rec_laptops]
        self.assertEqual(len(laptop_ids), len(set(laptop_ids)))

    def test_02_pipeline_returns_two_laptops(self):
        """2. Pipeline trả đúng 2 máy khi chỉ có 2 ứng viên."""
        from app.recommendation.pipeline import recommend
        df_2 = self.df.iloc[:2].copy()
        res = recommend(
            query="Cần tìm laptop RAM 16GB",
            df=df_2,
            use_gemini_nlu=False,
        )

        recs = res["recommendations"]
        rec_laptops = res["recommended_laptops"]
        self.assertEqual(len(recs), 2)
        self.assertEqual(len(rec_laptops), 2)
        self.assertEqual(res["recommended_laptop"]["laptop_model_id"], rec_laptops[0]["laptop_model_id"])

        laptop_ids = [l["laptop_model_id"] for l in rec_laptops]
        self.assertEqual(len(laptop_ids), len(set(laptop_ids)))

    def test_03_pipeline_returns_one_laptop(self):
        """3. Pipeline trả đúng 1 máy khi chỉ có 1 ứng viên."""
        from app.recommendation.pipeline import recommend
        df_1 = self.df.iloc[:1].copy()
        res = recommend(
            query="Cần tìm laptop RAM 16GB",
            df=df_1,
            use_gemini_nlu=False,
        )

        recs = res["recommendations"]
        rec_laptops = res["recommended_laptops"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(len(rec_laptops), 1)
        self.assertEqual(res["recommended_laptop"]["laptop_model_id"], rec_laptops[0]["laptop_model_id"])
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[0]["type"], "best_match")

    def test_04_pipeline_fallback_nearest_alternative(self):
        """4. Pipeline trả đúng 1 nearest alternative khi hard constraints không có ứng viên."""
        from app.recommendation.pipeline import recommend
        res = recommend(
            query="Cần tìm laptop RAM 64GB",
            df=self.df,
            use_gemini_nlu=False,
        )

        self.assertEqual(res["candidates_count"], 0)
        opt = res["optimization"]
        self.assertEqual(opt["status"], "RELAXED")
        self.assertFalse(opt["is_feasible"])
        self.assertIn("violations", opt)
        self.assertGreater(len(opt["violations"]), 0)

        # Trả về đúng 1 recommendation duy nhất (nearest_alternative)
        recs = res["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[0]["type"], "nearest_alternative")
        self.assertIn("violations", recs[0])
        self.assertGreater(len(recs[0]["violations"]), 0)

        # recommended_laptops có đúng 1 máy, recommended_laptop == recommended_laptops[0]
        rec_laptops = res["recommended_laptops"]
        self.assertEqual(len(rec_laptops), 1)
        self.assertIsNotNone(res["recommended_laptop"])
        self.assertEqual(res["recommended_laptop"]["laptop_model_id"], rec_laptops[0]["laptop_model_id"])
        self.assertEqual(res["recommended_laptop"]["laptop_model_id"], 3)  # Laptop 3 có 32GB RAM gần 64GB nhất

    def test_05_pipeline_empty_dataset(self):
        """5. Pipeline xử lý an toàn khi dataset rỗng."""
        from app.recommendation.pipeline import recommend
        res = recommend(
            query="Cần tìm laptop bất kỳ",
            df=pd.DataFrame(),
            use_gemini_nlu=False,
        )
        self.assertEqual(res["candidates_count"], 0)
        self.assertEqual(res["optimization"]["status"], "INFEASIBLE")
        self.assertFalse(res["optimization"]["is_feasible"])
        self.assertEqual(res["recommendations"], [])
        self.assertEqual(res["recommended_laptops"], [])
        self.assertIsNone(res["recommended_laptop"])

    def test_06_solve_nearest_alternative_direct(self):
        """6. Kiểm thử trực tiếp hàm solve_nearest_alternative với các vi phạm ràng buộc."""
        from app.optimizer.ortools_solver import solve_nearest_alternative
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=12000000, type="hard"),
                Constraint(field="gpu_discrete", operator="=", value=True, type="hard"),
            ]
        )
        res = solve_nearest_alternative(self.df, req)
        self.assertEqual(res["status"], "RELAXED")
        self.assertFalse(res["is_feasible"])
        self.assertIsNotNone(res["laptop_id"])
        self.assertEqual(len(res["recommendations"]), 1)
        self.assertEqual(res["recommendations"][0]["type"], "nearest_alternative")
        self.assertGreater(len(res["violations"]), 0)


if __name__ == "__main__":
    unittest.main()

