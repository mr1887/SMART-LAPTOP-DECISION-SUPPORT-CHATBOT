"""
Test so sánh giữa Google OR-Tools strict solver (solve_hard_constraints) và legacy solver (solver.py)
trên các trường hợp chỉ chứa Hard Constraints:
1. budget feasible
2. RAM feasible
3. GPU feasible
4. infeasible query
"""

import sys
import unittest
from pathlib import Path

# Thêm thư mục backend vào sys.path để import app modules
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd

from app.nlp.schema import Constraint, RequirementSet
from app.optimizer import solver as legacy_solver
from app.optimizer.ortools_solver import solve_hard_constraints


class TestSolverComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Tạo tập dữ liệu laptop thử nghiệm tiêu chuẩn
        cls.df = pd.DataFrame([
            {
                "laptop_model_id": 101,
                "laptop_name": "Asus TUF Gaming A15",
                "price": 22000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 2.2,
                "office_battery_minutes_final": 300,
                "gpu_name": "NVIDIA GeForce RTX 4060",
                "AI_Score": 0.92,
                "is_gaming_friendly": True,
                "is_office_friendly": False,
            },
            {
                "laptop_model_id": 102,
                "laptop_name": "Acer Nitro V 15",
                "price": 18500000,
                "ram_gb": 8,
                "storage_gb": 512,
                "laptop_weight": 2.1,
                "office_battery_minutes_final": 250,
                "gpu_name": "NVIDIA GeForce RTX 3050",
                "AI_Score": 0.85,
                "is_gaming_friendly": True,
                "is_office_friendly": False,
            },
            {
                "laptop_model_id": 103,
                "laptop_name": "Dell Inspiron 14",
                "price": 15000000,
                "ram_gb": 16,
                "storage_gb": 512,
                "laptop_weight": 1.4,
                "office_battery_minutes_final": 480,
                "gpu_name": "Intel Iris Xe Graphics",
                "AI_Score": 0.88,
                "is_gaming_friendly": False,
                "is_office_friendly": True,
            },
            {
                "laptop_model_id": 104,
                "laptop_name": "MacBook Air M2",
                "price": 24000000,
                "ram_gb": 16,
                "storage_gb": 256,
                "laptop_weight": 1.24,
                "office_battery_minutes_final": 600,
                "gpu_name": "Apple M2 8-Core GPU",
                "AI_Score": 0.95,
                "is_gaming_friendly": False,
                "is_office_friendly": True,
            },
        ])

    def test_1_budget_feasible(self):
        """Case 1: Ngân sách <= 20 triệu (Budget feasible)."""
        max_price = 20000000
        req = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=max_price, type="hard")
            ]
        )
        legacy_constraints = {"max_price": max_price}

        # 1. OR-Tools solver
        ortools_res = solve_hard_constraints(self.df, req)
        self.assertTrue(ortools_res["is_feasible"], "OR-Tools phải tìm được nghiệm feasible.")
        self.assertIsNotNone(ortools_res["laptop_id"])

        chosen_ortools = self.df.loc[self.df["laptop_model_id"] == ortools_res["laptop_id"]].iloc[0]
        self.assertLessEqual(chosen_ortools["price"], max_price)

        # 2. Legacy solver
        legacy_res = legacy_solver.solve(legacy_constraints, self.df, backend="pulp")
        self.assertTrue(legacy_res["is_feasible"], "Legacy solver phải tìm được nghiệm feasible.")
        chosen_legacy = self.df.loc[self.df["laptop_model_id"] == legacy_res["laptop_id"]].iloc[0]
        self.assertLessEqual(chosen_legacy["price"], max_price)

        # Cả hai solver cùng tìm ra nghiệm tối ưu (Dell Inspiron 14, id 103, score 0.88)
        self.assertEqual(ortools_res["laptop_id"], legacy_res["laptop_id"])
        self.assertEqual(ortools_res["laptop_id"], 103)

    def test_2_ram_feasible(self):
        """Case 2: RAM >= 16GB (RAM feasible)."""
        min_ram = 16
        req = RequirementSet(
            constraints=[
                Constraint(field="ram_gb", operator=">=", value=min_ram, type="hard")
            ]
        )

        ortools_res = solve_hard_constraints(self.df, req)
        self.assertTrue(ortools_res["is_feasible"], "OR-Tools phải tìm được nghiệm feasible cho RAM >= 16GB.")
        chosen_ortools = self.df.loc[self.df["laptop_model_id"] == ortools_res["laptop_id"]].iloc[0]
        self.assertGreaterEqual(chosen_ortools["ram_gb"], min_ram)
        # Máy có AI_Score cao nhất có RAM >= 16 là MacBook Air M2 (id 104, score 0.95)
        self.assertEqual(ortools_res["laptop_id"], 104)

    def test_3_gpu_feasible(self):
        """Case 3: Card đồ họa rời (GPU discrete feasible) hoặc GPU keyword."""
        # 3a. Yêu cầu card rời
        req_discrete = RequirementSet(
            constraints=[
                Constraint(field="gpu_discrete", operator="=", value=True, type="hard")
            ]
        )
        legacy_constraints_disc = {"require_discrete_gpu": True}

        ortools_res = solve_hard_constraints(self.df, req_discrete)
        legacy_res = legacy_solver.solve(legacy_constraints_disc, self.df, backend="pulp")

        self.assertTrue(ortools_res["is_feasible"])
        self.assertTrue(legacy_res["is_feasible"])

        # Cả hai solver đều chọn máy có card rời và AI_Score cao nhất (Asus TUF, id 101, score 0.92)
        self.assertEqual(ortools_res["laptop_id"], legacy_res["laptop_id"])
        self.assertEqual(ortools_res["laptop_id"], 101)

        # 3b. Yêu cầu GPU keyword "RTX 3050"
        req_kw = RequirementSet(
            constraints=[
                Constraint(field="gpu_keyword", operator="=", value="RTX 3050", type="hard")
            ]
        )
        legacy_constraints_kw = {"gpu_keyword": "RTX 3050"}

        ortools_kw_res = solve_hard_constraints(self.df, req_kw)
        legacy_kw_res = legacy_solver.solve(legacy_constraints_kw, self.df, backend="pulp")

        self.assertTrue(ortools_kw_res["is_feasible"])
        self.assertTrue(legacy_kw_res["is_feasible"])
        self.assertEqual(ortools_kw_res["laptop_id"], 102)
        self.assertEqual(legacy_kw_res["laptop_id"], 102)

    def test_4_infeasible_query(self):
        """Case 4: Truy vấn không thể thỏa mãn đồng thời (Infeasible query)."""
        req_infeasible = RequirementSet(
            constraints=[
                Constraint(field="price", operator="<=", value=10000000, type="hard"),
                Constraint(field="gpu_discrete", operator="=", value=True, type="hard"),
            ]
        )
        legacy_constraints_inf = {
            "max_price": 10000000,
            "require_discrete_gpu": True,
        }

        # OR-Tools strict solver phải trả về infeasible
        ortools_res = solve_hard_constraints(self.df, req_infeasible)
        self.assertFalse(ortools_res["is_feasible"])
        self.assertIsNone(ortools_res["laptop_id"])
        self.assertEqual(ortools_res["status"], "INFEASIBLE")
        self.assertTrue(len(ortools_res["hard_violations"]) > 0)

        # Legacy solver ở chế độ strict (vòng 1) cũng vô nghiệm (is_feasible=False, is_relaxed=True)
        legacy_res = legacy_solver.solve(legacy_constraints_inf, self.df, backend="pulp")
        self.assertFalse(legacy_res["is_feasible"])
        self.assertTrue(legacy_res.get("is_relaxed", False))


if __name__ == "__main__":
    unittest.main()
