"""
Unit Test Suite cho API Endpoint /api/chat (chat.py):
1. Response có trường recommendations
2. Response có trường recommended_laptops
3. Backward compatibility: laptop_details == recommended_laptops[0]
4. Soft violation vẫn giữ is_feasible = True (và is_relaxed = True)
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app


class TestChatApiTop3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_chat_returns_top3_recommendations(self):
        """1. /api/chat forward recommendations và recommended_laptops."""
        res = self.client.post(
            "/api/chat",
            json={"message": "Tư vấn laptop văn phòng giá tầm 15 đến 20 triệu"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Kiểm tra các trường Top-3 mới
        self.assertIn("recommendations", data)
        self.assertIn("recommended_laptops", data)
        self.assertIn("result", data)
        self.assertIn("laptop_details", data)

        recs = data.get("recommendations")
        rec_laptops = data.get("recommended_laptops")

        if recs and len(recs) > 0:
            self.assertIsInstance(recs, list)
            self.assertIsInstance(rec_laptops, list)
            self.assertEqual(len(recs), len(rec_laptops))

            # Kiểm tra backward compatibility: laptop_details trỏ tới recommended_laptops[0]
            self.assertIsNotNone(data["laptop_details"])
            self.assertEqual(
                data["laptop_details"]["laptop_model_id"],
                rec_laptops[0]["laptop_model_id"]
            )
            self.assertEqual(
                data["result"]["laptop_id"],
                rec_laptops[0]["laptop_model_id"]
            )

    def test_02_soft_violation_keeps_is_feasible_true(self):
        """2. Soft violation KHÔNG được làm is_feasible=False (is_feasible=True, is_relaxed=True)."""
        # Query có thể gây soft violation (ví dụ pin trâu hoặc giá thấp)
        res = self.client.post(
            "/api/chat",
            json={"message": "Tìm laptop gaming dưới 15 triệu"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        result = data.get("result")

        if result and result.get("is_relaxed"):
            # Khi có soft violation (relaxed), is_feasible VẪN PHẢI LÀ True
            self.assertTrue(
                result.get("is_feasible"),
                "Khi có soft violation, is_feasible vẫn phải là True vì thỏa mãn 100% hard constraints."
            )
            self.assertTrue(result.get("is_relaxed"))

    def test_03_chat_reset_session(self):
        """3. Reset session hoạt động đúng."""
        session_id = "test-session-top3-123"
        res = self.client.post(
            "/api/chat",
            json={"message": "Laptop giá 20 triệu", "session_id": session_id},
        )
        self.assertEqual(res.status_code, 200)

        del_res = self.client.delete(f"/api/chat/{session_id}")
        self.assertEqual(del_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
