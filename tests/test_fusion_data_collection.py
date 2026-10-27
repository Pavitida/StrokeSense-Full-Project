import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.fusion import data_collection as dc


class TestFusionDataCollection(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "fusion_test.csv"

        patcher = patch.object(dc, "DATA_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_sample(self):
        result = dc.save_fusion_sample(
            "TEST_001", 0.8, 0.6, 0.7,
            label=None,
            consent=True,
        )

        self.assertEqual(result["status"], "saved")

        with self.path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "")

    def test_duplicate_rejected(self):
        dc.save_fusion_sample(
            "TEST_001", 0.8, 0.6, 0.7,
            consent=True,
        )

        with self.assertRaises(ValueError):
            dc.save_fusion_sample(
                "TEST_001", 0.8, 0.6, 0.7,
                consent=True,
            )

    def test_missing_consent_rejected(self):
        with self.assertRaises(ValueError):
            dc.save_fusion_sample(
                "TEST_002", 0.8, 0.6, 0.7,
                consent=False,
            )

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            dc.save_fusion_sample(
                "TEST_003", 1.5, 0.6, 0.7,
                consent=True,
            )


if __name__ == "__main__":
    unittest.main()
