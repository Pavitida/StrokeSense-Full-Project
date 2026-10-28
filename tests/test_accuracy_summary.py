import unittest
from src.accuracy_summary import summarize

class TestAccuracySummary(unittest.TestCase):
    def test_denominator_threshold_and_split(self):
        rows = [
            dict(modality="speech", expected="dysarthria", score=0.5, status="success", split="test speaker per repository split"),
            dict(modality="speech", expected="healthy", score=0.9, status="success", split="test speaker per repository split"),
            dict(modality="speech", expected="healthy", status="error", split="test speaker per repository split"),
            dict(modality="speech", expected="healthy", score=float("nan"), status="success", split="test speaker per repository split"),
            dict(modality="face", expected="affected", score=0.8, status="success", split="unknown"),
        ]
        result={r["modality"]:r for r in summarize(rows)}
        self.assertEqual(result["speech"]["accuracy_percent"],50)
        self.assertEqual(result["speech"]["failed_or_unscorable"],2)
        self.assertEqual(result["speech"]["metric"],"Test Accuracy (subset)")
        self.assertIn("unverified",result["face"]["metric"])
        self.assertIsNone(result["movement"]["accuracy_percent"])

    def test_mixed_splits_not_test_accuracy(self):
        rows=[dict(modality="movement",expected="Complete",score=0.1,status="success",split="Train")]
        self.assertIn("unverified",summarize(rows)[2]["metric"])

if __name__ == "__main__": unittest.main()
