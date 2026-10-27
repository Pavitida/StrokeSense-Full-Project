import unittest
from unittest.mock import patch

from src.fusion.fusion_model import predict_fusion


class TestFusionBackend(unittest.TestCase):

    @patch("src.fusion.fusion_model.MODEL_PATH")
    def test_missing_model(self, model_path):
        model_path.exists.return_value = False
        result = predict_fusion(0.8, 0.6, 0.7)
        self.assertEqual(result["status"], "model_not_trained")
        self.assertIsNone(result["score"])

    def test_invalid_scores(self):
        for scores in [
            (-0.1, 0.5, 0.5),
            (0.5, 1.2, 0.5),
            (float("nan"), 0.5, 0.5),
        ]:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    predict_fusion(*scores)


if __name__ == "__main__":
    unittest.main()
