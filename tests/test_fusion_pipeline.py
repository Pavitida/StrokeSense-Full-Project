import unittest
from unittest.mock import patch

import sys
import types

# Stub only the modality imports before loading the pipeline.
for name, function in [
    ("face", "predict_face"),
    ("speech", "predict_speech"),
    ("movement", "predict_movement"),
]:
    module_name = f"src.{name}.{name}_inference"
    stub = types.ModuleType(module_name)
    setattr(stub, function, lambda *args, **kwargs: None)
    sys.modules[module_name] = stub

from src.fusion.run_pipeline import run_pipeline


class TestFusionPipeline(unittest.TestCase):

    @patch("src.fusion.run_pipeline.predict_fusion")
    @patch("src.fusion.run_pipeline.predict_movement")
    @patch("src.fusion.run_pipeline.predict_speech")
    @patch("src.fusion.run_pipeline.predict_face")
    def test_successful_pipeline(
        self, face, speech, movement, fusion
    ):
        face.return_value = {"status": "success", "score": 0.8}
        speech.return_value = {"status": "success", "score": 0.6}
        movement.return_value = {"status": "success", "score": 0.7}
        fusion.return_value = {
            "status": "success",
            "score": 0.75,
            "method": "logistic_regression",
        }

        result = run_pipeline("face.jpg", "speech.wav", "arms.mp4")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["fusion"]["score"], 0.75)
        fusion.assert_called_once_with(0.8, 0.6, 0.7)

    @patch("src.fusion.run_pipeline.predict_fusion")
    @patch("src.fusion.run_pipeline.predict_movement")
    @patch("src.fusion.run_pipeline.predict_speech")
    @patch("src.fusion.run_pipeline.predict_face")
    def test_untrained_fusion(
        self, face, speech, movement, fusion
    ):
        face.return_value = {"status": "success", "score": 0.8}
        speech.return_value = {"status": "success", "score": 0.6}
        movement.return_value = {"status": "success", "score": 0.7}
        fusion.return_value = {
            "status": "model_not_trained",
            "score": None,
        }

        result = run_pipeline("face.jpg", "speech.wav", "arms.mp4")

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["fusion"]["status"], "model_not_trained")
        self.assertIsNone(result["fusion"]["score"])


if __name__ == "__main__":
    unittest.main()
