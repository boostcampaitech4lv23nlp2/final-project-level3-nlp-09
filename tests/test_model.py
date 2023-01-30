import json
import unittest

from src.model import CLIP, build_model


class TestModel(unittest.TestCase):
    def test_build_model(self):
        with open("src/model_configs/baseline.json") as f:
            configs = json.load(f)
        text_cfg = configs["text_cfg"]
        vision_cfg = configs["vision_cfg"]

        model = build_model(vision_cfg, text_cfg)
        self.assertEqual(type(model), CLIP)
