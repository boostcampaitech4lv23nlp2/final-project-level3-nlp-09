import argparse
import json
import unittest

from src.dataset import FoodImageDataset
from src.preprocess import image_transform


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", default="tests/sample_data", type=str)
        parser.add_argument("--train_info_file_name", default="sample_dataset.json", type=str)
        parser.add_argument("--test_info_file_name", default="sample_dataset.json", type=str)
        parser.add_argument("--labels_info_file_name", default="sample_label.json", type=str)
        self.args = parser.parse_args()

        with open("src/model_configs/baseline.json") as f:
            configs = json.load(f)
        vision_cfg = configs["vision_cfg"]

        self.preprocess = image_transform(vision_cfg["image_size"], is_train=True)

        return super().setUp()

    def test_load_train_dataset(self):
        dataset = FoodImageDataset(self.args, self.preprocess, mode="train", ratio=1)
        print(dataset.__len__)

    def test_load_test_dataset(self):
        dataset = FoodImageDataset(self.args, self.preprocess, mode="test", ratio=0.1)
        print(dataset.__len__)

    def test_check_ratio_range(self):
        dataset = FoodImageDataset(self.args, self.preprocess, mode="train", ratio=2)
        print(dataset.__len__)
