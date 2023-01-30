import argparse
import json
import unittest

from torch import Tensor

from src.dataset import FoodImageDataset
from src.preprocess import image_transform


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", default="tests/sample_data", type=str)
        parser.add_argument("--train_info_file_name", default="sample_dataset.json", type=str)
        parser.add_argument("--test_info_file_name", default="sample_dataset.json", type=str)
        parser.add_argument("--labels_info_file_name", default="sample_label.json", type=str)
        self.args = parser.parse_args(args=[])

        with open("src/model_configs/baseline.json") as f:
            configs = json.load(f)
        vision_cfg = configs["vision_cfg"]

        self.train_file_path = "data/aihub_1.0_43_0.3_train_crop.json"
        self.test_file_path = "data/aihub_1.0_43_0.3_test_crop.json"

        self.preprocess = image_transform(vision_cfg["image_size"], is_train=True)
        self.sample_data = {
            "file_name": "1_가자미전_07_071_07011001_160298971793943_1.jpg",
            "id": 329824,
            "category_id": 1,
            "date": "2022-11-29 14:09:08.029536",
        }
        self.tensor = Tensor

        return super().setUp()

    def test_load_train_dataset(self):
        """
        if os.path.isfile(self.train_file_path):
            print("REAL TRAIN DATA DETECTED! Testing code with real data...")
        else:
            print("REAL TRAIN DATA !NOT! DETECTED. Testing code with dummy data...")
            print("If you didn't...")
        """
        dataset = FoodImageDataset(self.args, self.preprocess, mode="train", ratio=1)
        self.assertEqual(dataset.data[0], self.sample_data)
        self.assertEqual(dataset.__len__(), 2)
        self.assertEqual(dataset.__getitem__(0)[0], "가자미전")
        self.assertEqual(type(dataset.__getitem__(0)[1]), self.tensor)

    def test_load_test_dataset(self):
        dataset = FoodImageDataset(self.args, self.preprocess, mode="test", ratio=0.5)
        self.assertEqual(dataset.data[0], self.sample_data)
        self.assertEqual(dataset.__len__(), 1)
        self.assertEqual(dataset.__getitem__(0)[0], "가자미전")
        self.assertEqual(type(dataset.__getitem__(0)[1]), self.tensor)

    def test_check_ratio_range(self):
        with self.assertRaises(ValueError):
            FoodImageDataset(self.args, self.preprocess, mode="train", ratio=2)
