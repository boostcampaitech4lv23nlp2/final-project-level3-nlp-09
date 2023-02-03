import json
import os

from PIL import Image
from torch.utils.data import Dataset


class FoodImageDataset(Dataset):
    def __init__(self, args, transforms, mode="train", ratio=1):
        self.args = args
        self.dataset_path = self.args.dataset_path
        self.dataset_mode = "train" if mode == "train" else "test"
        self.labels_info_file_name = self.args.labels_info_file_name
        self.train_info_file_name = self.args.train_info_file_name
        self.test_info_file_name = self.args.test_info_file_name
        self.labels_file_path = os.path.join(self.dataset_path, self.labels_info_file_name)
        self.train_file_path = os.path.join(self.dataset_path, self.train_info_file_name)
        self.test_file_path = os.path.join(self.dataset_path, self.test_info_file_name)

        self.label_data = None
        self.train_data = None
        self.id_to_text_dict = None
        self.text_to_id_dict = None
        self.ratio = ratio

        if self.ratio <= 0 or self.ratio > 1:
            raise ValueError("ratio MUST BE BETWEEN 0 & 1")

        if mode == "train":
            self.labels, self.data = self._get_dataset(self.labels_file_path, self.train_file_path, self.ratio)
        elif mode == "test":
            self.labels, self.data = self._get_dataset(self.labels_file_path, self.test_file_path, self.ratio)

        self.id_to_text_dict = self.get_id_to_text(self.labels)
        self.text_to_id_dict = self.get_text_to_id(self.labels)

        self.data = self.data

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def _get_dataset(self, labels_file_path, data_file_path, ratio):
        with open(labels_file_path, "r") as file:
            labels = json.load(file)
            labels = labels["categories"]

        with open(data_file_path, "r") as file:
            data = json.load(file)
            data = data["images"]

        if ratio < 1:
            data = data[: int(len(data) * ratio)]

        return labels, data

    def get_id_to_text(self, label_data):
        return {item["id"]: item["label"] for item in label_data}

    def get_text_to_id(self, label_data):
        return {item["label"]: item["id"] for item in label_data}

    def __getitem__(self, idx):
        text_id = self.data[idx]["category_id"]
        text = self.id_to_text_dict[text_id]
        file_name = os.path.split(self.data[idx]["file_name"])[-1]
        file_path = os.path.join(self.dataset_path, self.dataset_mode, file_name)
        image = Image.open(file_path)
        image = self.transforms(image)
        item_id = self.data[idx]["id"]
        return text, image, item_id
