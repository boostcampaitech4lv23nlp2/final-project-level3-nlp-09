import json
import os

from PIL import Image
from torch.utils.data import Dataset


class FoodImageDataset(Dataset):
    def __init__(self, transforms, mode="train"):
        self.data_path = "data"
        self.dataset_dir = "train"
        self.labels_file_name = "labels.json"
        self.train_file_name = "aihub_1.0_43_0.3_train_crop.json"
        self.test_file_name = "aihub_1.0_43_0.3_test_crop.json"

        self.labels_file_path = os.path.join(self.data_path, self.labels_file_name)
        self.train_file_path = os.path.join(self.data_path, self.train_file_name)
        self.test_file_path = os.path.join(self.data_path, self.test_file_name)

        self.label_data = None
        self.train_data = None
        self.id_to_text_dict = None
        self.text_to_id_dict = None

        if mode == "train":
            self.labels, self.data = self.get_dataset(self.labels_file_path, self.train_file_path)
        elif mode == "test":
            self.labels, self.data = self.get_dataset(self.labels_file_path, self.test_file_path)

        self.id_to_text_dict = self.get_id_to_text(self.labels)
        self.text_to_id_dict = self.get_text_to_id(self.labels)

        self.data = self.data

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def get_dataset(self, labels_file_path, data_file_path):
        with open(labels_file_path, "r") as file:
            labels = json.load(file)
            labels = labels["categories"]

        with open(data_file_path, "r") as file:
            data = json.load(file)
            data = data["images"]

        return labels, data

    def get_id_to_text(self, label_data):
        return {item["id"]: item["label"] for item in label_data}

    def get_text_to_id(self, label_data):
        return {item["label"]: item["id"] for item in label_data}

    def transform_func(self, examples):
        examples["image"] = [self.preprocess(image) for image in examples["image"]]
        return examples

    def __getitem__(self, idx):
        text_id = self.data[idx]["category_id"]
        text = self.id_to_text_dict[text_id]
        file_name = os.path.split(self.data[idx]["file_name"])[-1]
        file_path = os.path.join(self.data_path, self.dataset_dir, file_name)
        image = Image.open(file_path)
        image = self.transforms(image)
        return text, image
