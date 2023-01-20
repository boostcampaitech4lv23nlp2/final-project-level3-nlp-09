# import json
# import os
import pickle

import torch

# from PIL import Image
# from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from torch.utils.data import random_split  # DataLoader, Dataset

# from tqdm import tqdm

# from model import build_model
# from preprocess import image_transform
# from utils import get_autocast

# with open("model_configs/baseline.json") as f:
#     configs = json.load(f)
# text_cfg = configs["text_cfg"]
# vision_cfg = configs["vision_cfg"]

# model = build_model(vision_cfg, text_cfg)
# checkpoint = torch.load(
#     "/opt/ml/final_project/final-project-level3-nlp-09/src/output/epoch_9.pt", map_location="cpu"
# )
# model.load_state_dict(checkpoint["state_dict"])


# class FoodImageDataset(Dataset):
#     def __init__(self, transforms, mode="train"):
#         # self.args = args
#         self.dataset_path = "../data"
#         self.dataset_mode = "train" if mode == "train" else "test"
#         self.labels_info_file_name = "labels.json"
#         self.train_info_file_name = "aihub_1.0_43_0.3_train_crop.json"
#         self.test_info_file_name = "aihub_1.0_43_0.3_test_crop.json"
#         self.labels_file_path = os.path.join(self.dataset_path, self.labels_info_file_name)
#         self.train_file_path = os.path.join(self.dataset_path, self.train_info_file_name)
#         self.test_file_path = os.path.join(self.dataset_path, self.test_info_file_name)

#         self.label_data = None
#         self.train_data = None
#         self.id_to_text_dict = None
#         self.text_to_id_dict = None

#         if mode == "train":
#             self.labels, self.data = self.get_dataset(self.labels_file_path, self.train_file_path)
#         elif mode == "test":
#             self.labels, self.data = self.get_dataset(self.labels_file_path, self.test_file_path)

#         self.id_to_text_dict = self.get_id_to_text(self.labels)
#         self.text_to_id_dict = self.get_text_to_id(self.labels)

#         self.data = self.data

#         self.transforms = transforms

#     def __len__(self):
#         return len(self.data)

#     def get_dataset(self, labels_file_path, data_file_path):
#         with open(labels_file_path, "r") as file:
#             labels = json.load(file)
#             labels = labels["categories"]

#         with open(data_file_path, "r") as file:
#             data = json.load(file)
#             data = data["images"]

#         return labels, data

#     def get_id_to_text(self, label_data):
#         return {item["id"]: item["label"] for item in label_data}

#     def get_text_to_id(self, label_data):
#         return {item["label"]: item["id"] for item in label_data}

#     def transform_func(self, examples):
#         examples["image"] = [self.preprocess(image) for image in examples["image"]]
#         return examples

#     def __getitem__(self, idx):
#         text_id = self.data[idx]["category_id"]
#         text = self.id_to_text_dict[text_id]
#         file_name = os.path.split(self.data[idx]["file_name"])[-1]
#         file_path = os.path.join(self.dataset_path, self.dataset_mode, file_name)
#         image = Image.open(file_path)
#         image = self.transforms(image)
#         return text, image


# def get_split_dataset(dataset, ratio):
#     dataset_a_len = int(len(dataset) * ratio)
#     dataset_b_len = int(len(dataset) - dataset_a_len)
#     dataset_a, dataset_b = random_split(dataset, [dataset_a_len, dataset_b_len])
#     return dataset_a, dataset_b


# preprocess = image_transform(vision_cfg["image_size"], is_train=True)

# train_dataset = FoodImageDataset(preprocess, mode="train")
# dataset = FoodImageDataset(preprocess, mode="test")
# valid_dataset, test_dataset = get_split_dataset(dataset, 0.05)

# valid_dataloader = DataLoader(dataset, batch_size=8, drop_last=True)

# all_images = []
# all_texts = []

# with open("../data/labels.json", "r") as file:
#     labels = json.load(file)
#     labels = labels["categories"]

# indices = {item["label"]: item["id"] for item in labels}
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# autocast = get_autocast("amp")

# for texts, images in tqdm(valid_dataloader):
#     images = images.to(device)
#     with autocast():
#         images_encoded = model.encode_image(images)
#         cpu_device = torch.device('cpu')
#         all_images.extend(images_encoded.tolist())

#         texts = [indices[item] for item in texts]
#         all_texts.extend(texts)
#         torch.cuda.empty_cache()

# with open('embed_img.pkl', 'wb') as f:
#     pickle.dump(all_images, f)
# with open('embed_text.pkl', 'wb') as f:
#     pickle.dump(all_texts, f)

with open("embed_img.pkl", "rb") as f:
    img = pickle.load(f)
with open("embed_text.pkl", "rb") as f:
    text = pickle.load(f)

len_img, len_text = len(img), text

clf = LogisticRegression(random_state=0).fit(img, text)
clf.predict(img)
clf.predict_proba(img)
clf.score(img, text)

from logistic import LogisticRegression

epochs = 10
input_dim = 512  # Two inputs x1 and x2
output_dim = 386  # Single binary output
learning_rate = 0.01

model = LogisticRegression(input_dim, output_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
ratio = 0.1

img_train_len = int(len(img) * ratio)
img_test_len = len(img) - len(img_train_len)
text_train_len = int(len(text) * ratio)
text_test_len = len(text) - len(text_train_len)

img_train, img_test = random_split(img, [img_train_len, img_test_len])
text_train, text_test = random_split(text, [img_train_len, img_test_len])
