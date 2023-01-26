import json
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_autocast


class EmbedSpace:
    def __init__(
        self, args, model=None, dataset=None, labels_path="data/labels.json", output_path="src/output"
    ) -> None:
        self.checkpoint_path = args.resume
        self.dataset_path = args.dataset_path
        self.model = model
        self.valid_dataloader = DataLoader(dataset, batch_size=8, drop_last=True)
        self.labels_path = labels_path
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = get_autocast("amp")
        # self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        # self.model.load_state_dict(self.checkpoint["state_dict"])

        self.save_embed(labels_path, output_path)
        self.img, self.text = self.get_embed(output_path)

        self.get_embed_space(self.img, self.text)

    def save_embed(self, labels_path, output_path):
        all_images = []
        all_texts = []

        with open(labels_path, "r") as file:
            labels = json.load(file)
            labels = labels["categories"]

        indices = {item["label"]: item["id"] for item in labels}
        self.model.to(self.device)

        img, text = self.get_embed(output_path)

        if not (img and text):
            for texts, images in tqdm(self.valid_dataloader):
                images = images.to(self.device)
                with self.autocast():
                    images_encoded = self.model.encode_image(images)
                    all_images.extend(images_encoded.tolist())

                    texts = [indices[item] for item in texts]
                    all_texts.extend(texts)
                    torch.cuda.empty_cache()

            with open(os.path.join(output_path, "embed_img.pkl"), "wb") as f:
                pickle.dump(all_images, f)
            with open(os.path.join(output_path, "embed_text.pkl"), "wb") as f:
                pickle.dump(all_texts, f)

        else:
            print("=> Take existing embedding vectors saved as .pkl")
        return

    def get_embed(self, output_path):
        img_embed_path = os.path.join(output_path, "embed_img.pkl")
        text_embed_path = os.path.join(output_path, "embed_text.pkl")

        with open(img_embed_path, "rb") as f:
            img = pickle.load(f)
        with open(text_embed_path, "rb") as f:
            text = pickle.load(f)

        return img, text

    def get_samples_per_class(self, text, num_samples_per_class):
        food_dict = defaultdict(int)
        text_idx_list = []

        for idx in range(len(text)):
            label = text[idx]
            if food_dict[label] < num_samples_per_class:
                food_dict[label] += 1
                text_idx_list.append(idx)
        return text_idx_list

    def get_food_to_category(self, food_to_id, category_to_food):
        food_to_category = defaultdict(int)

        for category in category_to_food:
            food_list = category_to_food[category]
            for food in food_list:
                food_to_category[food_to_id[food]] = int(category)

        return food_to_category

    def get_food_to_id(self):
        with open(self.labels_path) as f:
            labels = json.load(f)
        food_to_id = {label["label"]: label["id"] for label in labels["categories"]}

        return food_to_id

    def get_category_to_food(self):
        with open(os.path.join(self.dataset_path, "category_dict.json"), encoding="euc-kr") as f:
            category_to_food = json.load(f)

        return category_to_food

    def get_category_to_idxList(self, text_idx_list, text_sample, food_to_category):
        category_to_idxList = defaultdict(list)

        for idx, total_idx in enumerate(text_idx_list):
            food = text_sample[idx]
            category = food_to_category[food]
            category_to_idxList[category].append(total_idx)

        return category_to_idxList

    def visualize_embed_space_per_class(self, img, text, category_to_idxList):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(img)

        fig, ax = plt.subplots(4, 4, figsize=(30, 30))

        for idx, category in enumerate(category_to_idxList):
            food_list = category_to_idxList[category]

            img_list = embedding[food_list]
            text_list = text[food_list]

            x = img_list[:, 0]
            y = img_list[:, 1]

            ax[idx // 4][idx % 4].scatter(x, y, s=8, c=text_list, label=text_list)
            # for i, food in enumerate(list(text_list)):
            #     # TODO 한글 인코딩
            #     ax[idx // 4][idx % 4].annotate(food, (x[i], y[i]))

        plt.savefig(os.path.join(self.output_path, "embed_space_per_class.png"))

        return

    def get_embed_space(self, img, text, num_samples_per_class=100):
        text_idx_list = self.get_samples_per_class(text, num_samples_per_class)

        img = np.array(img)
        text = np.array(text)

        # img_sample = img[text_idx_list]
        text_sample = text[text_idx_list]

        food_to_id = self.get_food_to_id()
        category_to_food = self.get_category_to_food()
        food_to_category = self.get_food_to_category(food_to_id, category_to_food)
        category_to_idxList = self.get_category_to_idxList(text_idx_list, text_sample, food_to_category)

        # with open("../data/labels.json") as f:
        #     labels = json.load(f)
        # id_to_food = {label["id"]: label["label"] for label in labels["categories"]}

        self.visualize_embed_space_per_class(img, text, category_to_idxList)

        return

    def get_linear_probe(self, output_path):
        pass

    def logistic_regression(self):
        pass
