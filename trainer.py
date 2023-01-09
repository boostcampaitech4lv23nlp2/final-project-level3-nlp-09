import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer(object):
    def __init__(self, args, model=None, tokenizer=None, train_dataset=None, valid_dataset=None, test_dataset=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        with open(args.labels_file_path, "r") as f:
            json_data = json.load(f)

        self.id_class_dict = {v: k for k, v in json_data.items()}
        self.label_list = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        train_dataloader = DataLoader(self.train_dataset, self.args.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        loss_img = nn.CrossEntropyLoss()
        loss_text = nn.CrossEntropyLoss()
        pbar = tqdm(train_dataloader, leave=False)
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            learning_loss = 0.0
            step = 0
            for batch in pbar:
                optimizer.zero_grad()
                inputs = batch["image"].to(self.device, dtype=torch.float32)
                labels = batch["label"]
                labels_text = self.tokenizer([self.id_class_dict[id.item()] for id in labels]).to(self.device)
                image_features = self.model.encode_image(inputs)
                text_features = self.model.encode_text(labels_text)
                ground_truth = torch.arange(len(inputs)).to(self.device)
                total_loss = (loss_img(image_features, ground_truth) + loss_text(text_features, ground_truth)) / 2
                total_loss.backward()
                optimizer.step()
                learning_loss += total_loss.item()
                step += 1
                pbar.set_description(f"epoch: {epoch}/ train loss: {total_loss.item()}", refresh=True)
            learning_loss /= step
            print(f"epoch: {epoch} learning_loss: {learning_loss}")

    def evaluate(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
