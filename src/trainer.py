import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.loss import ClipLoss
from src.sampler import ContrastiveSampler


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

        train_sampler = ContrastiveSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, self.args.batch_size, sampler=train_sampler)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        loss_func = ClipLoss()

        pbar = tqdm(train_dataloader, leave=False)
        for epoch in range(self.args.num_train_epochs):
            train_loss = 0.0
            step = 0
            total_data_num = 0

            for texts, images in pbar:
                self.model.train()
                optimizer.zero_grad()
                images = images.to(self.device, dtype=torch.float32)
                texts = self.tokenizer(texts).to(self.device)
                logits_per_image, logits_per_text = self.model(images, texts)
                total_loss = loss_func(logits_per_image, logits_per_text)
                total_data_num += len(images)
                total_loss.backward()
                optimizer.step()

                step += 1
                train_loss += total_loss.item()

                pbar.set_description(f"epoch: {epoch}/ train loss: {total_loss.item()}", refresh=True)
            train_loss /= step
            print(f"epoch: {epoch} train loss: {train_loss}")

    def evaluate(self, mode):
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        pbar = tqdm(eval_dataloader, leave=False)
        step = 0
        eval_loss = 0
        loss_img = nn.CrossEntropyLoss()
        loss_text = nn.CrossEntropyLoss()
        for batch in pbar:
            self.model.eval()
            inputs = batch["image"].to(self.device, dtype=torch.float32)
            labels = batch["label"]
            labels = self.tokenizer([self.id_class_dict[id.item()] for id in labels]).to(self.device)
            image_features = self.model.encode_image(inputs)
            text_features = self.model.encode_text(labels)
            ground_truth = torch.arange(len(inputs)).to(self.device)
            loss = (loss_img(image_features, ground_truth) + loss_text(text_features, ground_truth)) / 2
            eval_loss += loss.item()
            step += 1
            pbar.set_description(f"evaluation loss: {loss.item()}", refresh=True)
        eval_loss /= step
        print(f"eval loss: {eval_loss}")

    def save_model(self):
        pass

    def load_model(self):
        pass
