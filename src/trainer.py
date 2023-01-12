import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.loss import ClipLoss
from src.sampler import ContrastiveSampler
from src.scheduler import cosine_lr


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
        train_dataloader = DataLoader(
            self.train_dataset, self.args.batch_size, sampler=train_sampler, num_workers=self.args.num_workers
        )
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = cosine_lr(optimizer, self.args.learning_rate, self.args.warmup, total_steps)
        loss_func = ClipLoss()

        scaler = torch.cuda.amp.GradScaler()

        pbar = tqdm(train_dataloader, leave=False)
        for epoch in range(self.args.num_train_epochs):
            train_loss = 0.0
            step = 0
            total_data_num = 0
            step = len(train_dataloader) * epoch
            scheduler(step)

            for texts, images in pbar:
                self.model.train()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    images = images.to(self.device, dtype=torch.float32)
                    texts = self.tokenizer(texts).to(self.device)
                    logits_per_image, logits_per_text = self.model(images, texts)
                    total_loss = loss_func(logits_per_image, logits_per_text)
                    total_data_num += len(images)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                train_loss += total_loss.item()

                pbar.set_description(f"epoch: {epoch}/ train loss: {total_loss.item()}", refresh=True)
            train_loss /= step
            print(f"epoch: {epoch} train loss: {train_loss}")

            if self.args.save_logs:
                checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                #                if scaler is not None:
                #                    checkpoint_dict["scaler"] = scaler.state_dict()

                if epoch + 1 == self.args.num_train_epochs or (
                    self.args.save_frequency > 0 and ((epoch + 1) % self.args.save_frequency) == 0
                ):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(self.args.checkpoint_path, f"epoch_{epoch}.pt"),
                    )
                    print(f"checkpoint 'epoch_{epoch}.pt' saved")

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
