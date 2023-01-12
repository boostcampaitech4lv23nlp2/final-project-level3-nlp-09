import numpy as np

import json
import os

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
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

            if self.args.val_frequency > 0 and (epoch + 1) % self.args.val_frequency == 0:
                self.evaluate(mode="valid")

    def evaluate(self, mode):
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        metrics = {}
        self.model.eval()
        eval_dataloader = DataLoader(dataset, self.args.eval_batch_size, sampler=eval_sampler)

        all_image_features, all_text_features = [], []
        cumulative_loss = 0.0
        num_samples = 0

        pbar = tqdm(eval_dataloader, leave=False)

        with torch.no_grad():
            for texts, images in pbar:
                images = images.to(self.device, dtype=torch.float32)
                texts = self.tokenizer(texts).to(self.device)
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                    logit_scale = self.model.logit_scale.exp()

                    all_image_features.append(image_features)
                    all_text_features.append(text_features)

                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=self.device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
                    ) / 2
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                pbar.set_description(f"validation loss: {total_loss.item()}", refresh=True)
            val_metrics = self.get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale,
            )
            loss = cumulative_loss / num_samples
            print(f"validation loss: {loss}")
            metrics.update({**val_metrics, "val_loss": loss.item(), "num_samples": num_samples})

    def get_metrics(self, image_features, text_features, logit_scale):
        metrics = {}
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1).to(self.device)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)

        return metrics

    def save_model(self):
        pass

    def load_model(self):
        pass
