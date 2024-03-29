import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from src.utils import get_autocast, get_cast_dtype


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


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

        with open(os.path.join(args.dataset_path, args.labels_info_file_name)) as f:
            labels_json = json.load(f)

        with open("data/food_to_category.json") as f:
            food_to_category = json.load(f)
        self.food_to_category = food_to_category

        self.text_to_id_dict = {item["label"]: item["id"] for item in labels_json["categories"]}
        self.id_to_text_dict = {item["id"]: item["label"] for item in labels_json["categories"]}
        num_labels = len(labels_json["categories"])
        self.labels = [self.id_to_text_dict[idx] for idx in range(num_labels)]

    def train(self):
        if self.args.do_wandb:
            kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
            name = (
                "category"
                + kor_time
                + "_epochs-"
                + str(self.args.num_train_epochs)
                + "_batch-"
                + str(self.args.batch_size)
            )
            wandb.init(
                project="FOOD CLIP",
                entity="ecl-mlstudy",
                name=name,
                config={
                    "learning_rate": self.args.learning_rate,
                    "epochs": self.args.num_train_epochs,
                    "batch_size": self.args.batch_size,
                },
            )

        autocast = get_autocast(self.args.precision)
        cast_dtype = get_cast_dtype(self.args.precision)
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset, self.args.batch_size, sampler=train_sampler, num_workers=self.args.num_workers
        )
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        optimizer = optim.AdamW(self.model.parameters(), lr=0)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=total_steps)
        loss_func = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler()

        with open("data/category_dict.json", "r", encoding="euc-kr") as f:
            category_dict_json = json.load(f)

        with open("data/food_to_category.json", "r") as f:
            food_to_category = json.load(f)
        category_labels = list(category_dict_json.keys())
        category_idx_dict = {key: idx for idx, key in enumerate(category_labels)}

        for epoch in range(self.args.num_train_epochs):
            train_loss = 0.0
            step = 0
            total_data_num = 0
            step = len(train_dataloader) * epoch

            pbar = tqdm(train_dataloader, leave=True)

            for texts, images in pbar:

                self.model.train()
                optimizer.zero_grad()
                with autocast():
                    categories = [food_to_category[text] for text in list(texts)]
                    labels = torch.tensor([category_idx_dict[category] for category in categories]).to(self.device)
                    images = images.to(self.device, dtype=cast_dtype)
                    probs = self.model(images)
                    loss = loss_func(probs, labels)
                    total_data_num += len(images)
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                step += 1
                train_loss += loss.item()
                scheduler.step()

                pbar.set_description(f"epoch: {epoch}/ train loss: {loss.item()}")

                if self.args.do_wandb:
                    wandb.log({"train_loss": loss.item(), "train_epoch": epoch, "lr": get_lr(optimizer)})
            train_loss /= step
            print(f"epoch: {epoch} train loss: {train_loss}")

            if self.args.val_frequency > 0 and (epoch + 1) % self.args.val_frequency == 0:
                self.evaluate(mode="valid")

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
                        os.path.join(self.args.checkpoint_path, f"category_epoch_{epoch}.pt"),
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
        metrics = {}
        self.model.eval()
        eval_dataloader = DataLoader(dataset, self.args.eval_batch_size, sampler=eval_sampler)
        autocast = get_autocast(self.args.precision)
        cast_dtype = get_cast_dtype(self.args.precision)
        cumulative_loss = 0.0
        num_samples = 0
        loss_func = nn.CrossEntropyLoss()

        with open("data/category_dict.json", "r", encoding="euc-kr") as f:
            category_dict_json = json.load(f)
        category_labels = list(category_dict_json.keys())
        category_idx_dict = {key: idx for idx, key in enumerate(category_labels)}
        with open("data/food_to_category.json", "r") as f:
            food_to_category = json.load(f)

        pbar = tqdm(eval_dataloader, leave=True)
        valid_acc = 0

        with torch.no_grad():
            for texts, images in pbar:
                with autocast():
                    categories = [food_to_category[text] for text in list(texts)]
                    labels = torch.tensor([category_idx_dict[category] for category in categories]).to(self.device)
                    images = images.to(self.device, dtype=cast_dtype)
                    probs = self.model(images)
                    loss = loss_func(probs, labels)
                    _, preds = torch.max(probs, 1)
                    batch_size = images.shape[0]

                    valid_acc += torch.sum(preds.cpu() == labels.cpu()) / batch_size
                cumulative_loss += loss * batch_size
                num_samples += batch_size

                pbar.set_description(f"validation loss: {loss.item()}")
            """val_metrics = self.get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale,
            )"""
            loss = cumulative_loss / num_samples
            valid_acc = valid_acc / len(eval_dataloader)
            print(f"validation loss: {loss} validation acc: {valid_acc}")
            metrics.update()
            # metrics.update({**val_metrics, "val_loss": loss.item(), "num_samples": num_samples})
            if self.args.do_wandb:
                wandb.log({"valid_loss": loss.item(), "valid_acc": valid_acc})

    def inference(self, mode):
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        metrics = {}
        self.model.eval()
        eval_dataloader = DataLoader(dataset, 1, sampler=eval_sampler)
        autocast = get_autocast(self.args.precision)
        cast_dtype = get_cast_dtype(self.args.precision)
        num_samples = 0
        correct_num = 0
        pbar = tqdm(eval_dataloader, leave=True)
        valid_acc = 0
        pred_texts = []
        correct_texts = []

        with open("data/category_dict.json", encoding="euc-kr") as f:
            category_json = json.load(f)
        category_to_id = list(category_json.keys())

        with torch.no_grad():
            for texts, images in pbar:
                org_texts = [self.food_to_category[text] for text in list(texts)]
                images = images.to(self.device, dtype=cast_dtype)
                # org_texts_id = self.text_to_id_dict[list(texts)[0]]
                texts = self.tokenizer(category_to_id).to(self.device)
                with autocast():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                batch_size = images.shape[0]
                _, indices = similarity[0].topk(1)
                num_samples += batch_size
                pred_text_id = indices[0].item()
                pred_texts.append(category_to_id[pred_text_id])
                correct_texts.append(org_texts[0])
                correct_num += org_texts[0] == category_to_id[pred_text_id]
            valid_acc = correct_num / len(eval_dataloader)
            df = pd.DataFrame({"pred_texts": pred_texts, "correct_texts": correct_texts})
            df.to_csv(os.path.join(self.args.dataset_path, "01181252_epochs-10_batch-16_9.csv"))
            print(f"validation acc: {valid_acc}")
            metrics.update()

    def get_metrics(self, image_features, text_features, logit_scale):
        metrics = {}
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

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
