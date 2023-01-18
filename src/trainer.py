import json
import os
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.loss import ClipLoss
from src.sampler import ContrastiveSampler
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

    def train(self):
        if self.args.do_wandb:
            kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
            name = kor_time + "_epochs-" + str(self.args.num_train_epochs) + "_batch-" + str(self.args.batch_size)
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
        train_sampler = ContrastiveSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset, self.args.batch_size, sampler=train_sampler, num_workers=self.args.num_workers
        )
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        optimizer = optim.AdamW(self.model.parameters(), lr=0)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=total_steps)
        loss_func = ClipLoss()

        scaler = torch.cuda.amp.GradScaler()

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
                    images = images.to(self.device, dtype=cast_dtype)
                    texts = self.tokenizer(texts).to(self.device)
                    logits_per_image, logits_per_text = self.model(images, texts)
                    total_loss = loss_func(logits_per_image, logits_per_text)
                    total_data_num += len(images)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)

                scaler.update()
                step += 1
                train_loss += total_loss.item()
                scheduler.step()

                pbar.set_description(f"epoch: {epoch}/ train loss: {total_loss.item()}")

                if self.args.do_wandb:
                    wandb.log({"train_loss": total_loss.item(), "train_epoch": epoch, "lr": get_lr(optimizer)})
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
                        os.path.join(self.args.checkpoint_path, f"{name}_{epoch}.pt"),
                    )
                    print(f"checkpoint '{name}_{epoch}.pt' saved")

    def evaluate(self, mode):
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset
        # eval_sampler = SubsetContrastiveSampler(dataset)
        eval_sampler = SequentialSampler(dataset)
        metrics = {}
        self.model.eval()
        eval_dataloader = DataLoader(dataset, self.args.eval_batch_size, sampler=eval_sampler)
        autocast = get_autocast(self.args.precision)
        cast_dtype = get_cast_dtype(self.args.precision)
        all_image_features, all_text_features = [], []
        cumulative_loss = 0.0
        num_samples = 0

        pbar = tqdm(eval_dataloader, leave=True)
        valid_acc = 0

        with open("data/category_dict.json", encoding="euc-kr") as f:
            category_json = json.load(f)
        category_to_id = {k: idx for idx, k in enumerate(category_json.keys())}

        # with open("data/labels.json") as f:
        #    labels_json = json.load(f)
        food_labels = list(category_to_id.keys())

        tokenized_food_labels = self.tokenizer(food_labels).to(self.device)

        with open("data/food_to_category.json") as f:
            food_to_category = json.load(f)

        with torch.no_grad():
            for texts, images in pbar:
                images = images.to(self.device, dtype=cast_dtype)
                texts = [food_to_category[text] for text in texts]
                tokenized_texts = self.tokenizer(texts).to(self.device)
                with autocast():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(tokenized_texts)
                    food_features = self.model.encode_text(tokenized_food_labels)
                    logit_scale = self.model.logit_scale.exp()

                    all_image_features.append(image_features)
                    all_text_features.append(text_features)

                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    food_features = food_features / food_features.norm(dim=1, keepdim=True)

                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_image_food = logit_scale * image_features @ food_features.t()
                    logits_per_text = logits_per_image.t()
                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=self.device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
                    ) / 2
                    _, preds = torch.max(logits_per_image_food, 1)
                    valid_acc += sum(np.asarray(food_labels)[preds.cpu()] == np.asarray(texts)) / batch_size
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

                pbar.set_description(f"validation loss: {total_loss.item()}")
            """val_metrics = self.get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale,
            )"""
            loss = cumulative_loss / num_samples
            valid_acc = valid_acc / len(eval_dataloader)
            print(f"validation loss: {loss}")
            print(f"validation accuraccy: {valid_acc}")
            metrics.update()
            # metrics.update({**val_metrics, "val_loss": loss.item(), "num_samples": num_samples})
            if self.args.do_wandb:
                wandb.log({"valid_loss": loss.item(), "valid_acc": valid_acc})

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
