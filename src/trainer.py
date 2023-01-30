import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loss import ClipLoss
from src.sampler import CustomSampler
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

        self.food_labels = [item["label"] for item in labels_json["categories"]]

        self.text_to_id_dict = {item["label"]: item["id"] for item in labels_json["categories"]}
        self.id_to_text_dict = {item["id"]: item["label"] for item in labels_json["categories"]}
        num_labels = len(labels_json["categories"])
        self.labels = [self.id_to_text_dict[idx] for idx in range(num_labels)]

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
        train_sampler = CustomSampler(do_hard_negative=self.args.do_hard_negative, dset=self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset, self.args.batch_size, sampler=train_sampler, num_workers=self.args.num_workers
        )
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        optimizer = optim.AdamW(self.model.parameters(), lr=0)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.learning_rate, total_steps=total_steps)
        loss_func = ClipLoss()

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.num_train_epochs):
            train_loss = 0.0
            step = 0
            total_data_num = 0
            step = len(train_dataloader) * epoch

            pbar = tqdm(train_dataloader, total=len(train_dataloader) * 3, leave=True)

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
                    model_name = f"{name}_{epoch}.pt"

                    torch.save(
                        checkpoint_dict,
                        os.path.join(self.args.checkpoint_path, model_name),
                    )
                    print(f"checkpoint {model_name} saved")

                    if self.args.do_wandb:
                        model_artifact = wandb.Artifact(model_name, type="model")
                        model_artifact.add_file("src/output/" + model_name)
                        wandb.log_artifact(model_artifact)

    def evaluate(self, mode):
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset
        eval_sampler = CustomSampler(dset=dataset)
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

        tokenized_food_labels = self.tokenizer(self.food_labels).to(self.device)

        with torch.no_grad():
            for texts, images in pbar:
                images = images.to(self.device, dtype=cast_dtype)
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
                    valid_acc += sum(np.asarray(self.food_labels)[preds.cpu()] == np.asarray(texts)) / batch_size
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
        eval_sampler = CustomSampler(dset=dataset)
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

        with torch.no_grad():
            for texts, images in pbar:
                images = images.to(self.device, dtype=cast_dtype)
                org_texts_id = self.text_to_id_dict[list(texts)[0]]
                texts = self.tokenizer(self.labels).to(self.device)
                with autocast():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                batch_size = images.shape[0]
                _, indices = similarity[0].topk(1)
                correct_num += org_texts_id == indices[0].item()
                num_samples += batch_size
                pred_text_id = indices[0].item()
                pred_texts.append(self.id_to_text_dict[pred_text_id])
                correct_texts.append(self.id_to_text_dict[org_texts_id])
            valid_acc = correct_num / len(eval_dataloader)
            df = pd.DataFrame({"pred_texts": pred_texts, "correct_texts": correct_texts})
            df.to_csv(os.path.join(self.args.dataset_path, "result.csv"))
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

    def save_model(self):
        pass

    def load_model(self):
        pass


class HardNegativeTrainer(Trainer):
    def __init__(self, args, model=None, tokenizer=None, train_dataset=None, valid_dataset=None, test_dataset=None):
        super().__init__(args, model, tokenizer, train_dataset, valid_dataset, test_dataset)

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
        train_sampler = CustomSampler(do_hard_negative=self.args.do_hard_negative, dset=self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            self.args.batch_size * 3,
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        optimizer = optim.AdamW(self.model.parameters(), lr=0)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.learning_rate, total_steps=total_steps)
        loss_func = ClipLoss()

        scaler = torch.cuda.amp.GradScaler()

        t = 0.07
        for epoch in range(self.args.num_train_epochs):
            train_loss = 0.0
            step = 0
            total_data_num = 0
            step = len(train_dataloader) * epoch

            pbar = tqdm(train_dataloader, total=len(train_dataloader) * 3, leave=True)

            for texts, images in pbar:
                outputs_texts = []
                outputs_images = []
                self.model.train()
                optimizer.zero_grad()

                with autocast():
                    for index in range(0, self.args.batch_size * 3, self.args.batch_size):
                        torch.cuda.empty_cache()
                        text = texts[index : index + self.args.batch_size]
                        image = images[index : index + self.args.batch_size]
                        image = image.to(self.device, dtype=cast_dtype)
                        text = self.tokenizer(list(text)).to(self.device)
                        features_per_image = self.model.encode_image(image)
                        features_per_text = self.model.encode_text(text)

                        features_per_image = features_per_image / features_per_image.norm(dim=1, keepdim=True)
                        features_per_text = features_per_text / features_per_text.norm(dim=1, keepdim=True)

                        outputs_texts.append(features_per_text)
                        outputs_images.append(features_per_image)
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_texts = torch.cat(outputs_texts)
                    logits_per_images = torch.cat(outputs_images)

                    logits_per_texts = logits_per_texts.view(self.args.batch_size, 3, -1)
                    logits_per_images = logits_per_images.view(self.args.batch_size, 3, -1)

                    logits_per_texts = logits_per_texts[:, 0, :]

                    pos_logits = torch.sum(logits_per_images[:, 0, :] * logits_per_images[:, 1, :], dim=1)
                    neg_logits = torch.sum(logits_per_images[:, 0, :] * logits_per_images[:, 2, :], dim=1)
                    logits = torch.cat([pos_logits.view(-1, 1), neg_logits.view(-1, 1)], dim=1)
                    logits = logits / t
                    logits = torch.exp(logits)
                    logits = logits / torch.sum(logits)
                    loss = torch.sum(-torch.log(logits[:, 0] / torch.sum(logits, dim=1))) / logits.size(0)

                    multiplied_embeddings = logit_scale * (logits_per_images[:, 0, :] @ logits_per_texts.t())

                    total_loss = loss_func(multiplied_embeddings, multiplied_embeddings.t())

                    total_loss = loss + total_loss
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
                    model_name = f"{name}_{epoch}.pt"

                    torch.save(
                        checkpoint_dict,
                        os.path.join(self.args.checkpoint_path, model_name),
                    )
                    print(f"checkpoint {model_name} saved")

                    if self.args.do_wandb:
                        model_artifact = wandb.Artifact(model_name, type="model")
                        model_artifact.add_file("src/output/" + model_name)
                        wandb.log_artifact(model_artifact)
