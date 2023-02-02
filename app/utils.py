import argparse
import json
import os

import pandas as pd
import requests
import streamlit as st
import torch
import wandb

from src.dataset import FoodImageDataset
from src.model import build_model
from src.preprocess import image_transform
from src.tokenizer import FoodTokenizer
from src.trainer import HardNegativeTrainer, Trainer
from src.utils import set_seed


class ModelWeakness:
    def __init__(self, artifact):
        self.artifact = artifact
        self.configs = self.get_model_config()
        self.text_cfg = self.configs["text_cfg"]
        self.vision_cfg = self.configs["vision_cfg"]
        self.set_args()
        set_seed(self.args.seed)
        self.food_to_category = self.get_food_to_category()
        if torch.cuda.is_available():
            # This enables tf32 on Ampere GPUs which is only 8% slower than
            # float16 and almost as accurate as float32
            # This was a default in pytorch until 1.12
            torch.backends.cuda.matmul.allow_tf32 = True
        self.preprocess = self.get_preprocess(self.vision_cfg)
        self.model = self.get_model(self.args, self.vision_cfg, self.text_cfg, self.artifact)
        self.test_dataset = self.get_test_dataset(self.args, self.preprocess)
        self.tokens_path = "./src/model_configs/tokens_by_length.json"
        self.tokenizer = self.get_tokenizer(self.tokens_path, self.configs)
        self.trainer = self.get_trainer(self.args, self.model, self.tokenizer, test_dataset=self.test_dataset)

        self.weakness, self.acc = self.trainer.inference(mode="test")

    def get_model_config(self):
        with open("src/model_configs/baseline.json") as f:
            configs = json.load(f)
        return configs

    def set_args(self):
        # TODO: args를 전역에서 관리할 필요성 -> config.yml로 관리하기
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--seed", default=200, type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--eval_batch_size", default=386, type=int)
        parser.add_argument("--num_train_epochs", default=40, type=int)
        parser.add_argument("--warmup", default=10000, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--do_train", default=True, type=bool)
        parser.add_argument("--do_wandb", default=True, type=bool)
        parser.add_argument("--do_eval", default=True, type=bool)
        parser.add_argument("--do_inference", default=True, type=bool)
        parser.add_argument("--do_hard_negative", default=True, type=bool)
        parser.add_argument("--do_category_inference", default=False, type=bool)
        parser.add_argument("--dataset_path", default="data", type=str)
        parser.add_argument("--train_info_file_name", default="train/aihub:1.0_43_0.3_train_crop.json", type=str)
        parser.add_argument("--test_info_file_name", default="test/aihub:1.0_43_0.3_test_crop.json", type=str)
        parser.add_argument("--labels_info_file_name", default="labels.json", type=str)
        parser.add_argument("--save_logs", default=True, type=bool)
        parser.add_argument("--save_frequency", default=5, type=int)
        parser.add_argument("--checkpoint_path", default="src/output", type=str)
        parser.add_argument(
            "--resume",
            default="/opt/ml/final_project/final-project-level3-nlp-09/src/output/epoch_9.pt",
            type=str,
            help="path to latest checkpoint (default: None)",
        )
        parser.add_argument(
            "--category_resume",
            default=None,
            type=str,
            help="path to latest checkpoint (default: None)",
        )
        parser.add_argument(
            "--val_frequency", default=1, type=int, help="How often to run evaluation with validation data."
        )
        parser.add_argument(
            "--precision",
            choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
            default="amp",
            help="Floating point precision.",
        )
        parser.add_argument("--get_embed_space", default=False, type=bool)

        self.args = parser.parse_args()

    def get_preprocess(self, vision_cfg):
        return image_transform(vision_cfg["image_size"], is_train=True)

    def get_model(self, args, vision_cfg, text_cfg, artifact):
        path = os.path.join("app/artifacts", artifact[: artifact.find(".pt") + 3])
        model = build_model(vision_cfg, text_cfg)
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"=> from resuming checkpoint '{artifact}' ")
        return model

    def get_test_dataset(self, args, preprocess):
        return FoodImageDataset(args, preprocess, mode="test", ratio=0.001)

    def get_tokenizer(self, tokens_path, configs):
        return FoodTokenizer(tokens_path, configs=configs)

    def get_trainer(self, args, model, tokenizer, test_dataset, train_dataset=None, valid_dataset=None):
        trainer = (
            HardNegativeTrainer(args, model, tokenizer, train_dataset, valid_dataset, test_dataset)
            if args.do_hard_negative
            else Trainer(args, model, tokenizer, train_dataset, valid_dataset, test_dataset)
        )
        return trainer

    def get_model_weakness(self):
        self.weakness = self.weakness.loc[self.weakness.pred_texts != self.weakness.correct_texts]
        pred_category_ids = [self.food_to_category[food] for food in self.weakness["pred_texts"]]
        correct_category_ids = [self.food_to_category[food] for food in self.weakness["correct_texts"]]
        self.weakness["pred_category_id"] = pred_category_ids
        self.weakness["correct_category_id"] = correct_category_ids

        return self.weakness, self.acc

    def get_food_to_category(self):
        with open(os.path.join("./data", "category_dict.json"), encoding="euc-kr") as f:
            category_to_food = json.load(f)

        food_to_category = {}
        for category_id in category_to_food:
            food_list = category_to_food[category_id]
            for food in food_list:
                food_to_category[food] = category_id

        return food_to_category


def get_model_options(runs_df):
    model_option = runs_df[runs_df["artifacts"].apply(len) > 0]["run_name"].tolist()
    return model_option


def get_artifact_options(model_option, runs_df):
    df = runs_df[runs_df["artifacts"].apply(len) > 0]
    artifact_option = df.loc[df["run_name"] == model_option]["artifacts"].tolist()[0]
    return artifact_option


@st.cache
def get_wandb_runs_df(entity: str = "ecl-mlstudy", project: str = "FOOD CLIP"):
    # wandb project 관련 데이터 가져오기
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    summary_list, config_list, name_list, id_list, commit_id_list, artifact_list = [], [], [], [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)
        id_list.append(run.id)
        commit_id_list.append(run.commit)
        artifacts = []
        for artifact in run.logged_artifacts():
            artifacts.append(artifact.name)
        artifact_list.append(artifacts)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "run_name": name_list,
            "id": id_list,
            "commit_id": commit_id_list,
            "artifacts": artifact_list,
        }
    )

    return runs_df


def get_artifact(artifact_name, entity: str = "ecl-mlstudy", project: str = "FOOD CLIP"):
    api = wandb.Api()
    artifact = api.artifact(entity + "/" + project + "/" + artifact_name)
    artifact.download(root="app/artifacts")
    return artifact_name


def get_commit_id(runs_df, model_option):
    commit_id = runs_df.loc[runs_df["run_name"] == model_option]["commit_id"].iloc[0]
    return commit_id


def send_weakness(url, method, artifact, weakness_df):
    artifact = artifact[: artifact.find(".pt") + 3]
    data = {}
    data["model_name"] = artifact
    data["errors"] = get_error_list(weakness_df)

    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, data=data, verify=False)
        print("response status %r" % response.status_code)
        print("response text %r" % response.text)
        print("response content%r" % response.content)
        print("response url%r" % response.url)
        return response
    except Exception as ex:
        print(ex)


def get_error_list(df):
    error_list = []
    for correct_category, pred_category, correct_label, pred_label in zip(
        df["correct_category_id"], df["pred_category_id"], df["correct_texts"], df["pred_texts"]
    ):
        error = {
            "correct_category": correct_category,
            "pred_category": pred_category,
            "correct_label": correct_label,
            "pred_label": pred_label,
        }
        error_list.append(error)

    return error_list
