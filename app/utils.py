import argparse
import json
import os

import pandas as pd
import torch
import wandb
import streamlit as st

from src.dataset import FoodImageDataset
from src.model import build_model
from src.preprocess import image_transform
from src.tokenizer import FoodTokenizer
from src.trainer import HardNegativeTrainer, Trainer
from src.utils import set_seed


class ModelWeakness:
    def __init__(self, model_artifact, artifact_path):
        self.model_artifact = model_artifact
        self.artifact_path = artifact_path
        self.configs = self.get_model_config()
        self.text_cfg = self.configs["text_cfg"]
        self.vision_cfg = self.configs["vision_cfg"]
        self.set_args()
        set_seed(self.args.seed)
        if torch.cuda.is_available():
            # This enables tf32 on Ampere GPUs which is only 8% slower than
            # float16 and almost as accurate as float32
            # This was a default in pytorch until 1.12
            torch.backends.cuda.matmul.allow_tf32 = True
        self.preprocess = self.get_preprocess(self.vision_cfg)
        self.model = self.get_model(self.args, self.vision_cfg, self.text_cfg, self.model_artifact, self.artifact_path)
        self.test_dataset = self.get_test_dataset(self.args, self.preprocess)
        self.tokens_path = "./src/model_configs/tokens_by_length.json"
        self.tokenizer = self.get_tokenizer(self.tokens_path, self.configs)
        self.trainer = self.get_trainer(self.args, self.model, self.tokenizer, test_dataset=self.test_dataset)

        self.weakness = self.trainer.inference(mode="test")

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


    def get_model(self, args, vision_cfg, text_cfg, model_artifact, artifact_path):
        # TODO: wandb artifact로 바꾸기
        path = os.path.join("app/artifacts", model_artifact[: model_artifact.find(".pt") + 3])

        model = build_model(vision_cfg, text_cfg)
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        # if scaler is not None and 'scaler' in checkpoint:
        #    scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> from resuming checkpoint '{model_artifact}' ")
        return model

    def get_test_dataset(self, args, preprocess):
        return FoodImageDataset(args, preprocess, mode="test", ratio=0.0001)

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
        return self.weakness


def get_model_options(runs_df):
    model_option = runs_df[runs_df["artifacts"].apply(len) > 0]["run_name"].tolist()
    return model_option


def get_model_artifact_options(model_option, runs_df):
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
