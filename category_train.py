import argparse
import json

import torch
import torch.nn as nn
from torchvision import models

from src.category_trainer import Trainer
from src.dataset import FoodImageDataset
from src.preprocess import image_transform
from src.tokenizer import FoodTokenizer
from src.utils import set_seed


def main(args):

    with open("src/model_configs/baseline.json") as f:
        configs = json.load(f)
    vision_cfg = configs["vision_cfg"]

    set_seed(args.seed)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True

    preprocess = image_transform(vision_cfg["image_size"], is_train=True)

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    num_classes = 16
    model.fc = nn.Linear(num_features, num_classes)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        # if scaler is not None and 'scaler' in checkpoint:
        #    scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> from resuming checkpoint '{args.resume}' ")
    train_dataset = FoodImageDataset(args, preprocess, mode="train", ratio=1)
    valid_dataset = FoodImageDataset(args, preprocess, mode="test", ratio=0.01)
    test_dataset = FoodImageDataset(args, preprocess, mode="test", ratio=1)

    tokens_path = "./src/model_configs/tokens_by_length.json"
    tokenizer = FoodTokenizer(tokens_path, configs=configs)
    trainer = Trainer(args, model, tokenizer, train_dataset, valid_dataset, test_dataset)

    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate(mode="valid")
    if args.do_inference:
        trainer.inference(mode="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--seed", default=200, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup", default=10000, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_wandb", default=True, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)
    parser.add_argument("--do_inference", default=False, type=bool)
    parser.add_argument("--do_hard_negative", default=False, type=bool)
    parser.add_argument("--do_category_inference", default=True, type=bool)
    parser.add_argument("--dataset_path", default="data", type=str)
    parser.add_argument("--train_info_file_name", default="aihub_1.0_43_0.3_train_crop.json", type=str)
    parser.add_argument("--test_info_file_name", default="aihub_1.0_43_0.3_test_crop.json", type=str)
    parser.add_argument("--labels_info_file_name", default="labels.json", type=str)
    parser.add_argument("--save_logs", default=True, type=bool)
    parser.add_argument("--save_frequency", default=5, type=int)
    parser.add_argument("--checkpoint_path", default="src/output", type=str)
    parser.add_argument(
        "--resume", default="output/epoch_9.pt", type=str, help="path to latest checkpoint (default: None)"
    )
    parser.add_argument(
        "--category_resume",
        default="output/category_epoch_9.pt",
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

    args = parser.parse_args()

    main(args)
