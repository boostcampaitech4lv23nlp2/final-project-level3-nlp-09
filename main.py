import argparse
import json

from src.dataset import FoodImageDataset
from src.model import build_model
from src.preprocess import image_transform
from src.tokenizer import get_tokenizer
from src.trainer import Trainer


def main(args):

    with open("src/model_configs/baseline.json") as f:
        configs = json.load(f)
    text_cfg = configs["text_cfg"]
    vision_cfg = configs["vision_cfg"]

    preprocess = image_transform(vision_cfg["image_size"], is_train=True)

    model = build_model(vision_cfg, text_cfg)
    train_dataset = FoodImageDataset(args, preprocess, mode="train")
    valid_dataset = FoodImageDataset(args, preprocess, mode="valid")
    tokenizer = get_tokenizer()
    trainer = Trainer(args, model, tokenizer, train_dataset, valid_dataset)
    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)
    parser.add_argument("--dataset_path", default="data", type=str)
    parser.add_argument("--train_info_file_name", default="aihub_1.0_43_0.3_train_crop_crop.json", type=str)
    parser.add_argument("--test_info_file_name", default="aihub_1.0_43_0.3_test_crop_crop.json", type=str)
    parser.add_argument("--labels_info_file_name", default="labels.json", type=str)
    parser.add_argument(
        "--val_frequency", default=1, type=int, help="How often to run evaluation with validation data."
    )
    args = parser.parse_args()

    main(args)
