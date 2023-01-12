import argparse
import json

from src.dataset import FoodImageDataset
from src.model import build_model
from src.preprocess import image_transform
from src.tokenizer import FoodTokenizer
from src.trainer import Trainer


def main(args):

    with open("src/model_configs/baseline.json") as f:
        configs = json.load(f)
    text_cfg = configs["text_cfg"]
    vision_cfg = configs["vision_cfg"]

    preprocess = image_transform(vision_cfg["image_size"], is_train=True)

    model = build_model(vision_cfg, text_cfg)
    dataset = FoodImageDataset(preprocess, mode="train")
    # tokenizer = get_tokenizer()
    tokens_path = "./src/model_configs/tokens_by_length.json"
    tokenizer = FoodTokenizer(tokens_path, configs=configs)
    trainer = Trainer(args, model, tokenizer, dataset)
    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--num_train_epochs", default=32, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_eval", default=False, type=bool)
    parser.add_argument("--labels_file_path", default="class_labels.json", type=str)
    args = parser.parse_args()

    main(args)
