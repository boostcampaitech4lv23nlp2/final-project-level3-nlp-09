import argparse

import open_clip

from dataset import FoodDataset
from trainer import Trainer


def main(args):

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="laion400m_e32")
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    dataset = FoodDataset(preprocess)
    train_dataset = dataset.get_train_dataset()
    trainer = Trainer(args, model, tokenizer, train_dataset)
    if args.do_train:
        trainer.train()
    if args.do_eval:
        pass


# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_eval", default=False, type=bool)
    parser.add_argument("--labels_file_path", default="class_labels.json", type=str)
    args = parser.parse_args()

    main(args)
