import argparse

# import json
import unittest

# from src.dataset import FoodImageDataset
# from src.preprocess import image_transform


class SamplerTester(unittest.TestCase):
    def setUp(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--seed", default=200, type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--eval_batch_size", default=64, type=int)
        parser.add_argument("--num_train_epochs", default=10, type=int)
        parser.add_argument("--warmup", default=10000, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--do_train", default=False, type=bool)
        parser.add_argument("--do_wandb", default=False, type=bool)
        parser.add_argument("--do_eval", default=True, type=bool)
        parser.add_argument("--do_inference", default=True, type=bool)
        parser.add_argument("--do_hard_negative", default=True, type=bool)
        parser.add_argument("--dataset_path", default="data", type=str)
        parser.add_argument("--train_info_file_name", default="aihub_1.0_43_0.3_train_crop.json", type=str)
        parser.add_argument("--test_info_file_name", default="aihub_1.0_43_0.3_test_crop.json", type=str)
        parser.add_argument("--labels_info_file_name", default="labels.json", type=str)
        parser.add_argument("--save_logs", default=True, type=bool)
        parser.add_argument("--save_frequency", default=5, type=int)
        parser.add_argument("--checkpoint_path", default="src/output", type=str)
        parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
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

        # args = parser.parse_args(args=[])

        # with open("src/model_configs/baseline.json") as f:
        #     configs = json.load(f)
        # vision_cfg = configs["vision_cfg"]

        # preprocess = image_transform(vision_cfg["image_size"], is_train=True)

        # self.train_dataset = FoodImageDataset(args, preprocess, mode="train", ratio=1)

        # TODO: sampler에 필요한 dataset이 action에 올라오지 않아서 문제발생
        # TODO: sample dataset 구축하기
        return super().setUp()

    # TODO: sampler class의 멤버 함수에 대한 유닛 테스트
    def test_len_dset(self):
        pass

    def test_len_cls(self):
        pass
