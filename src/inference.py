import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FoodImageDataset
from src.model import build_model
from src.preprocess import image_transform
from src.sampler import CategoryContrastiveSampler
from src.tokenizer import FoodTokenizer
from src.utils import set_seed


def inference(args):

    with open("src/model_configs/baseline.json") as f:
        configs = json.load(f)
    text_cfg = configs["text_cfg"]
    vision_cfg = configs["vision_cfg"]

    set_seed(args.seed)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True

    preprocess = image_transform(vision_cfg["image_size"], is_train=True)
    clip_model = build_model(vision_cfg, text_cfg)
    category_model = build_model(vision_cfg, text_cfg)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        clip_model.load_state_dict(checkpoint["state_dict"])
        # if scaler is not None and 'scaler' in checkpoint:
        #    scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> from resuming checkpoint '{args.resume}' ")

    if args.category_resume is not None:
        checkpoint = torch.load(args.category_resume, map_location="cpu")
        category_model.load_state_dict(checkpoint["state_dict"])
        # if scaler is not None and 'scaler' in checkpoint:
        #    scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> from resuming checkpoint '{args.category_resume}' ")

    valid_dataset = FoodImageDataset(args, preprocess, mode="test", ratio=0.01)

    dataset = valid_dataset

    tokens_path = "./src/model_configs/tokens_by_length.json"
    tokenizer = FoodTokenizer(tokens_path, configs=configs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)
    category_model = category_model.to(device)
    clip_model.eval()
    category_model.eval()
    eval_sampler = CategoryContrastiveSampler(dataset)
    eval_dataloader = DataLoader(dataset, 1, sampler=eval_sampler)

    num_samples = 0
    pbar = tqdm(eval_dataloader, leave=True)
    valid_acc = 0
    pred_texts = []
    correct_texts = []
    correct_num = 0

    with open("data/category_dict.json", encoding="euc-kr") as f:
        category_json = json.load(f)
        category_dict = category_json
        category_to_id = list(category_json.keys())

    with torch.no_grad():
        for texts, images in pbar:
            org_text = list(texts)[0]
            images = images.to(device)
            texts = tokenizer(category_to_id).to(device)
            image_features = category_model.encode_image(images)
            text_features = category_model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            batch_size = images.shape[0]
            _, indices = similarity[0].topk(1)
            num_samples += batch_size
            pred_text_id = indices[0].item()
            pred_text = category_to_id[pred_text_id]

            input_texts = category_dict[pred_text]
            if org_text in input_texts:
                org_text_id = input_texts.index(org_text)
            else:
                org_text_id = 0
            texts = tokenizer(input_texts).to(device)
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            batch_size = images.shape[0]
            _, indices = similarity[0].topk(1)

            correct_num += org_text_id == indices[0].item()
            num_samples += batch_size
            pred_text_id = indices[0].item()
            pred_texts.append(input_texts[pred_text_id])
            correct_texts.append(org_text)
        valid_acc = correct_num / len(eval_dataloader)
        df = pd.DataFrame({"pred_texts": pred_texts, "correct_texts": correct_texts})
        df.to_csv(os.path.join(args.dataset_path, "two_stepresult.csv"))
        print(f"validation acc: {valid_acc}")
