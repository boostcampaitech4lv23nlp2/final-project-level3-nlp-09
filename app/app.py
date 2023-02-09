import json
import os
import sys

import streamlit as st
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.model import build_model
from src.preprocess import image_transform
from src.tokenizer import FoodTokenizer


def app():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.image(uploaded_file)
        # labels = ["짬뽕", "콩나물", "떡갈비", "잡채", "부추전", "무지개떡"]
        image = preprocess(Image.open(uploaded_file)).unsqueeze(0)
        text = tokenizer(labels)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        for value, index in zip(values, indices):
            st.text(f"{labels[index]}: {100*value.item():.2f}%")


with open("src/model_configs/baseline.json") as f:
    configs = json.load(f)

with open("data/labels.json") as f:
    labels_json = json.load(f)

labels = [item["label"] for item in labels_json["categories"]]

text_cfg = configs["text_cfg"]
vision_cfg = configs["vision_cfg"]
preprocess = image_transform(vision_cfg["image_size"], is_train=True)
model = build_model(vision_cfg, text_cfg)
tokens_path = "src/model_configs/tokens_by_length.json"
tokenizer = FoodTokenizer(tokens_path, configs=configs)
checkpoint = torch.load("src/output/epoch_0.pt", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])


app()
