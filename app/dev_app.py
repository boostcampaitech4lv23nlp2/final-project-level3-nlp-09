import os
import pickle
import sys
import warnings
from os import listdir
from os.path import isfile, join

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import (
    ModelWeakness,
    empty_cache,
    get_artifact,
    get_artifact_options,
    get_commit_id,
    get_model_options,
    get_size_of_cache,
    get_wandb_runs_df,
    send_weakness,
)

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title("📝 Model Analysis Tool")

st.markdown(
    """
            This is a dashboard where you can analyize inference/zero-shot performance of any model that is saved in WandB.
            """
)

test_path = "./data/test"
test_data_pkl_path = "./app/data/test_data.pkl"
if os.path.exists(test_data_pkl_path):
    test_file_list = pd.read_pickle(test_data_pkl_path)
else:
    test_file_list = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    with open(test_data_pkl_path, "wb") as f:
        pickle.dump(test_file_list, f)

if os.path.exists("./app/data/runs_df.pkl"):
    runs_df = pd.read_pickle("./app/data/runs_df.pkl")
else:
    runs_df = get_wandb_runs_df()
    runs_df.to_pickle("./app/data/runs_df.pkl")

model_option_list = get_model_options(runs_df)
model_option = st.selectbox("Select your model ✅", model_option_list)
commit_id = st.write("commit id: ", get_commit_id(runs_df, model_option))

artifact_option_list = get_artifact_options(model_option, runs_df)
artifact_option = st.selectbox("Select your model artifact ✅", artifact_option_list)

number_of_test_data = st.number_input(
    "How many test data do you want to inference?", min_value=1, max_value=202398, value=202, step=1
)
if number_of_test_data:
    dataset_ratio = number_of_test_data / 202398
    st.write(f"Inference will take approx. {number_of_test_data/1050:.2f} mins")

artifact_path = artifact_path = "./app/data/" + artifact_option[: artifact_option.find(".pt") + 3] + ".pkl"
pkl_path = (
    "./app/data/" + artifact_option[: artifact_option.find(".pt") + 3] + "_" + str(int(number_of_test_data)) + ".pkl"
)

url = "https://kyc-system.mynetgear.com/result"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    download_button = st.button("Download Artifact 🔍")
with col2:
    embed_space_button = st.button("Show embedding space of model 🖌")
with col3:
    send_weakness_button = st.button("Send Weakness to Database 🛫")

if download_button:
    if not os.path.exists(artifact_path):
        get_artifact(artifact_option)
    if not os.path.exists(pkl_path):
        modelWeakness = ModelWeakness(artifact_option, dataset_ratio=dataset_ratio)
        with open(pkl_path, "wb") as f:
            pickle.dump(modelWeakness.get_model_weakness(), f)

if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
        total_df, weakness_df, acc = pkl[0], pkl[1], pkl[2]

    food_to_count_dict = dict(total_df["pred_texts"].value_counts())
    total_df["correct"] = total_df.pred_ids == total_df.correct_ids
    total = total_df.groupby("pred_texts")["correct"].sum().reset_index()
    total["total_guess"] = total["pred_texts"].apply(lambda x: food_to_count_dict[x])
    total["corr_perc"] = total.apply(lambda x: x["correct"] / food_to_count_dict[x["pred_texts"]], axis=1)

    weakness_df["same_category"] = weakness_df["pred_category_id"] == weakness_df["correct_category_id"]
    st.write(
        f"""
             accuracy: {acc * 100:.2f}% \n
             클래스 평균 acc.: {total["corr_perc"].sum()/len(total) * 100:.2f}% \n
             데이터셋: {len(total_df)} \n
             오답: {len(weakness_df)}/{len(total_df)} ({len(weakness_df)/len(total_df) * 100:.2f}%) \n
             대분류 내 오답: {weakness_df["same_category"].sum()}/{len(weakness_df)} ({weakness_df["same_category"].sum()/len(weakness_df)*100:.2f}%) \n
             대분류 외 오답: {len(weakness_df) - weakness_df["same_category"].sum()}/{len(weakness_df)} ({100 - weakness_df["same_category"].sum()/len(weakness_df)*100:.2f}%) \n
             """
    )

    col1, col2 = st.columns(2)
    with col2:
        category_fig1 = px.pie(
            total_df,
            values="correct_category_id",
            names="correct_category",
            title="Pie Chart of Categories in Test Dataset",
        )

        st.plotly_chart(category_fig1)

    with col1:

        category_fig2 = px.pie(
            weakness_df,
            values="correct_category_id",
            names="correct_category",
            title="Pie Chart of Categories among Incorrect Test Dataset",
        )
        category_fig2.update_layout(showlegend=False)
        st.plotly_chart(category_fig2)

    df1 = weakness_df[["item_id", "pred_texts", "correct_texts", "pred_category", "correct_category", "same_category"]]
    gb = GridOptionsBuilder.from_dataframe(df1)
    # gb.configure_pagination(paginationPageSize=20)
    gb.configure_selection("single", use_checkbox=False)
    gridOptions = gb.build()

    warnings.simplefilter(action="ignore", category=FutureWarning)
    st.write("대분류 묶음 오답 노트")
    grid_response = AgGrid(
        df1,
        gridOptions=gridOptions,
        data_return_mode="AS_INPUT",
        update_mode="MODEL_CHANGED",  # 'VALUE_CHANGED'
        fit_columns_on_grid_load=True,
        theme="material",
        enable_enterprise_modules=True,
        height=550,
        width="100%",
        reload_data=False,
    )

    col1, col2 = st.columns(2)
    with col1:
        try:
            pred_food = dict(grid_response)["selected_rows"][0]["pred_texts"]
            image_name = next(x for x in test_file_list if pred_food in x)
            with Image.open(test_path + "/" + image_name) as f:
                st.image(f, width=400, caption=f"Predicted : {pred_food}")
        except IndexError:
            pass
    with col2:
        try:
            correct_food = dict(grid_response)["selected_rows"][0]["correct_texts"]
            image_name = next(x for x in test_file_list if correct_food in x)
            with Image.open(test_path + "/" + image_name) as f:
                st.image(f, width=400, caption=f"Ground Truth : {correct_food}")
        except IndexError:
            pass

    df2 = total
    gb = GridOptionsBuilder.from_dataframe(df2)
    # gb.configure_pagination(paginationPageSize=20)
    gb.configure_selection("single", use_checkbox=False)
    gridOptions = gb.build()

    warnings.simplefilter(action="ignore", category=FutureWarning)
    st.write("클래스 묶음 오답 노트")
    grid_response = AgGrid(
        df2,
        gridOptions=gridOptions,
        data_return_mode="AS_INPUT",
        update_mode="MODEL_CHANGED",  # 'VALUE_CHANGED'
        fit_columns_on_grid_load=True,
        theme="material",
        enable_enterprise_modules=True,
        height=550,
        width="100%",
        reload_data=False,
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            pred_food = dict(grid_response)["selected_rows"][0]["pred_texts"]
            image_name = next(x for x in test_file_list if pred_food in x)
            with Image.open(test_path + "/" + image_name) as f:
                st.image(f, width=400, caption=f"Predicted : {pred_food}")
        except IndexError:
            pass

if send_weakness_button:
    res = send_weakness(url, "POST", artifact_option, weakness_df)
    st.write("response: ", res)

if embed_space_button:
    # TODO: 실제 임베딩 벡터로 임베딩 스페이스 그리기 (텍스트, 이미지 둘다)
    image_embed_space = Image.open("app/data/image_embed_space_dummy.png")
    text_embed_space = Image.open("app/data/text_embed_space_dummy.png")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_embed_space, caption="Image embedding space")
    with col2:
        st.image(image_embed_space, caption="Text embedding space")

# Sidebar

with st.sidebar:
    if st.button("Refresh 🔄️", help="If you can't find your model press this button!"):
        runs_df = get_wandb_runs_df()
    if st.button("Empty Cache 🗑️", help=f"clear all cache: {get_size_of_cache():.3f}GB"):
        empty_cache()
