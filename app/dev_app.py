import os
import pickle
import sys
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import (
    ModelWeakness,
    empty_cache,
    get_artifact,
    get_artifact_options,
    get_commit_id,
    get_model_options,
    get_wandb_runs_df,
    send_weakness,
)

st.title("📝 Model Analysis Tool")

st.markdown(
    """
            This is a dashboard where you can analyize inference/zero-shot performance of any model that is saved in WandB.
            What you can see:
            - the pie chart shows the distribution of 대분류 of the test dataset
            - the table shows all the wrong predictions
            """
)

if os.path.exists("./app/artifacts/runs_df.pkl"):
    runs_df = pd.read_pickle("./app/artifacts/runs_df.pkl")
else:
    runs_df = get_wandb_runs_df()
    runs_df.to_pickle("./app/artifacts/runs_df.pkl")

model_option_list = get_model_options(runs_df)
model_option = st.selectbox("Select your model ✅", model_option_list)
commit_id = st.write("commit id: ", get_commit_id(runs_df, model_option))

artifact_option_list = get_artifact_options(model_option, runs_df)
artifact_option = st.selectbox("Select your model artifact ✅", artifact_option_list)
artifact_path = "./app/artifacts/" + artifact_option + ".pkl"

url = "https://kyc-system.mynetgear.com/result"

col1, col2, col3, col4 = st.columns(4)
with col1:
    download_button = st.button("Download Artifact 🔍")
with col2:
    send_weakness_button = st.button("Send Weakness to Database 🛫")

if download_button:
    if not os.path.exists(artifact_path):
        get_artifact(artifact_option)
        modelWeakness = ModelWeakness(artifact_option)
        with open(artifact_path, "wb") as f:
            pickle.dump(modelWeakness.get_model_weakness(), f)

if os.path.exists(artifact_path):
    with open(artifact_path, "rb") as f:
        pkl = pickle.load(f)
        total_df, weakness_df, acc = pkl[0], pkl[1], pkl[2]

    weakness_df["same_category"] = weakness_df["pred_category_id"] == weakness_df["correct_category_id"]
    st.write(
        f"""
             accuracy: {acc * 100:.2f}% \n
             데이터셋: {len(total_df)} \n
             오답: {len(weakness_df)}/{len(total_df)} ({len(weakness_df)/len(total_df) * 100:.2f}%) \n
             대분류 내 오답: {weakness_df["same_category"].sum()}/{len(weakness_df)} ({weakness_df["same_category"].sum()/len(weakness_df)*100:.2f}%) \n
             대분류 외 오답: {len(weakness_df) - weakness_df["same_category"].sum()}/{len(weakness_df)} ({100 - weakness_df["same_category"].sum()/len(weakness_df)*100:.2f}%)
             """
    )
    category_fig1 = px.pie(
        total_df,
        values="correct_category_id",
        names="correct_category",
        title="Pie Chart of Categories in Test Dataset",
    )
    st.plotly_chart(category_fig1)
    category_fig2 = px.pie(
        weakness_df,
        values="correct_category_id",
        names="correct_category",
        title="Pie Chart of Categories among Incorrect Test Dataset",
    )
    st.plotly_chart(category_fig2)

    df1 = weakness_df[["pred_texts", "correct_texts", "pred_category", "correct_category", "same_category"]]
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
        theme="alpine",
        enable_enterprise_modules=True,
        height=550,
        width="100%",
        reload_data=False,
    )
    try:
        row_select = dict(grid_response)["selected_rows"][0]
        st.write(row_select)
    except IndexError:
        st.write("no row selected")

    food_to_count_dict = dict(total_df["pred_texts"].value_counts())
    total_df["correct"] = total_df.pred_ids == total_df.correct_ids
    total = total_df.groupby("pred_texts")["correct"].sum().reset_index()
    total["total_guess"] = total["pred_texts"].apply(lambda x: food_to_count_dict[x])
    total["corr_perc"] = total.apply(lambda x: x["correct"] / food_to_count_dict[x["pred_texts"]], axis=1)

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
        theme="alpine",
        enable_enterprise_modules=True,
        height=550,
        width="100%",
        reload_data=False,
    )

    # res = send_weakness(url, "POST", artifact_option, weakness_df)
    # st.write("response: ", res)

if send_weakness_button:
    res = send_weakness(url, "POST", artifact_option, weakness_df)
    st.write("response: ", res)


# Sidebar

size = 0
# get size
for path, dirs, files in os.walk("./app/artifacts"):
    for f in files:
        fp = os.path.join(path, f)
        size += os.path.getsize(fp)


with st.sidebar:
    if st.button("Refresh 🔄️", help="If you can't find your model press this button!"):
        runs_df = get_wandb_runs_df()
    if st.button("Empty Cache 🗑️", help=f"clear all cache: {size * (1/1073741824):.3f}GB"):
        empty_cache()
