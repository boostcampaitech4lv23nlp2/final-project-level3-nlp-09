import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import (
    ModelWeakness,
    get_artifact,
    get_artifact_options,
    get_commit_id,
    get_model_options,
    get_wandb_runs_df,
    send_weakness,
)

runs_df = get_wandb_runs_df()
model_option_list = get_model_options(runs_df)
model_option = st.selectbox("Select your model ✅", model_option_list)
commit_id = st.write("commit id: ", get_commit_id(runs_df, model_option))

artifact_option_list = get_artifact_options(model_option, runs_df)
artifact_option = st.selectbox("Select your model artifact ✅", artifact_option_list)
weakness_df = pd.DataFrame()

url = "https://kyc-system.mynetgear.com/result"
download_button = st.button("Download Artifact 🔍")
send_weakness_button = st.button("Send Weakness to Database 🛫")

if download_button:
    get_artifact(artifact_option)
    modelWeakness = ModelWeakness(artifact_option)
    weakness_df, acc = modelWeakness.get_model_weakness()

    category_df = weakness_df["correct_category_id"]
    category_fig = px.pie(
        category_df, values="correct_category_id", names="correct_category_id", title="Pie Chart of categories"
    )
    st.write(f"accuracy: {acc * 100:.3f}%")
    st.plotly_chart(category_fig)
    st.dataframe(weakness_df)

    res = send_weakness(url, "POST", artifact_option, weakness_df)
    st.write("response: ", res)

if send_weakness_button:
    res = send_weakness(url, "POST", artifact_option, weakness_df)
    st.write("response: ", res)
