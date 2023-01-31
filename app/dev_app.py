import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import ModelWeakness, get_artifact, get_model_artifact_options, get_model_options, get_wandb_runs_df

runs_df = get_wandb_runs_df()
model_option_list = get_model_options(runs_df)
model_option = st.selectbox("Select your model ✅", model_option_list)

model_artifact_option_list = get_model_artifact_options(model_option, runs_df)
model_artifact_option = st.selectbox("Select your model artifact ✅", model_artifact_option_list)

download_button = st.button("Download artifact")

if download_button:
    artifact_path = get_artifact(model_artifact_option)
    modelWeakness = ModelWeakness(model_artifact_option, artifact_path)
    weakness_df = modelWeakness.get_model_weakness()
    st.dataframe(weakness_df)
