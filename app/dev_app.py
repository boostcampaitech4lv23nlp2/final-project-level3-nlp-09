import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import ModelWeakness, get_artifact, get_artifact_options, get_commit_id, get_model_options, get_wandb_runs_df

runs_df = get_wandb_runs_df()
model_option_list = get_model_options(runs_df)
model_option = st.selectbox("Select your model ✅", model_option_list)
commit_id = st.write("commit id: ", get_commit_id(runs_df, model_option))

artifact_option_list = get_artifact_options(model_option, runs_df)
artifact_option = st.selectbox("Select your model artifact ✅", artifact_option_list)

download_button = st.button("Download artifact")

if download_button:
    artifact_path = get_artifact(artifact_option)
    modelWeakness = ModelWeakness(artifact_option)
    weakness_df = modelWeakness.get_model_weakness()
    weakness_df = weakness_df.loc[weakness_df.pred_texts != weakness_df.correct_texts]
    st.dataframe(weakness_df)
