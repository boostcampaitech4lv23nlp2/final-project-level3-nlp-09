import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
import warnings
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

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
st.title("📝 Model Analysis Tool")

st.markdown("""
            This is a dashboard where you can analyize inference/zero-shot performance of any model that is saved in WandB.
            What you can see:
            - the pie chart shows the distribution of 대분류 of the test dataset
            - the table shows all the wrong predictions
            """)

if os.path.exists("./app/runs_df.pkl"):
    runs_df = pd.read_pickle("./app/runs_df.pkl")
else:
    runs_df = get_wandb_runs_df()

if st.button("Refresh 🔄️", help="If you can't find your model press this button!"):
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
    # get_artifact(artifact_option)
    # modelWeakness = ModelWeakness(artifact_option)
    # weakness_df, acc = modelWeakness.get_model_weakness()
    weakness_df = pd.read_pickle("./app/artifacts/"+ artifact_option + "_df.pkl")
    acc = 0.99
    
    category_fig = px.pie(
        weakness_df, values="correct_category_id", names="correct_category", title="Pie Chart of categories"
    )
    st.write(f"accuracy: {acc * 100:.3f}%")
    st.plotly_chart(category_fig)
    
    df = weakness_df[["pred_texts", "correct_texts","pred_category", "correct_category"]]
    gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_pagination(paginationPageSize=20)
    gb.configure_selection('single', use_checkbox=False)
    gridOptions = gb.build()

    warnings.simplefilter(action="ignore", category=FutureWarning)
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='VALUE_CHANGED', #'MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        theme='alpine',
        enable_enterprise_modules=True,
        height=550, 
        width='100%',
        reload_data=False
    )

    res = send_weakness(url, "POST", artifact_option, weakness_df)
    st.write("response: ", res)
    
if send_weakness_button:
    res = send_weakness(url, "POST", artifact_option, weakness_df)
    st.write("response: ", res)
