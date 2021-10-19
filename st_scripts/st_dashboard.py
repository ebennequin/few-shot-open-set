import streamlit as st

from st_compare_experiments import st_compare_experiments
from st_dig import st_dig

st.set_page_config(page_title="Analyse experiments", layout="wide")


class DashboardActions:
    METRICS = "Compare experiments"
    DIG = "Dig an experiment"


selected_action = st.sidebar.selectbox(
    label="What now?",
    options=(
        DashboardActions.METRICS,
        DashboardActions.DIG,
    ),
)

if selected_action == DashboardActions.METRICS:
    st_compare_experiments()
elif selected_action == DashboardActions.DIG:
    st_dig()
