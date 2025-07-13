import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ✅ Ensure session_state.rounds stays across reruns
if "rounds" not in st.session_state:
    st.session_state.rounds = []

# 🛡️ Optional safeguard
if not isinstance(st.session_state.rounds, list):
    st.session_state.rounds = []