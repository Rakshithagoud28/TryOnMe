import streamlit as st

def init_state():
    if "current_outfit" not in st.session_state:
        st.session_state["current_outfit"] = None

def update_state(key, value):
    st.session_state[key] = value