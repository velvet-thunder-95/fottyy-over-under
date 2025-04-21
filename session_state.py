import streamlit as st

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'login_submitted' not in st.session_state:
        st.session_state.login_submitted = False

def check_login_state():
    return st.session_state.logged_in
