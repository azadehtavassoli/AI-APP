"""Main Streamlit entrypoint for the frontend UI v3 layout."""

import streamlit as st
import os
from app.client.api_client import BackendAPIClient
from app.ui import sources, chat, chat_history, chat_gpt
from app.ui.components import styles

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Bureaucracy Copilot (DE)", page_icon="🤖")

# --- Constants & State ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FEATURE_CHATGPT_UI = os.getenv("FEATURE_CHATGPT_UI", "false").lower() == "true"

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "queue" not in st.session_state:
    st.session_state.queue = []
if "sources_count" not in st.session_state:
    st.session_state.sources_count = 0

# --- Initialization ---
client = BackendAPIClient(base_url=BACKEND_URL)
styles.load_css()
styles.render_header()

# --- Health Check (Banner) ---
try:
    response = client.health_check()
    if not response.success:
        st.error("Backend is unreachable. Please check connection.")
except:
    st.error("Backend connection failed.")

# --- Layout: 3 Columns ---
# Ratios: Left (Sources) 0.25, Center (Chat) 0.5, Right (History) 0.25
col_left, col_center, col_right = st.columns([0.25, 0.5, 0.25])

# --- LEFT: Sources Panel ---
with col_left:
    sources.render(client)

# --- CENTER: Chat Panel ---
with col_center:
    if FEATURE_CHATGPT_UI:
        chat_gpt.render(client=client, show_history_sidebar=False)
    else:
        chat.render(client=client)

# --- RIGHT: History Panel ---
with col_right:
    chat_history.render_history_panel(max_sessions=20)