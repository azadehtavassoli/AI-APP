"""Shared style helpers used across Streamlit UI panels."""

import streamlit as st

def load_css():
    """Inject centralized CSS definitions for UI v3.

    Args:
        None.

    Returns:
        None: Writes CSS markup into the active Streamlit page.
    """
    st.markdown("""
        <style>
        /* Main Container Adjustments */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 100% !important;
        }

        /* Column styling */
        [data-testid="column"] {
            background-color: #f8f9fa; /* Light gray background for panels */
            border-radius: 10px;
            padding: 15px;
            margin: 0px 5px;
            height: 100%;
        }
        
        /* Center column usually has white background in NotebookLM, 
           but Streamlit structure makes specific column targeting hard without JS.
           We keep a clean look. */

        /* Source Cards */
        .source-card {
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 10px;
            font-size: 0.9em;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .source-card-title {
            font-weight: 600;
            color: #333;
        }
        .source-card-meta {
            font-size: 0.8em;
            color: #666;
        }

        /* History Items */
        .history-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            transition: background 0.2s;
        }
        .history-item:hover {
            background-color: #eef;
        }
        .history-active {
            background-color: #e6f3ff;
            border-left: 3px solid #0068c9;
        }

        /* Buttons & Toggles */
        .stButton button {
            width: 100%;
            border-radius: 6px;
        }
        
        /* Chat Message Styling override */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: row;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #f0f2f6;
        }
        .chat-message.assistant {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
        }
        
        /* Analysis Mode Box */
        .analysis-box {
            background-color: #fff8dc;
            padding: 15px;
            border: 1px solid #f0e68c;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the top-level application header.

    Args:
        None.

    Returns:
        None: Writes header elements to the page.
    """
    st.title("Bureaucracy Copilot for Immigrants (DE)")
    st.markdown("---")