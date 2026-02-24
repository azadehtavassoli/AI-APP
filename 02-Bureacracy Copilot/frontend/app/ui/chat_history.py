"""Chat history panel renderer for session selection and cleanup."""

import streamlit as st
import datetime

def render_history_panel(max_sessions=20):
    """Render the sidebar history panel for chat sessions.

    Args:
        max_sessions (int): Maximum number of most-recent sessions to show.

    Returns:
        None: Writes history controls and session rows to Streamlit.
    """
    st.subheader("History")
    
    # New Chat Button
    if st.button("+ New Chat", key="btn_new_chat_sidebar"):
        st.session_state.current_session_id = None
        st.rerun()

    st.markdown("---")

    # Get sessions (reversed to show newest first)
    sessions = st.session_state.get("chat_sessions", [])
    # Sort by logic if needed, here assuming appended order
    recent_sessions = list(reversed(sessions))[:max_sessions]

    if not recent_sessions:
        st.caption("No chat history.")
    else:
        for idx, s in enumerate(recent_sessions):
            # Determine active state
            is_active = (s.get("id") == st.session_state.get("current_session_id"))
            
            # Label generation
            title = s.get("title", "Untitled Chat")
            if len(title) > 25:
                title = title[:25] + "..."
            
            # Visual indicator for active session
            prefix = "🔵 " if is_active else ""
            
            # Use columns for layout: [Title button] [Delete button]
            c1, c2 = st.columns([0.8, 0.2])
            
            with c1:
                # Clicking title loads the session
                if st.button(f"{prefix}{title}", key=f"hist_sel_{idx}", help=s.get("id")):
                    st.session_state.current_session_id = s.get("id")
                    st.rerun()
            
            with c2:
                # Delete specific session
                if st.button("🗑️", key=f"hist_del_{idx}", help="Delete this session"):
                    # Remove from actual list
                    # We need to find the index in the original list
                    original_idx = sessions.index(s)
                    sessions.pop(original_idx)
                    
                    # If we deleted the current session, reset
                    if st.session_state.current_session_id == s.get("id"):
                        st.session_state.current_session_id = None
                    st.rerun()

    st.markdown("---")
    if st.button("Clear All History", type="primary"):
        st.session_state.chat_sessions = []
        st.session_state.current_session_id = None
        st.rerun()