"""Primary chat panel rendering for Ask and Analyze modes."""

import uuid
import streamlit as st


MODEL_OPTIONS = [
    "OpenAI · gpt-4.1-mini",
    "OpenAI · gpt-4.1",
    "OpenAI · gpt-4o-mini",
]

LANGUAGE_OPTIONS = ["Auto", "English", "Deutsch", "فارسی"]

MODEL_VALUE_MAP = {
    "OpenAI · gpt-4.1-mini": "gpt-4.1-mini",
    "OpenAI · gpt-4.1": "gpt-4.1",
    "OpenAI · gpt-4o-mini": "gpt-4o-mini",
}

LANGUAGE_VALUE_MAP = {
    "Auto": "auto",
    "English": "en",
    "Deutsch": "de",
    "فارسی": "fa",
}


def _render_usage_metrics(message: dict) -> None:
    """Render compact token usage and cost metadata for one assistant message.

    Args:
        message (dict): Assistant message dictionary from session history.

    Returns:
        None: Writes usage/cost details to the Streamlit UI.
    """
    token_usage = message.get("token_usage")
    model_used = message.get("model_used")
    estimated_cost_usd = message.get("estimated_cost_usd")

    if not token_usage and model_used is None and estimated_cost_usd is None:
        return

    with st.expander("Usage", expanded=False):
        if model_used:
            st.markdown(f"**Model:** {model_used}")

        if isinstance(token_usage, dict):
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.metric("Prompt", int(token_usage.get("prompt_tokens", 0) or 0))
            with col_2:
                st.metric("Completion", int(token_usage.get("completion_tokens", 0) or 0))
            with col_3:
                st.metric("Total", int(token_usage.get("total_tokens", 0) or 0))

        if estimated_cost_usd is not None:
            st.caption(f"Estimated cost: ${estimated_cost_usd:.8f}")

def render(client):
    """Render the center chat panel in Ask mode with per-session model selection.

    Args:
        client: Frontend API client used for query and ingestion calls.

    Returns:
        None: Writes interactive chat/analyze widgets and handles user actions.
    """
    # Step 1: initialize state containers required by the chat workflow.
    # Ensure a per-session model map exists so each session remembers its model choice.
    if "session_model_map" not in st.session_state:
        st.session_state.session_model_map = {}

    # Ensure an active session exists; create one when user lands with no session yet.
    if not st.session_state.current_session_id:
        # Create a stable UUID so message history and model selection can be keyed safely.
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.chat_sessions.append({"id": new_id, "messages": [], "title": "New Chat"})

    # Step 2: resolve current session object from stored list.
    # Use generator lookup for O(n) traversal over a relatively small session list.
    current_session = next((s for s in st.session_state.chat_sessions if s["id"] == st.session_state.current_session_id), None)
    if not current_session:
        # Fallback branch repairs inconsistent state where id exists but session entry is missing.
        new_id = st.session_state.current_session_id
        current_session = {"id": new_id, "messages": [], "title": "New Chat"}
        st.session_state.chat_sessions.append(current_session)

    # Step 3: render toolbar controls for model selection.
    # Analyze mode is intentionally removed; this panel now operates in Ask-only mode.
    st.session_state.chat_mode = "ask"
    with st.container():
        # Add a formal helper label so users understand this control clearly.
        st.caption("Select the model for this chat session")
        # Resolve persisted model for this session; fallback to first curated provider option.
        current_model = st.session_state.session_model_map.get(current_session["id"], MODEL_OPTIONS[0])
        # Precompute index for resilient selectbox rendering, even if old values exist in state.
        selected_index = MODEL_OPTIONS.index(current_model) if current_model in MODEL_OPTIONS else 0
        # Render a compact dropdown with mixed providers (OpenAI, Google, Anthropic).
        selected_model = st.selectbox(
            "Model", 
            MODEL_OPTIONS,
            index=selected_index,
            key="model_selector",
            label_visibility="collapsed"
        )
        # Persist model choice per session id so switching sessions restores previous selection.
        st.session_state.session_model_map[current_session["id"]] = selected_model

    # Render visual separator between toolbar and conversation transcript.
    st.markdown("---")

    # Step 4: render previous messages to preserve conversation continuity.
    for msg in current_session["messages"]:
        # Read normalized role/content fields with safe defaults for backward compatibility.
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Render each historical message bubble using Streamlit chat primitives.
        with st.chat_message(role):
            # Render markdown content for rich formatting support in answers.
            st.markdown(content)
            # Render citations only for assistant messages and only when citations exist.
            if role == "assistant" and "citations" in msg:
                # Use an expander so source details stay available but visually compact.
                with st.expander("References"):
                    for idx, cit in enumerate(msg["citations"]):
                        if isinstance(cit, dict):
                            # Resolve citation fields from multiple backend key variants.
                            c_source = cit.get("source_uri") or cit.get("source_name") or "Unknown Source"
                            c_score = cit.get("score")
                            c_text = cit.get("snippet") or cit.get("text") or "..."
                            c_type = cit.get("source_type", "doc")

                            # Render source heading first for quick provenance scanning.
                            st.markdown(f"**{idx+1}. {c_source}** ({c_type})")
                            if c_score:
                                # Render similarity score when available to indicate relevance confidence.
                                st.caption(f"Score: {c_score:.4f}")
                            # Render snippet as quote block for visual separation from answer text.
                            st.markdown(f"> {c_text}")
                            # Separate citations for readability in long citation lists.
                            st.markdown("---")
                        else:
                            # Fallback rendering for non-dict citations from legacy payloads.
                            st.markdown(f"- {cit}")
            if role == "assistant":
                _render_usage_metrics(msg)

    # Step 5: render bottom-side language preference near the composer.
    if "preferred_language" not in st.session_state:
        st.session_state.preferred_language = LANGUAGE_OPTIONS[0]

    # Place language control directly above input to match requested bottom positioning.
    selected_language = st.selectbox(
        "Response language",
        options=LANGUAGE_OPTIONS,
        index=LANGUAGE_OPTIONS.index(st.session_state.preferred_language)
        if st.session_state.preferred_language in LANGUAGE_OPTIONS
        else 0,
        key="response_language_selector",
        help="Choose preferred answer language. Auto keeps model default behavior.",
    )
    st.session_state.preferred_language = selected_language

    # Step 6: capture new user input and execute Ask workflow.
    # Render chat input at the bottom of the panel.
    if user_input := st.chat_input("Ask a question about your sources..."):
        # Persist user message into session history before backend call to keep timeline complete.
        current_session["messages"].append({"role": "user", "content": user_input})
        # Echo user input immediately so the interface feels responsive.
        with st.chat_message("user"):
            st.markdown(user_input)

        # Render assistant response container for loading + final answer.
        with st.chat_message("assistant"):
            # Show spinner during network + model latency.
            with st.spinner("Thinking..."):
                try:
                    # Build backend request payload with explicit response language and model override.
                    response_language = LANGUAGE_VALUE_MAP.get(selected_language, "auto")
                    selected_model_value = MODEL_VALUE_MAP.get(selected_model)

                    # Use low-level request helper to keep API client public method signatures unchanged.
                    response = client._make_request(
                        "POST",
                        "/api/query",
                        json_data={
                            "question": user_input,
                            "top_k": 5,
                            "chat_history": [],
                            "response_language": response_language,
                            "model": selected_model_value,
                        },
                        timeout=60,
                    )

                    if response.success:
                        # Read normalized backend payload while guarding against None responses.
                        data = response.data or {}
                        answer = data.get("answer", "No answer returned.")
                        citations = data.get("citations", [])
                        token_usage = data.get("token_usage")
                        estimated_cost_usd = data.get("estimated_cost_usd")
                        model_used = data.get("model_used")

                        # Render generated answer in assistant bubble.
                        st.markdown(answer)
                        if citations:
                            # Render fresh citations for this answer using same visual pattern as history.
                            with st.expander("References"):
                                for idx, cit in enumerate(citations):
                                    if isinstance(cit, dict):
                                        c_source = cit.get("source_uri") or cit.get("source_name") or "Unknown Source"
                                        c_score = cit.get("score")
                                        c_text = cit.get("snippet") or cit.get("text") or "..."
                                        c_type = cit.get("source_type", "doc")

                                        st.markdown(f"**{idx+1}. {c_source}** ({c_type})")
                                        if c_score:
                                            st.caption(f"Score: {c_score:.4f}")
                                        st.markdown(f"> {c_text}")
                                        st.markdown("---")
                                    else:
                                        st.markdown(f"- {cit}")

                        _render_usage_metrics(
                            {
                                "token_usage": token_usage,
                                "estimated_cost_usd": estimated_cost_usd,
                                "model_used": model_used,
                            }
                        )

                        # Persist assistant output so it appears in subsequent history renders.
                        current_session["messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "token_usage": token_usage,
                            "estimated_cost_usd": estimated_cost_usd,
                            "model_used": model_used,
                        })

                        # Set title from first user message to improve session navigation labels.
                        if len(current_session["messages"]) <= 2:
                            current_session["title"] = user_input[:30]
                    else:
                        # Surface backend-reported errors directly in the UI.
                        st.error(f"Error from server: {response.error}")

                except Exception as e:
                    # Catch unexpected client/runtime errors to prevent full app crash.
                    st.error(f"Error calling API: {e}")