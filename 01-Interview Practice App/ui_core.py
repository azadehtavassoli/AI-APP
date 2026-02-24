import streamlit as st


def render_sidebar(openai_key_present: bool) -> dict:
    with st.sidebar:
        st.header("Settings")

        app_mode = st.radio("App mode", ["Practice", "Compare prompts"], index=0, key="app_mode")

        if app_mode == "Practice":
            practice_mode = st.selectbox("Practice mode", ["Behavioral", "Technical"]) 
            difficulty = st.slider("Difficulty", 1, 3, 2)
            st.caption("1 = Easy, 2 = Medium, 3 = Hard")

            feedback_prompt = st.selectbox(
                "Feedback prompt",
                [
                    "Zero-shot feedback",
                    "Few-shot feedback",
                    "CoT-guided feedback",
                    "Rubric scoring (1–5)",
                    "STAR feedback (Behavioral)",
                ],
                index=0,
            )

            with st.expander("Help"):
                st.write("Zero-shot feedback: concise automated feedback without examples.")
                st.write("Few-shot feedback: shows a short example then gives feedback.")
                st.write("CoT-guided feedback: chain-of-thought style evaluation.")
                st.write("Rubric scoring (1–5): numeric score plus improvement tip.")
                st.write("STAR feedback (Behavioral): feedback structured as Situation/Task/Action/Result.")

            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)

            next_question = st.button("Next question")
            reset_session = st.button("Reset session")

            if not openai_key_present:
                st.warning("OPENAI_API_KEY is not set. Add it to your .env file.")
        else:
            practice_mode = st.selectbox("Practice mode", ["Behavioral", "Technical"]) 
            difficulty = st.slider("Difficulty", 1, 3, 2)
            st.caption("1 = Easy, 2 = Medium, 3 = Hard")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)

            if st.session_state.get("compare_ready", False):
                preset_q = st.session_state.get("compare_question", "")
                preset_a = st.session_state.get("compare_answer", "")
            else:
                preset_q = ""
                preset_a = ""

            compare_question = st.text_area("Fixed question", value=preset_q, key="compare_question", height=120)
            compare_answer = st.text_area("Fixed answer", value=preset_a, key="compare_answer", height=160)

            compare_selected = st.multiselect(
                "Prompts to compare",
                ["Zero-shot", "Few-shot", "CoT", "Rubric", "STAR"],
                default=["Zero-shot", "Few-shot", "CoT", "Rubric", "STAR"],
                key="compare_selected",
            )

            run_comparison = st.button("Run comparison")
            st.warning("This runs multiple API calls.")

            next_question = False
            reset_session = False

    return {
        "app_mode": app_mode,
        "practice_mode": practice_mode,
        "difficulty": difficulty,
        "temperature": temperature,
        "feedback_prompt": locals().get("feedback_prompt", "Zero-shot feedback"),
        "next_question": next_question,
        "reset_session": reset_session,
        "compare_question": locals().get("compare_question", ""),
        "compare_answer": locals().get("compare_answer", ""),
        "compare_selected": locals().get("compare_selected", []),
        "run_comparison": locals().get("run_comparison", False),
    }


def render_history() -> None:
    import re
    import streamlit as st

    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "assistant":
            st.markdown(f"**Interviewer:** {content}")
        elif role == "user":
            st.markdown(f"**You:** {content}")
        elif role == "feedback":
            m = re.search(r"score[:\s]*([1-5])", content, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                if score >= 4:
                    st.success(content)
                elif score == 3:
                    st.info(content)
                else:
                    st.error(content)
            else:
                st.info(content)
        else:
            st.markdown(content)
