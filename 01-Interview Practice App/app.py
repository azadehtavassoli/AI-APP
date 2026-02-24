import os
from dotenv import load_dotenv
import streamlit as st

# Load env early (services.openai_client also does this)
load_dotenv()

from core import init_state, get_next_question, OPENAI_API_KEY, evaluate_answer, compare_prompts
from ui_core import render_sidebar, render_history


st.set_page_config(page_title="Interview Practice App")

# Title and short description (identical text)
st.title("Interview Practice App")
st.write("A lightweight app to practice interview questions in different modes and difficulties.")


# Small helpers kept local to the UI wiring
def add_message(role: str, content: str) -> None:
    """Append a message dict to the session history."""
    st.session_state["messages"].append({"role": role, "content": content})


def generate_intro(mode: str, difficulty: int) -> str:
    """Create the first assistant prompt based on selected mode and difficulty."""
    return (
        f"Hi! I'm your interviewer. We'll do {mode} questions at difficulty {difficulty}. "
        + "Tell me about yourself."
    )


def generate_assistant_reply(mode: str, _user_text: str) -> str:
    """Rule-based follow-up replies for each practice mode (no external API calls)."""
    if mode == "Behavioral":
        return "Thanks — can you describe that using the STAR format? Start with the Situation and Task."
    if mode == "Technical":
        return "Good; can you explain the underlying approach in more detail and describe trade-offs?"
    if mode == "Job Description":
        return "Thanks — which specific skills from the job description do you think map to your experience?"
    return "Thanks — tell me more."


# Initialize session state (questions loaded from project_root/questions.csv)
init_state(st)


# Render sidebar and read values
sidebar_values = render_sidebar(openai_key_present=bool(OPENAI_API_KEY))

# Unpack sidebar selections into local names used by the main flow
app_mode = sidebar_values["app_mode"]
practice_mode = sidebar_values["practice_mode"]
difficulty = sidebar_values["difficulty"]
temperature = sidebar_values["temperature"]
feedback_prompt = sidebar_values["feedback_prompt"]
next_question = sidebar_values["next_question"]
reset_session = sidebar_values["reset_session"]
compare_question = sidebar_values["compare_question"]
compare_answer = sidebar_values["compare_answer"]
compare_selected = sidebar_values["compare_selected"]
run_comparison = sidebar_values["run_comparison"]


# Main area: conversation / history and input controls
st.subheader("Conversation")

if app_mode == "Compare prompts":
    # Compare mode: render a dedicated compare page (do not run chat UI)
    st.markdown("### Compare prompts mode — run multiple evaluations on the same Q&A")

    fixed_q = compare_question
    fixed_a = compare_answer
    selected = compare_selected

    st.text_area("Fixed question", value=fixed_q, height=120, disabled=True)
    st.text_area("Fixed answer", value=fixed_a, height=160, disabled=True)

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
        st.stop()

    if run_comparison:
        if not fixed_q.strip() or not fixed_a.strip():
            st.error("Please provide both a fixed question and a fixed answer to run comparison.")
        else:
            with st.spinner("Running comparisons..."):
                results = compare_prompts(fixed_q, fixed_a, difficulty, temperature, selected)

            order = ["Zero-shot", "Few-shot", "CoT", "Rubric", "STAR"]
            for label in order:
                if label not in selected:
                    continue
                if label not in results:
                    continue
                st.subheader(f"{label} result")
                text = results[label]
                if label == "Rubric":
                    import re

                    m = re.search(r"Score[:\s]*(\d)", text)
                    if m:
                        st.write("Score:", int(m.group(1)))
                st.write(text)

    st.stop()

else:
    # Practice mode: same behavior
    if next_question:
        questions = st.session_state.get('questions', [])
        used = set(st.session_state.get('used_question_ids', []))
        q = get_next_question(questions, practice_mode, difficulty, used)
        if q is None:
            add_message('assistant', "No questions available for this mode/difficulty. Try another difficulty or add more questions to questions.csv.")
        else:
            st.session_state['used_question_ids'].append(q['id'])
            st.session_state['current_question'] = q
            st.session_state['last_question_text'] = q['question']
            add_message('assistant', f"Question ({q['id']}): {q['question']}")

    if reset_session:
        st.session_state['messages'] = []
        st.session_state['used_question_ids'] = []
        st.session_state['current_question'] = None
        st.session_state['last_answer'] = None
        try:
            st.rerun()
        except Exception:
            st.stop()

    render_history()

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your answer", key="user_input")
        sent = st.form_submit_button("Send")

    if sent and user_input:
        add_message("user", user_input)
        st.session_state['last_answer_text'] = user_input

        current_q = st.session_state.get('current_question')
        if current_q is None:
            add_message('assistant', 'No current question selected. Press "Next question" to load one.')
            try:
                st.rerun()
            except Exception:
                st.stop()

        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
            add_message('feedback', '[Local] OPENAI_API_KEY is not configured — cannot generate AI feedback.')
            try:
                st.rerun()
            except Exception:
                st.stop()

        try:
            mapping = {
                "Zero-shot feedback": "Zero-shot",
                "Few-shot feedback": "Few-shot",
                "CoT-guided feedback": "CoT",
                "Rubric scoring (1–5)": "Rubric",
                "STAR feedback (Behavioral)": "STAR",
            }
            strategy_key = mapping.get(feedback_prompt, "Zero-shot")

            feedback_text = evaluate_answer(current_q['question'], user_input, strategy_key, difficulty, temperature)
            add_message('feedback', feedback_text)

            try:
                st.rerun()
            except Exception:
                st.stop()
        except Exception as e:
            st.error('Failed to evaluate answer. See console for details.')
            add_message('feedback', '[Error] Unable to evaluate answer.')
            try:
                st.rerun()
            except Exception:
                st.stop()

    if st.button("Compare this Q&A"):
        st.session_state["compare_question"] = st.session_state.get("last_question_text", "")
        st.session_state["compare_answer"] = st.session_state.get("last_answer_text", "")
        st.session_state["compare_ready"] = True
        try:
            st.rerun()
        except Exception:
            st.stop()

    if not st.session_state["messages"]:
        st.info("Conversation will appear here after you press Next question.")

