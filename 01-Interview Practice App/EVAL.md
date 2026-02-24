# Prompt Evaluation (EVAL)

## Purpose

This document describes how different prompt strategies were evaluated in the **Interview Practice App** developed for **AI Engineering Sprint 1 (Prompt Engineering)**.

The goal of this evaluation is to compare multiple system prompt strategies used for **answer evaluation and feedback**, and to understand their strengths, weaknesses, and ideal use cases.

---

## Evaluation Setup

All prompt strategies were evaluated using the app’s **Compare Prompts mode**, which ensures a controlled and fair comparison.

### Fixed Inputs
- **Question:** A single interview question (behavioral or technical)
- **Answer:** A fixed user-provided answer (including weak, partial, or adversarial answers)

Using fixed inputs ensures that differences in outputs are caused by the **prompt design**, not by variations in input.

### Model and Parameters
- **Model:** `gpt-4o-mini`
- **Temperature:** Controlled via UI slider (typically low to medium for consistency)
- **Context:** No chat history is used in Compare mode
- **Fair comparison note:** During a single comparison run, temperature was kept constant across strategies to reduce randomness and isolate prompt effects.

---

## Prompt Strategies Evaluated

Five system prompt strategies were implemented and evaluated:

1. **Zero-shot**
2. **Few-shot**
3. **Chain-of-Thought (CoT-guided)**
4. **Rubric-based**
5. **STAR-based**

Each strategy is implemented via `get_feedback_system_prompt()` in `core.py`.

---

## Evaluation Criteria

Prompt outputs were compared using the following qualitative criteria:

- **Clarity:** Is the feedback easy to understand?
- **Structure:** Is the output well-organized and consistent?
- **Usefulness:** Does the feedback help the candidate improve?
- **Consistency:** Does the strategy behave predictably across runs?
- **Safety & Scope Control:** Does the strategy respect evaluation-only guardrails?

---

## Results and Observations

### 1. Zero-shot Prompt
**Strengths**
- Simple and fast baseline
- Works well for basic feedback
- Minimal prompt complexity

**Weaknesses**
- Output quality can vary
- Less structured compared to other strategies

**Best Use Case**
- Baseline evaluation
- Lightweight feedback when speed and simplicity matter

---

### 2. Few-shot Prompt
**Strengths**
- More consistent output structure
- Better tone and formatting due to example guidance

**Weaknesses**
- Slightly longer prompt
- Less flexible if examples do not match the scenario

**Best Use Case**
- General interview practice
- When consistent formatting is important

---

### 3. Chain-of-Thought (CoT-guided)
**Strengths**
- Produces deeper and more nuanced feedback
- Better at identifying strengths and weaknesses

**Weaknesses**
- More verbose
- Requires careful instruction to avoid exposing reasoning

**Best Use Case**
- In-depth coaching
- Advanced candidates seeking detailed feedback

---

### 4. Rubric-based Prompt
**Strengths**
- Quantifiable output (Score 1–5)
- Easy to compare answers objectively
- Highly structured and predictable

**Weaknesses**
- Less conversational
- Can feel rigid for coaching scenarios

**Best Use Case**
- Benchmarking and evaluation
- Comparing multiple answers or candidates

---

### 5. STAR-based Prompt
**Strengths**
- Excellent for behavioral interviews
- Clearly highlights missing components (Situation, Task, Action, Result)

**Weaknesses**
- Not suitable for technical questions
- Assumes STAR-style answers

**Best Use Case**
- Behavioral interview coaching
- Teaching structured storytelling

---

## Guardrails and Safety Observations

All strategies were evaluated under the same **three-layer guardrail system**:

1. System-level rules embedded in prompts
2. User payload reminders
3. Output post-check for violations

The guardrails successfully prevented:
- Generation of full sample answers
- Generation of new interview questions
- Prompt injection attempts (e.g. “ignore system rules”)

**Note:** The output post-check is heuristic (keyword-based). It reduces risk but cannot guarantee perfect prevention of undesired outputs.

---

## Conclusion

**Practical recommendation (based on this evaluation):**
- **Practice mode (general feedback):** Few-shot (more consistent formatting)
- **Benchmarking / scoring:** Rubric (measurable and comparable)
- **Behavioral coaching:** STAR (highly aligned with behavioral answers)

The app demonstrates that **prompt selection should be task-driven**, not one-size-fits-all.  
This evaluation highlights the importance of systematic prompt comparison in real-world LLM applications.

---
