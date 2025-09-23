import string


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(f"{get_uppercase_letter(idx)}) {choice}" for idx, choice in enumerate(choices))


def format_question(question, choices, correct_answer_index=None):
    formatted_question = f"""**Question:** {question}
{join_choices(choices)}

**Correct Answer:** """
    return formatted_question


def format_shot(shot):
    return format_question(shot["question"], shot["answer_options"]) + get_uppercase_letter(
        shot["correct_answer_index"])


def get_prompt_1(question, choices, shots):
    # Formatting the shots
    examples = "\n\n".join(f"### Example {idx + 1}:\n{format_shot(shot)}" for idx, shot in enumerate(shots))

    # Formatting the new question
    formatted_new_question = format_question(question, choices)

    return f"""You are a highly capable language model trained for multiple-choice question answering.
Below are three examples of multiple-choice questions with labeled answer choices. Each example includes the correct answer.
After the examples, you will be given a new question with four labeled answer choices (A, B, C, D).

Your task is to select the answer choice you believe is correct by responding with only the corresponding label: A, B, C, or D.
Do not include any explanation or additional text.

{examples}

---

{formatted_new_question}"""
