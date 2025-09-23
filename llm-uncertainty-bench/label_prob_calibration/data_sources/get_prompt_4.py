import string


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(f"{get_uppercase_letter(idx)}) {choice}" for idx, choice in enumerate(choices))


def format_question(question, choices, correct_answer=None):
    formatted_question = f"""Question: {question}
Answer Choices:
{join_choices(choices)}

<ANSWER>""" + (f"{correct_answer}<ANSWER>" if correct_answer else "")
    return formatted_question


def format_shot(shot):
    return format_question(shot["question"], shot["answer_options"], get_uppercase_letter(shot["correct_answer_index"]))


def get_prompt_4(question, choices, shots):
    examples = "\n\n".join(f"Example {idx + 1}:\n" + format_shot(shot) for idx, shot in enumerate(shots))
    formatted_new_question = format_question(question, choices)
    return f"""You are a highly capable multiple-choice question answering model. Below are three examples that show the format you must follow. Each question has four answer choices labeled A, B, C, and D. Your task is to answer a new question by outputting the correct answer in the following format: <ANSWER>X<ANSWER>, where X is the label corresponding to the correct answer, A, B, C or D. Do not add any extra text or explanation.

{examples}

Now, please answer the following question in the same format.

{formatted_new_question}"""
