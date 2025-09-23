import string


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(f"{get_uppercase_letter(idx)}) {choice}" for idx, choice in enumerate(choices))


def format_question(question, choices, correct_answer_index=None):
    formatted_question = f"""Question: {question}
Answer Choices:
{join_choices(choices)}

The label of the correct answer choice is """
    return formatted_question


def format_shot(shot):
    return format_question(shot["question"], shot["answer_options"]) + get_uppercase_letter(
        shot["correct_answer_index"])


def get_prompt_3(question, choices, shots):
    examples = "\n\n".join(format_shot(shot) for idx, shot in enumerate(shots))
    formatted_new_question = format_question(question, choices)
    return f"""You are a highly capable language model trained for multiple-choice question answering. In the following examples, you will see questions with answer choices. The answer choices are preceded by the phrase "Answer Choices:". Each answer choice is annotated with one of the labels A, B, C or D. The correct answer to the question is given by the sentence "The label of the correct answer choice is" followed by the corresponding label. Your task is to answer the new question in the same format, outputting only the label of the correct answer to the question you are provided. Do not output anything other than one of the labels A, B, C or D.

{examples}\n\n{formatted_new_question}"""
