import string


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(f"{get_uppercase_letter(idx)}) {choice}" for idx, choice in enumerate(choices))


def format_question(question, choices, correct_answer_index=None):
    formatted_question = f"""Question: {question}
{join_choices(choices)}

The correct answer is """
    return formatted_question


def format_shot(shot):
    return format_question(shot["question"], shot["answer_options"]) + get_uppercase_letter(
        shot["correct_answer_index"])


def get_prompt_2(question, choices, shots):
    examples = "\n\n".join(format_shot(shot) for idx, shot in enumerate(shots))
    formatted_new_question = format_question(question, choices)
    return f"{examples}\n\n{formatted_new_question}"
