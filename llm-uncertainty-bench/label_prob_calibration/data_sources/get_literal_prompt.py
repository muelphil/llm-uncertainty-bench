import string


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(f"* {choice}" for choice in choices)


def format_question(question, choices, correct_answer_index=None):
    formatted_question = f"""Question: {question}
Options:
{join_choices(choices)}

Answer: The correct answer is"""
    if correct_answer_index is not None:
        formatted_question += f" {choices[correct_answer_index]}."
    return formatted_question


def format_shot(shot):
    return format_question(shot["question"], shot["answer_options"], shot["correct_answer_index"])


def get_prompt(question, choices, shots):
    # Formatting the shots
    examples = "\n\n".join(format_shot(shot) for shot in shots)

    # Formatting the new question
    formatted_new_question = format_question(question, choices)

    return f"""You are a language model trained for multiple-choice question answering.
Below are several examples of questions with answer options. In each example, the answer sentence ends with the correct option using the phrase "The correct answer is ...". Each option is listed on its own line, preceded by *.
After the examples, you will see a new question with options. Your task is to complete the answer sentence with only the correct answer option. Do not include any extra explanation or text.

{examples}

Now answer the following question:
{formatted_new_question}"""