from get_prompt_1 import get_prompt_1
from get_prompt_2 import get_prompt_2
from get_prompt_3 import get_prompt_3
from get_prompt_4 import get_prompt_4

mmlu_shots = [
    {
        "question": "What is the capital of France?",
        "answer_options": [
            "Berlin",
            "Madrid",
            "Paris",
            "Rome"
        ],
        "correct_answer_index": 2  # C (zero-based index)
    },
    {
        "question": "Which element has the chemical symbol 'O'?",
        "answer_options": [
            "Oxygen",
            "Gold",
            "Silver",
            "Hydrogen"
        ],
        "correct_answer_index": 0  # A
    },
    {
        "question": "Who wrote 'Hamlet'?",
        "answer_options": [
            "Charles Dickens",
            "William Shakespeare",
            "Mark Twain",
            "Jane Austen"
        ],
        "correct_answer_index": 1  # B
    }
]

for idx, get_prompt in enumerate([get_prompt_1, get_prompt_2, get_prompt_3, get_prompt_4]):
    prompt = get_prompt("<QUESTION>", ["<ANSWER CHOICE A>", "<ANSWER CHOICE B>", "<ANSWER CHOICE C>", "<ANSWER CHOICE D>"],
                     mmlu_shots)
    with open(f"prompt_{idx+1}_design.txt", "w") as file:
        file.write(prompt)