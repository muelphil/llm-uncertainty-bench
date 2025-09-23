def format_assistant_message(assistant_text, reasoning_message):
    return {"role": "assistant", "content": assistant_text}
    # # Note: including reasoning in previous chat messages currently seems not to be supported in any way. Tried reasoning, reasoning_texts, content arrays. All fail as of 15.09.2025.
    # if len(reasoning_thoughts) == 0:
    #     return {"role": "assistant", "content": assistant_text}
    # return {"role": "assistant", "content": [
    #     {"type": "thinking", "content": reasoning_message},
    #     {"type": "final", "content": assistant_text}
    # ]}
