# reasoning_patterns = {
#     "mistral" : {
#         "start":  [34],
#         "end": [35]
#     },
#     "deepseek" : {
#         "start":  ['<think>'],
#         "end": ['</think>', '\n\n']
#     },
#     "mistral" : {
#         "start":  [34],
#         "end": ['<|end|>', '<|start|>', 'assistant', '<|channel|>', 'final', '<|message|>']
#     },
# }

from functools import partial
def parse_gpt_oss_reasoning(tokens):
    """
    Parse gpt-oss response tokens into reasoning spans and a final message span.

    Returns:
        {
            "reasoning": [(start_idx, end_idx, channel_name), ...],
            "message": (start_idx, end_idx) or None
        }
    """
    result = {"reasoning": [], "message": (len(tokens), len(tokens))}
    current_channel, start_idx = None, None

    def flush(end_idx):
        nonlocal current_channel, start_idx
        if current_channel and start_idx is not None:
            if current_channel == "final":
                result["message"] = (start_idx, end_idx)
            else:
                result["reasoning"].append((start_idx, end_idx, current_channel))
        current_channel, start_idx = None, None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "<|channel|>":
            if i + 1 < len(tokens):
                current_channel = tokens[i + 1]
            i += 2
        elif start_idx is None and tok == "<|message|>":
            start_idx = i + 1
            i += 1
        elif start_idx is not None and tok in ("<|end|>", "<|return|>", "<|message|>"):
            flush(i)
            start_idx = None
            i += 1
        else:
            i += 1

    flush(len(tokens))  # catch any truncated message
    return result

def find_subarray(tokens, sub, start=0):
    """Find the first index of subarray `sub` in `tokens` starting at `start`."""
    n = len(sub)
    for i in range(start, len(tokens) - n + 1):
        if tokens[i:i + n] == sub:
            return i
    return -1


def find_subarray_backwards(tokens, sub, start=0, end=None):
    """
    Find the **last** index of subarray `sub` in `tokens[start:end]`, scanning backwards.

    Parameters
    ----------
    tokens : list
        The list to search.
    sub : list
        The subarray pattern to match.
    start : int, default 0
        Inclusive lower bound of the search range.
    end : int or None, default None
        Exclusive upper bound of the search range. If None, uses len(tokens).

    Returns
    -------
    int
        The starting index of the **last** match within [start, end), or -1 if not found.
    """
    n = len(sub)
    if end is None:
        end = len(tokens)
    # Make sure range is valid
    start = max(0, start)
    end = min(len(tokens), end)
    if end - start < n:
        return -1

    # iterate backwards starting from the latest possible start index inside [start, end)
    for i in range(end - n, start - 1, -1):
        if tokens[i:i + n] == sub:
            return i
    return -1

def parse_reasoning(tokens, start_pattern, end_pattern):
    i = 0
    reasoning = []
    message_start = 0
    while True:
        end_idx = find_subarray(tokens, ['</think>', '\n\n'], i)
        if end_idx != -1:
            start_idx = find_subarray_backwards(tokens, ['<think>'], i, end_idx)
            if start_idx == -1:
                start_idx = i
            else:
                start_idx += len(start_pattern)
            i = end_idx + len(end_pattern)
            reasoning.append((start_idx, end_idx, "thinking"))
        else:
            message_start = reasoning[-1][1] + len(end_pattern) if len(reasoning) else 0
            break
    return {"reasoning": reasoning, "message": (message_start, len(tokens))}


parse_deepseek_reasoning = partial(
    parse_reasoning,
    start_pattern=["<think>"],
    end_pattern=["</think>", "\n\n"],
)

parse_mistral_reasoning = partial(
    parse_reasoning,
    start_pattern=[34],
    end_pattern=[35],
)
