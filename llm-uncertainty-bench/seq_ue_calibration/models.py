import random

DEFAULTS = {
    # assumes 4x A100 available
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "enable_prefix_caching": True,
    "enforce_eager": True,
    "max_model_len": 16384,
    "seed": random.getrandbits(32)
}

# https://huggingface.co/mistralai/Magistral-Small-2506
MAGISTRAL_SYSTEM_PROMPT = """A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \\boxed{} in your response.

Your thinking process must follow the template below:
[THINK]
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
[/THINK]

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

Problem:"""
MODELS = [
    # OpenAI GPT-OSS
    {
        "name": "openai/gpt-oss-20b",
        "kwargs": {"reasoning_parser": "openai_gptoss"},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "reasoning_parser": "gpt-oss",
        "basename": "gpt-oss-20b",
        "yes_no_ids": [11377, 3004],
    },
    {
        "name": "openai/gpt-oss-120b",
        "kwargs": {"reasoning_parser": "openai_gptoss"},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "reasoning",
        "reasoning_parser": "gpt-oss",
        "basename": "gpt-oss-120b",
        "yes_no_ids": [11377, 3004],
    },

    # Mistral
    {
        "name": "mistralai/Ministral-8B-Instruct-2410",
        "shortname": "Ministral-8B",
        "kwargs": {"tokenizer_mode": "mistral", "tensor_parallel_size": 1},
        "type": "instruct",
        "basename": "Ministral-8B-Instruct-2410",
        "yes_no_ids": [13830, 3501],
    },
    {
        "name": "mistralai/Mistral-Nemo-Base-2407",
        "kwargs": {"tensor_parallel_size": 1, "tokenizer_mode": "mistral"},
        "type": "base",
        "basename": "Mistral-Nemo-Base-2407",
        "yes_no_ids": [13830, 3501],
    },
    {
        "name": "mistralai/Mistral-Nemo-Instruct-2407",
        "shortname": "Mistral-Nemo-7B",
        "kwargs": {"tensor_parallel_size": 1, "tokenizer_mode": "mistral"},
        "type": "instruct",
        "basename": "Mistral-Nemo-Instruct-2407",
        "yes_no_ids": [13830, 3501],
    },
    {
        "name": "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "kwargs": {
            "tokenizer_mode": "mistral",
            "load_format": "mistral",
            "config_format": "mistral",
        },
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "base",
        "shortname": "Mistral-Small-3.1-24B",
        "basename": "Mistral-Small-3.1-24B-Base-2503",
        "yes_no_ids": [13830, 3501],
    },
    {
        "name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "shortname": "Mistral-Small-3.2-24B",
        "kwargs": {"tokenizer_mode": "mistral", "load_format": "mistral", "config_format": "mistral",
                   "gpu_memory_utilization": 0.95},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "instruct",
        "yes_no_ids": [13830, 3501],
    },
    {
        # To enable reasoning, the above MAGISTRAL_SYSTEM_PROMPT must be used. The response will then containt [THINK][/THINK] tags in outputs[0].outputs[0].text
        # The tokens according to logprobs will NOT contain these tokens as plaintexts. Instead, the token_ids must be searched for 34 ([THINK]) and 35 ([/THINK])
        # To get the Magistral tokenizer:
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Magistral-Small-2507")
        # decoded_with_specials = tokenizer.decode([34,35], skip_special_tokens=False) # '[THINK][/THINK]'
        "name": "mistralai/Magistral-Small-2507",
        "kwargs": {"tokenizer_mode": "mistral", "load_format": "mistral",
                   "config_format": "mistral", "gpu_memory_utilization": 0.90},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "reasoning_parser": "mistral",
        "yes_no_ids": [13830, 3501],
    },
    {
        "name": "mistralai/Magistral-Small-2507",
        "basename": "Magistral-Small-2507",  # <===============
        "shortname": "Magistral-Small-24B-Reasoning",  # <===============
        "kwargs": {"tokenizer_mode": "mistral", "load_format": "mistral", "reasoning_parser": "mistral",
                   "config_format": "mistral", "gpu_memory_utilization": 0.90},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "system_prompt": MAGISTRAL_SYSTEM_PROMPT,  # <===============
        "reasoning_parser": "mistral",
        "yes_no_ids": [13830, 3501],
    },

    # Meta Llama
    {
        "name": "meta-llama/Llama-3.1-70B",
        "kwargs": {"gpu_memory_utilization": 0.95},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "base",
        "basename": "Llama-3.1-70B",
        "yes_no_ids": [7566, 2360],
    },
    {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "kwargs": {"gpu_memory_utilization": 0.95},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "instruct",
        "shortname": "Llama-3.3-70B",
        "basename": "Llama-3.3-70B-Instruct",
        "yes_no_ids": [7566, 2360],
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E",
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "base",
        "basename": "Llama-4-Scout-17B-16E",
        "yes_no_ids": [15580, 3318],
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "instruct",
        "shortname": "Llama-4-Scout-17B",
        "basename": "Llama-4-Scout-17B-16E-Instruct",
        "yes_no_ids": [15580, 3318],
    },

    # Qwen
    {
        "name": "Qwen/Qwen3-30B-A3B-Base",
        "kwargs": {"enable_expert_parallel": True},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "base",
        "basename": "Qwen3-30B-A3B-Base",
        "yes_no_ids": [7414, 2308],
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "shortname": "Qwen3-30B-A3B-Instruct",
        "kwargs": {"enable_expert_parallel": True},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "instruct",
        "basename": "Qwen3-30B-A3B-Instruct-2507",
        "yes_no_ids": [7414, 2308],
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "shortname": "Qwen3-30B-A3B-Thinking",
        "kwargs": {
            "enable_expert_parallel": True,
            "reasoning_parser": "qwen3",
        },
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "reasoning_parser": "deepseek",
        "basename": "Qwen3-30B-A3B-Thinking-2507",
        "yes_no_ids": [7414, 2308],
    },

    # Deepseek
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "shortname": "DeepSeek-Llama-70B",
        "kwargs": {
            "gpu_memory_utilization": 0.95,
            "reasoning_parser": "deepseek_r1",
        },
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "reasoning",
        "reasoning_parser": "deepseek",
        "basename": "DeepSeek-R1-Distill-Llama-70B",
        "yes_no_ids": [7566, 2360],
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "shortname": "DeepSeek-Qwen-32B",
        "kwargs": {"reasoning_parser": "deepseek_r1"},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "reasoning_parser": "deepseek",
        "basename": "DeepSeek-R1-Distill-Qwen-32B",
        "yes_no_ids": [7414, 2308],
    },

    # Google
    {
        "name": "google/gemma-3-27b-pt",
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "base",
        "basename": "gemma-3-27b-pt",
        "yes_no_ids": [8438, 2301],  # ["▁Yes","▁No"]
    },
    {
        "name": "google/gemma-3-27b-it",
        "shortname": "google/gemma-3-27b",
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "instruct",
        "basename": "gemma-3-27b-it",
        "yes_no_ids": [8438, 2301],
    },
]

for model in MODELS:
    provider, basename = model["name"].split("/", 1)
    if "basename" not in model:
        model["basename"] = basename
