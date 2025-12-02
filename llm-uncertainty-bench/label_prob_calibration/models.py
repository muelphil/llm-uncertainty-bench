import random

DEFAULTS = {
    # assumes 4x A100 available
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "enable_prefix_caching": True,
    "enforce_eager": True,
    "max_model_len": 4096,
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
        # pass chat_template_kwargs={"reasoning_effort": "low"} to handle reasoning
        # the following was tested again and didn't prove correct (leaving it for documentation): adding "<|channel|>analysis<|message|>a<|end|><|start|>assistant<|channel|>final<|message|>" to the input prompt "disables" the reasoning completely
        "name": "openai/gpt-oss-20b",
        "kwargs": {"reasoning_parser": "openai_gptoss"},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "end_of_reasoning_pattern": ['<|end|>', '<|start|>', 'assistant', '<|channel|>', 'final', '<|message|>'],
        "reasoning_parser": "gpt-oss"
    },
    {
        "name": "openai/gpt-oss-120b",
        "kwargs": {"gpu_memory_utilization": 0.95, "reasoning_parser": "openai_gptoss"},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "reasoning",
        "end_of_reasoning_pattern": ['<|end|>', '<|start|>', 'assistant', '<|channel|>', 'final', '<|message|>'],
        "reasoning_parser": "gpt-oss"
    },

    # Mistral / Nemo / Small
    {
        "name": "mistralai/Ministral-8B-Instruct-2410",
        "kwargs": {"tokenizer_mode": "mistral", "tensor_parallel_size": 1},
        "type": "instruct"
    },
    {
        "name": "mistralai/Mistral-Nemo-Base-2407",
        "kwargs": {"tensor_parallel_size": 1, "tokenizer_mode": "mistral"},
        "type": "base"
    },
    {
        "name": "mistralai/Mistral-Nemo-Instruct-2407",
        "kwargs": {"tensor_parallel_size": 1, "tokenizer_mode": "mistral"},
        "type": "instruct"
    },
    {
        "name": "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "kwargs": {"tokenizer_mode": "mistral", "load_format": "mistral", "config_format": "mistral"},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "base"
    },
    {
        "name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "kwargs": {"tokenizer_mode": "mistral", "load_format": "mistral", "config_format": "mistral", "gpu_memory_utilization": 0.95},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "instruct"
    },
    {
        # To enable reasoning, the above MAGISTRAL_SYSTEM_PROMPT must be used. The response will then contain [THINK][/THINK] tags in outputs[0].outputs[0].text
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
        "type": "instruct", # instruct cause reasoning through the MAGISTRAL_SYSTEM_PROMPT has not been enabled
        "end_of_reasoning_pattern": [35],  # [/THINK]
        "reasoning_parser": "mistral"
    },
    {
        "name": "mistralai/Magistral-Small-2507",
        "basename": "Magistral-Small-2507-Reasoning-Enabled", # <===============
        "shortname": "Magistral-Small-Reasoning",
        "kwargs": {"tokenizer_mode": "mistral", "load_format": "mistral", "reasoning_parser": "mistral",
                   "config_format": "mistral", "gpu_memory_utilization": 0.90, "max_model_len": 10240,},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "end_of_reasoning_pattern": [35],  # [/THINK]
        "system_prompt": MAGISTRAL_SYSTEM_PROMPT,  # <===============
        "reasoning_parser": "mistral"
    },

    # Meta Llama
    {
        "name": "meta-llama/Llama-3.1-70B",
        "kwargs": {"gpu_memory_utilization": 0.95},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "base"
    },
    {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "kwargs": {"gpu_memory_utilization": 0.95},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "instruct"
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E",
        "kwargs_a100": {"tensor_parallel_size": 8},
        "kwargs_h100": {"tensor_parallel_size": 4},
        "type": "base"
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "kwargs_a100": {"tensor_parallel_size": 8},
        "kwargs_h100": {"tensor_parallel_size": 4},
        "type": "instruct"
    },

    # Qwen 30B A3B family
    {
        "name": "Qwen/Qwen3-30B-A3B-Base",
        "kwargs": {"enable_expert_parallel": True},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "base"
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "kwargs": {"enable_expert_parallel": True},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "instruct"
    },
    {
        # pass chat_template_kwargs={"enable_thinking": True} to handle reasoning ouput
        "name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "kwargs": {"enable_expert_parallel": True, "reasoning_parser": "qwen3", "max_model_len": 10240,},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "end_of_reasoning_pattern": ['</think>', '\n\n'],
        "reasoning_parser": "qwen"
    },

    # DeepSeek R1 distills
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "kwargs": {"gpu_memory_utilization": 0.95, "reasoning_parser": "deepseek_r1", "max_model_len": 10240},
        "kwargs_a100": {"tensor_parallel_size": 4},
        "kwargs_h100": {"tensor_parallel_size": 2},
        "type": "reasoning",
        "end_of_reasoning_pattern": ['</think>', '\n\n'],
        "reasoning_parser": "deepseek"
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "kwargs": {"reasoning_parser": "deepseek_r1", "max_model_len": 10240},
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "reasoning",
        "end_of_reasoning_pattern": ['</think>', '\n\n'],
        "reasoning_parser": "deepseek"
    },

    # Google Gemma 3 27B
    {
        "name": "google/gemma-3-27b-pt",
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "base"
    },
    {
        "name": "google/gemma-3-27b-it",
        "kwargs_a100": {"tensor_parallel_size": 2},
        "kwargs_h100": {"tensor_parallel_size": 1},
        "type": "instruct"
    },
]

for model in MODELS:
    provider, basename = model["name"].split("/", 1)
    model["provider"] = provider
    if "basename" not in model:
        model["basename"] = basename
