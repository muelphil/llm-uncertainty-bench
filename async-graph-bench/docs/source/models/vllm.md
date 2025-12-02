# VLLM Model

The `VLLMModel` provides a wrapper for using **vLLM** language models within the framework, exposing the same interface as other [`Model`](../api/model.md) implementations. Usage patterns are very similar to `OpenAIAPIModel`, with additional options for reasoning parsing and offline inference.

Note: Using `VLLMModel` with offline inference can be significantly faster than querying OpenAI API endpoints, especially when processing large batches of prompts. To take advantage of batching, configure the `batch_size` parameter when initializing the [`NodeConfig`](../api/nodeconfig.md) for your node. This allows multiple prompts to be processed simultaneously, maximizing throughput and reducing overall inference time. Depending on the use case you can try out batch sizes from 20 to 250 or more.

---

## Creating the Model Standalone

You can create a `VLLMModel` by first instantiating a `vLLM` `LLM` object and wrapping it:

```python
from vllm import LLM
from async_graph_bench.models.vllm_model import VLLMModel

# Create vLLM backend
llm = LLM("openai/gpt-oss-20b", tensor_parallel_size=1, enable_prefix_caching=True)

# Wrap in framework model
vllm_model = VLLMModel(llm, use_chat_template=True, reasoning_parser_mode="gpt-oss")
```

You can now use `vllm_model` in the same way as other [`Model`](../api/model.md) instances to query prompts.

---

## Using the Model to Compare Sequence Probabilities

```python
# Notice how this section is identical to the OpenAIAPIModel example due to the unified interface
import math

from async_graph_bench import GenerationParameters

params = GenerationParameters(logprobs=5, max_tokens=20)

# Query the model
response = await vllm_model.query(
    ["Say Hello World!", "Respond with Hello World!"],
    params
)

# Get log probabilities of assistant tokens
assistant_logprobs = response.get_assistant_logprobs()

# Compute sequence probabilities
sequence_probabilities = [math.prod(logprobs) for logprobs in assistant_logprobs]

# Compare sequences
print(
    f"Sequence {'1' if sequence_probabilities[0] > sequence_probabilities[1] else '2'} has higher probability"
)
```

The usage is identical to `OpenAIAPIModel`, but with the option to leverage reasoning parsing.

---

## Creating the Model in a Resource Builder

```python
from vllm import LLM
from async_graph_bench.models.vllm_model import VLLMModel
from async_graph_bench import ResourcePool


async def build_main_model(env):
    if not hasattr(env, "main_model_pool"):
        llm = LLM("openai/gpt-oss-20b", tensor_parallel_size=1, enable_prefix_caching=True)
        vllm_model = VLLMModel(llm, use_chat_template=True, reasoning_parser_mode="gpt-oss")
        resource_pool = ResourcePool([vllm_model])
        resource_pool.on_close(vllm_model.close)
        env.main_model_pool = resource_pool
    return env.main_model_pool
```

This function can be provided as the `resource_builder` when initializing a node:

```python
NodeConfig(
    MyNode(),
    resource_builder=build_main_model
)
```

---

## Reasoning Parser

Because vLLM does not natively parse reasoning outputs in the offline batch inference API, the `reasoning_parser_mode` parameter allows specifying a parsing strategy implemented by the framework. Supported modes:

* `gpt-oss` – OpenAI Harmony reasoning format
* `deepseek` – Think block format: `<think>...</think>`
* `mistral` – Mistral format, specifically for `mistralai/Magistral-Small-2507`

This enables extracting structured reasoning from model responses consistently across different vLLM-based models.
