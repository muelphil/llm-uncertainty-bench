# LLM Inference Resources (Models)

The framework provides an abstraction layer for interacting with large language models (LLMs), allowing flexible benchmarking across different inference backends. This abstraction ensures that benchmarks can seamlessly switch between APIs, local deployments, or custom model servers without modifying benchmark logic.

At the core of this system are three key components:
[`Model`](./api/model.md), [`ResponseWrapper`](./api/responsewrapper.md), and [`GenerationParameters`](./api/generationparameters.md).

---

## Overview

The **model abstraction** provides a unified interface for querying language models and handling their responses.
The framework currently includes two main implementations:

* **`VLLMModel`** — uses a local or remote [vLLM](https://vllm.ai) server for high-performance inference.
* **`OpenAIAPIModel`** — queries OpenAI-compatible endpoints using the standard API format.

Additionally, the package `async_graph_bench.models.multi_instance.vllm` supports **running multiple vLLM instances in parallel**, each in its own process, to distribute inference across hardware resources efficiently.

This modular design makes it easy to plug in alternative model interfaces or extend the system with custom integrations.

---

## [`Model`](./api/model.md) Interface

The [`Model`](./api/model.md) abstract base class defines how model implementations should handle queries and responses.
Each subclass must implement the asynchronous `query()` method, which sends a batch of prompts to the model and returns a standardized [`ResponseWrapper`](./api/responsewrapper.md)` instance.

```python
class Model(ABC):
    @abstractmethod
    async def query(self, prompt: List[str] | str | List[dict] | List[List[dict]],
                    generation_params: GenerationParameters) -> ResponseWrapper:
        ...
```

Typical usage:

```python
generation_params = GenerationParameters(temperature=0.7, top_p=0.9)
responses = await my_model.query(["Hello, world!"], generation_params)
```

The [`ResponseWrapper`](./api/responsewrapper.md)` returned ensures that downstream nodes can access messages, tokens, and log probabilities in a consistent way, regardless of the backend.

---

## [`GenerationParameters`](./api/generationparameters.md)

[`GenerationParameters`](./api/generationparameters.md) defines a unified structure for specifying text generation settings such as temperature, top-p, penalties, or token limits.

Rather than using backend-specific argument names, all parameters are stored in a model-agnostic dictionary.
Each model implementation is responsible for converting these parameters into the format required by its API — e.g. OpenAI’s JSON schema or vLLM’s `SamplingParameters`.

### Example

```python
params = GenerationParameters(
    temperature=0.8,
    top_p=0.9,
    n=3,
    max_tokens=512
)
```

The stored parameters can be adapted to a specific model via:

```python
params_for_openai = params.adapt_for_model({
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_completion_tokens"
})
```

This design makes it easy to write benchmark code that runs identically across different inference backends.
See the API reference for a full list of supported parameters and conversion utilities.

---

## [`ResponseWrapper`](./api/responsewrapper.md)` Interface

[`ResponseWrapper`](./api/responsewrapper.md)` defines a standardized interface for accessing model outputs.
It provides methods for retrieving messages, tokens, token IDs, log probabilities, and finish reasons — including reasoning and assistant-specific components if the model exposes them.

This ensures that even when responses differ structurally between APIs (e.g., vLLM vs OpenAI), benchmark code can interact with them uniformly.



## Data Access Overview

The [`ResponseWrapper`](./api/responsewrapper.md)` organizes model responses in a **list of responses**, where each element corresponds to a single prompt provided in the query. Responses can be accessed at two levels:

1. **Message level** — methods that operate on entire messages or reasoning segments.
2. **Token level** — methods that provide data for each token in the output.

### 1. Message-Level Data

| Method                     | Return Type       | Description                                                                      | Example                |
| -------------------------- | ----------------- | -------------------------------------------------------------------------------- | ---------------------- |
| `get_messages()`           | `List[str]`       | All text messages generated for each prompt.                                     | `["Hello World"]`      |
| `get_finish_reasons()`     | `List[str]`       | Reason why generation ended for each prompt (`"stop"`, `"length"`, etc.).        | `["stop"]`             |

> **Note:** The outer list always corresponds to **prompts** in the query, preserving order.

---

### 2. Token-Level Data

| Method                                | Return Type                                 | Description                                                                    | Example                                      |
| ------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------ |----------------------------------------- |
| `get_tokens()`                        | `List[List[str]]`                           | Decoded string tokens for each response.                                       | `[["Hello", "World"]]`                       |
| `get_token_ids()`                     | `List[List[int]]`                           | Token IDs corresponding to each token.                                         | `[[15496, 995]]`                             |
| `get_logprobs()`                      | `List[List[float]]`                         | Log probabilities of each token.                                               | `[[-0.1, -0.05]]`                            |
| `get_tokens_alternatives()`           | `List[List[List[Tuple[str, float]]]]`       | Lists of alternative tokens per token. Each alternative is `(token, logprob)`. | `[[[("Hello", -0.1), ("Hi", -0.3)], [("World", -0.05), ("Earth", -0.2)]]]]` |

The methods of [`ResponseWrapper`](./api/responsewrapper.md)` have **assistant** and **reasoning** variants (e.g., `get_assistant_tokens`, `get_reasoning_tokens`) which should be used instead of the base methods when working with reasoning-capable models. These variants handle models that produce reasoning outputs, which are split into multiple reasoning blocks per response. Using the base methods directly may not correctly capture the nested structure of reasoning outputs.

**Example:**

```python
# get_reasoning_messages returns reasoning segments per response
# Outer list → responses (one per prompt)
# Inner list → reasoning blocks (e.g. <think>Step 1: Hello</think><think>Step 2: World</think>Actual Response)
response_wrapper.get_reasoning_messages()
# Example return:
[[ "Step 1: Hello", "Step 2: World" ]]
```

Here, a single prompt produced two reasoning blocks, showing how the additional list level organizes reasoning segments.
For more details, see the [API reference](./api/responsewrapper).

### Availability per Model Implementation
Here’s the updated table with uncertain or conditional availability replaced with 🟡:

| Method                                | VLLMModel | OpenAIAPIModel                                     |
| ------------------------------------- | --------- |----------------------------------------------------|
| `get_messages()`                      | ✅         | ✅                                                  |
| `get_assistant_messages()`            | ✅         | ✅                                                  |
| `get_reasoning_messages()`            | ✅         | ✅ (reasoning_content in singular segment)          |
| `get_tokens()`                        | ✅         | ✅              |
| `get_reasoning_tokens()`              | ✅         | ⛔                                                  |
| `get_assistant_tokens()`              | ✅         | ✅                                                  |
| `get_token_ids()`                     | ✅         | 🟡 (requires API to return token ids)              |
| `get_reasoning_token_ids()`           | ✅         | ⛔                                                  |
| `get_assistant_token_ids()`           | ✅         | 🟡 (requires API to return token ids)              |
| `get_logprobs()`                      | ✅         | ✅              |
| `get_reasoning_logprobs()`            | ✅         | ⛔                                                  |
| `get_assistant_logprobs()`            | ✅         | 🟡             |
| `get_tokens_alternatives()`           | ✅         | ✅  |
| `get_reasoning_tokens_alternatives()` | ✅         | ⛔                                                  |
| `get_assistant_tokens_alternatives()` | ✅         | ✅ |
| `get_finish_reasons()`                | ✅         | ✅                                                  |

This now clearly distinguishes fully supported (✅), conditional or partial (🟡), and unavailable (⛔) methods for each model. Note that for logprob availability, the `logprobs` parameter in GenerationParameters must be set accordingly.

---

## Extending the Model Interface

You can easily implement new model backends by subclassing [`Model`](./api/model.md) and [`ResponseWrapper`](./api/responsewrapper.md)`.
Your model must:

1. Accept a [`GenerationParameters`](./api/generationparameters.md) instance in its `query()` method.
2. Return a subclass of [`ResponseWrapper`](./api/responsewrapper.md)` that implements the required data-access methods.

This ensures full compatibility with the benchmarking and evaluation pipeline.
