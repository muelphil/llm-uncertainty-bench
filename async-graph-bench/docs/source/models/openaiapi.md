# OpenAI API Model

The `OpenAIAPIModel` provides a simple way to query OpenAI-compatible endpoints using the framework’s unified model interface. Below are usage examples for standalone usage, integration in a resource builder, and an illustrative sequence-probability comparison.

---

## Creating the Model Standalone

You can instantiate the model directly with your endpoint and API key:

```python
from async_graph_bench.models.openai import OpenAIAPIModel

model = OpenAIAPIModel(
    openai_endpoint="https://your-api.com/v1",
    openai_api_key="YOUR API KEY"
)
```

Once created, the model can be used immediately to query prompts.

---

## Using the Model to Compare Sequence Probabilities

Here’s an example showing how to query two prompts and compare the sequence probability of the generated outputs:

```python
import math
from async_graph_bench import GenerationParameters

# Define generation parameters
params = GenerationParameters(logprobs=5, max_tokens=20)

# Query the model
response = await model.query(
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

This example uses the `get_assistant_logprobs()` method of [`ResponseWrapper`](../api/responsewrapper.md)` to compute the product of log probabilities for each sequence.

---

## Creating the Model in a Resource Builder

To use the model within a benchmark node, wrap it in a resource builder function. This ensures the model is provided to nodes via the [`NodeConfig`](../api/nodeconfig.md):

```python
from async_graph_bench.models.openai import OpenAIAPIModel
from async_graph_bench import ResourcePool

def build_openai_model(env):
    if not hasattr(env, "main_model"):
        model = OpenAIAPIModel(
            openai_endpoint="https://your-api.com/v1",
            openai_api_key="YOUR API KEY"
        )
        env.main_model = model
    return ResourcePool([env.main_model])
```

You can then provide this function as the `resource_builder` argument when configuring a node:

```python
NodeConfig(
    MyNode(),
    resource_builder=build_openai_model
)
```

For more details on using resources in nodes, see [Resources Documentation](../resources.md).
