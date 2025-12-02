# LLM Resource Comparison Example — async-graph-bench

This example demonstrates how **async-graph-bench** can benchmark large language models (LLMs) across **different configurations and providers**. It allows you to compare the current model implementations, **online OpenAI-style API endpoints** and **local offline vLLM inference**, including multi-instance setups for GPU parallelism.

The goal is to provide a flexible benchmarking setup to evaluate **performance, throughput, and token statistics** across multiple LLM backends.

---

## 🧩 Overview

The benchmark runs a simple computation graph that queries LLMs using a data source of prompts that will induce lengthy responses. The QueryModel node represents a model query with configurable parameters, batching, and resource management.

The workflow demonstrates how you can:

* Run LLM benchmarks with **multiple different providers** (OpenAI endpoints or local vLLM)
* Compare **different model configurations**, including multi-instance parallelization
* Collect **detailed metrics** such as token lengths and time per query
* Store results for later inspection
* Visualize the **execution graph** to understand node dependencies

---

## 🗂 Components

> ![Execution Graph](./execution_graph.svg)
> *Execution graph of the LLM benchmark workflow.*

### **DataSource**

* **`DummyDataSource`**
  Provides a set of prompts for testing, serving as a consistent input for all LLM backends.

### **Computation Node**

| Node           | Description                                                                                            | Requires | Provides                    |
|----------------|--------------------------------------------------------------------------------------------------------|----------|-----------------------------|
| **QueryModel** | Sends prompts to the configured LLM resource and returns the model output along with token statistics. | `prompt` | `response`, `token_lengths` |

> Additional nodes (e.g., statistics or evaluation) can be added similarly for post-processing or benchmarking metrics.

---

## 🏗 Output Structure

* **`data/`** – Folder containing JSON files with model responses and token statistics per resource configuration.
* **`execution_graph.svg`** – Visualization of the execution graph showing nodes and edges for the benchmark.

---

## ▶️ How to Run

1. **Set environment variables** for online endpoints in a `.env` file:

   ```bash
   OPENAI_API_ENDPOINT_1_BASE_URL="https://api.openai.com/v1"
   OPENAI_API_ENDPOINT_1_API_KEY="key_here"
   OPENAI_API_ENDPOINT_1_MODEL="gpt-3.5-turbo"
   OPENAI_API_ENDPOINT_2_BASE_URL="https://api.openai.com/v1"
   OPENAI_API_ENDPOINT_2_API_KEY="key_here"
   OPENAI_API_ENDPOINT_2_MODEL="gpt-4"
   ```
<
2. **Run the benchmark script:**

   ```bash
   python run.py --resources endpoint-1 --batch-size 50
   ```

   **Options for `--resources`:**

    * `endpoint-1` / `endpoint-2` – Single OpenAI-style API endpoint
    * `both-endpoints` – Run across both endpoints
    * `offline-vllm` – Single local vLLM instance
    * `offline-vllm-multi-instance` – Multiple local vLLM instances across available GPUs

3. **Optional arguments:**

    * `--model` – Specify the model name for offline vLLM
    * `--llm-args` – Dictionary of additional arguments for vLLM initialization

4. **Inspect results:**

    * Check the `data/` folder for JSON outputs
    * Open `execution_graph.svg` for a visual overview of the benchmark
    * Review the console output and `benchmark.csv` for detailed statistics

---

## ⚙️ Metrics and Reporting

After the benchmark:

* **Token statistics** – Track lengths of responses for each prompt
* **Time per query** – Measure efficiency of different LLM resources
* **Aggregated CSV report** – Includes resources, model names, batch size, average token lengths, and time per query

This allows you to **compare both performance and output characteristics** between online and offline LLM setups.

---

## 💡 What to Learn from This Example

* How to use async-graph-bench with **multiple LLM providers**
* How to benchmark both **cloud APIs and local inference** efficiently
* How to configure **batch sizes, parallel instances, and LLM parameters**
* How to collect and store **detailed benchmarking metrics** for reproducibility
* How to visualize and inspect the **execution graph** for complex resource setups

## Note

When examining `benchmark.csv`, you may observe that using `vllm-offline-multi-instance` increases processing speed, but the improvement is not linear—for example, using 8 instances does not yield an 8× speedup. This behavior arises from the framework’s **asynchronous, concurrent execution model**. Once the combined throughput of all model instances exceeds the rate at which items can be processed, serialized, and passed between nodes on the main thread, additional instances no longer contribute to overall speed. Increasing the `batch_size` may help mitigate this bottleneck, but a **performance ceiling** will eventually be reached due to these synchronization and utility overheads.


