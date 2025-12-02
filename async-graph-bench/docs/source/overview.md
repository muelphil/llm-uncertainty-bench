# Overview

The `async_graph_bench` framework provides a flexible, modular way to design and run benchmarks as **directed acyclic
graphs (DAGs)**. It is designed for multistep computations, where intermediate computation steps are encapsulated by a
[`Node`](./api/node.md) that defines its required and provided dependencies. The framework manages execution order, resource scheduling, and caching automatically.

## Core Concepts

* **Graph-Based Execution**

    * The [`BenchmarkManager`](./api/benchmarkmanager.md) builds a **directed acyclic graph (DAG)** of nodes based on their declared dependencies.
    * Each node represents a calculation, transformation, or metric applied to incoming data.
    * The graph ensures results are computed in the correct order and reused where possible.

* **Asynchronous Execution**

    * Nodes are executed asynchronously, allowing the framework to make efficient use of external resources (e.g., GPUs, APIs, or databases).
    * Although mainly designed for concurrent programming, custom resources using multiprocessing can be leveraged to run computations in parallel locally.

* **Data Flow**

    * A [`DataSource`](./api/datasource.md) emits items (Python `dict`s).
    * These items travel through the graph and are progressively extended with additional dependencies computed by nodes.

* **Stepwise Execution**

    * For resource-limited environments, benchmarks can be executed in **multiple steps**.
    * Nodes are activated iteratively so that graphs too large to compute in one pass can still be processed in a single benchmark script.

* **Iterative Runs & Sampling**

    * Benchmarks can run multiple iterations over the same dataset.
    * This enables the computation of scores or statistics across these multiple iterations using sampling strategies, which is especially interesting for scenarios where resources are involved that carry inherent randomness (e.g. LLM inference).

* **Node Utilities via Configuration**

    * Nodes can be extended with optional features such as:

        * **Caching/Result persistence**: Caching of outputs from nodes, which may be reused across benchmark runs to avoid redundant computation.
        * **Batching**: Process items in groups for efficiency when working with expensive resources.

---

## Why `async_graph_bench`? — Advantages

The framework was originally motivated by **scientific benchmarking of LLM uncertainty
**, but its design generalizes to a wide range of benchmarking scenarios.

* **Modular & Extensible**

    * Each node is independent and easily replaceable, enabling extensible and flexible benchmarking pipelines.

* **Scalable & Efficient**

    * Supports asynchronous execution of multiple resources.
    * Only recalculates values that are not cached, minimizing redundant work.

* **Transparent Caching Layer**

    * Intermediate results are automatically cached, loaded, and reused.
    * Caching is configurable and requires minimal effort from the user.

* **Preservation of Intermediate Outputs**

    * All intermediate results are accessible, enabling qualitative inspection and debugging.

* **Iterative Updates Without Full Reruns**

    * Metrics or analyses can be added or updated without rerunning the entire benchmark—only affected nodes are recomputed.

* **Result Sharing**

    * Intermediate results (e.g., model generations, metrics) can be shared across researchers or projects.
    * This reduces compute costs and allows others to apply new metrics or analyses on the same cached data.

---

## Notes on Documentation

* Examples in this documentation will be intentionally **simplified** for clarity.
* The actual framework was designed for **uncertainty estimation in large-scale, resource-intensive settings** (e.g., large language model evaluation).
* However, the same principles apply to smaller, general-purpose benchmarking tasks.
