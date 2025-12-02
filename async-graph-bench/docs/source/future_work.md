# Future Work & Improvements

This section outlines potential directions to extend and improve the framework. These ideas aim to increase flexibility, robustness, and performance of benchmarking workflows.

---

## Resource Management

* Provide **helpers and clearer patterns** for implementing resources that run in **subprocesses**.

    * Currently, multi-instance vLLM setup works, but a more generalized API could simplify creating and managing subprocess-based resources for any type of task.
* Improve **resource usage tracking and optimization**, allowing dynamic scaling of resources depending on benchmark throughput.

---

## Execution & Concurrency

* Implement **Executors** that run nodes in different threads, extending `async_graph_data_flow.AsyncExecutor`.

    * This would allow concurrent execution of nodes that are CPU-bound or I/O-bound without blocking the event loop.
* Explore **advanced scheduling strategies** for nodes, including prioritization or dynamic resource assignment, to maximize throughput.

---

## Error Handling & Recalculation

* Implement **item-level error handling** and selective recalculation.

    * Nodes could mark individual items as erroneous.
    * Only these items would be requeued for recalculation, avoiding the cost of recomputing the entire batch.
    * **Use case:** LLMs may return malformed responses (e.g., 9 items instead of the expected 10 in a batch). Currently, recalculating a full batch for a single error is expensive.

---

## LLM & Model Support

* Expand the set of **LLM inference APIs** supported beyond OpenAI and vLLM.

    * New APIs could include local or cloud-based models, specialized reasoning models, or task-specific LLMs.
* Investigate **hybrid setups** combining multiple model types in a single benchmark, using resources in parallel or sequentially.
* Implement **fine-grained control over reasoning parsing** for additional model formats.

---

## Usability Enhancements

* Provide more **example resource builders and node templates** for common use cases.
* Improve **documentation and helper functions** for visualizing execution graphs, analyzing performance metrics, and debugging node outputs.
* Consider **profiling and logging tools** to monitor resource utilization and benchmark efficiency.

---

## Benchmarking & Performance Analysis

* Introduce **profiling of utility functions and decorators** used in the framework.

    * Measure the execution time of individual decorators and utility layers that handle item passing, batching, and node orchestration.
    * Identify **performance bottlenecks** in the main thread or in asynchronous coordination that limit overall throughput.
    * Use profiling results to **optimize the framework**, improve batch processing, or reduce overhead, especially for multi-instance resources (e.g., vLLM) that eventually reach a speed ceiling.

---


These improvements aim to make the framework more robust, scalable, and user-friendly for benchmarking complex workflows, especially those involving LLMs and asynchronous data dependencies.
