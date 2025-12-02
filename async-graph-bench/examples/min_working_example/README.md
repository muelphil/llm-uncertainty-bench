# Minimal Example using `async-graph-bench`

This example demonstrates how to use **async-graph-bench** on a small, self-contained computation graph that can be run on any laptop. It serves as a **minimal working example** to explore the framework’s structure, node dependencies, and sampling capabilities without relying on specialized hardware or large datasets.

---

## 🧩 Overview

The example builds and benchmarks a small computational graph that processes a list of number pairs. Each item flows through multiple **nodes**, where intermediate and final results are computed asynchronously and optionally sampled.

The workflow demonstrates how you can:

* Decompose **complex multi-step computations** into clear, reusable nodes
* Use different **sampling strategies** to compute aggregate statistics across samples
* Store intermediate results in **CSV files** for inspection and reuse
* Execute and monitor the **benchmark run**
* Visualize the **execution graph** to understand data and dependency flow
* Export **report data** to a file for later analysis

---

## 🗂 Components

> ![Execution Graph](./execution_graph.svg)
> *Execution graph showing data flow and dependencies between node classes.*

### **DataSource**

* **`NumberTupleDataSource`**
  Provides the input data — a list of `(first_number, second_number)` tuples. Exposes them via the properties:

    * `first_number`
    * `second_number`

---

### **Computation Nodes**

| Node                   | Description                                                                                                         | Requires                                      | Provides                |
|------------------------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|-------------------------|
| **NoiseAdder**         | Adds Gaussian noise to `first_number` using a shared `DummyNoiseResource` (simulates a slow or expensive resource). | `first_number`                                | `first_number_noisy`    |
| **SquareCalculator**   | Squares each `second_number`.                                                                                       | `second_number`                               | `second_number_squared` |
| **AdditionCalculator** | Adds `first_number_noisy` and `second_number_squared` to produce the final `sum`.                                   | `first_number_noisy`, `second_number_squared` | `sum`                   |

---

### **Sampling Nodes**

These nodes **sample** the items for the dependency `sum` to estimate variance and other metrics. Each uses a different **sampling mode**:

| Node                            | Sampling Mode | Description                                                                                                                                                                                         |
|---------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **VarianceEstimator**           | `first`       | Samples the sums and computes their variance to estimate noise impact.                                                                                                                              |
| **DiffFromMeanEstimator**       | `extend`      | Samples the sum anc computes the difference of the individual sum from the mean of sampled sums across iterations.                                                                                  |
| **DiffFromMeanSpreadEstimator** | `spread`      | Similar to the previous node but computes the result batched for all items in the sampling set, then spreads computed differences across all sampled items (more efficient for batch computations). |

---

## 🏗 Output Structure

* **`./data/`** – Folder containing CSV files with intermediate and final results from each node.
* **`./report.csv`** – Summary of benchmark computation statistics and time taken.
* **`./execution_graph.svg`** – Visualization of the full execution graph (nodes and edges).

---

## ▶️ How to Run

1. **Install the framework** (if not already):

   ```bash
   pip install async-graph-bench
   ```

2. Navigate to the example directory and **run the example**:

   ```bash
   python run.py
   ```

3. **Inspect results:**

    * View generated CSV data in `./data/`
    * Open `execution_graph.svg` to see the visual graph
    * Review benchmark summary in `report.csv`

---

## ⚙️ Key Parameters

* `iterations = 25` — Number of times individual items are computed by the graph - will lead to different results for the same item due to the probabilistic nature of noise.
* `batch_size = 50` — Batch size per node.
* `queue_size = 100` — Node queue buffer size.
* `always_recompute=True` — Forces recomputation even if cached data exists by deleting the data before the run.

---

## 💡 What to Learn from This Example

* How to define and connect **custom nodes** with `requires` / `provides` fields.
* How **sampling configurations** affect computation and data flow.
* How to benchmark asynchronous, multi-step computations in a **graph-based** manner.
* How to visualize and inspect the entire pipeline.
