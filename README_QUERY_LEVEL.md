# Query-Level Benchmark

This project includes a benchmarking framework for evaluating UQ methods in LLMs, `async_graph_bench`, which is primarily included so that the benchmark can be run, but it is **not directly relevant** to the insights presented here.
The project includes helper functions for computing metrics and visualizing calibration results, located in `./llm-uncertainty-bench/calibration_visualization`.

The main benchmark is included in `./llm-uncertainty-bench/label_prob_calibration`

---

## Experiments Overview

For our previous benchmark (which we submitted as a paper) we conducted two experiments:

1. **Token-level calibration** – evaluates how well the model’s token-level probabilities align with the correctness of its predicted labels.
2. **Sequence-level calibration** – assesses uncertainty at the level of complete sequences (not the focus of this benchmark release).

### Focus of This Benchmark: Query-Level

This benchmark extends our previous token-level calibration experiment to evaluate **your query-level UQ method**, as proposed in
> Chen, L., de Melo, G., Suchanek, F., & Varoquaux, G. (2025) Query-Level Uncertainty in Large Language Models. arXiv:[2506.09669](https://arxiv.org/abs/2506.09669)

and implemented in [https://github.com/tigerchen52/query_level_uncertainty](https://github.com/tigerchen52/query_level_uncertainty)

The benchmark is located in `./llm-uncertainty-bench/label_prob_calibration`. We did not yet evaluate your method on our sequence level benchmark.

**Setup:**

* A subset of models defined in `models.py` is used as a classifier. For models evaluated, please refer to the figures provided in `resources_query_level/figures`.
* Four datasets are used (defined in `benchmark_datasets.py`): **MMLU, ARC, GSM8K, GPQA**.
* Models are prompted to respond **only with the correct label**.
* The predicted label is extracted as the one with the highest assigned probability.
* In a separate run, your proposed methods are computed on the dataset items.

We ran your method only for a subset of models due to the following issues:

* **Base models**: Errors occurred because your method requires a chat template, which base models do not have.
* **Large Mixture-of-Experts models** were extremely slow (e.g., `Qwen3-30B-A3B-Instruct` took 11.27s to evaluate individual MMLU items on all your proposed methods on average), likely because your method evaluates all experts, not just the activated ones. We have only evaluated `Qwen3-30B-A3B-Instruct` on a subset of the datasets and items.
* **RAM Overflow Issues**: `Llama-3.3-70B-Instruct` was throwing errors, when used with your method out of the box, as there was not enough RAM on the GPUs left, causing memory overflow on `baseline.py:145`:`.to(model.model.device)` in your implementation
* **`google/gemma-3-27b-it`**: Consistently returned `NaN` outputs.
* We did not yet include reasoning models

We generated scores for all methods provided (`max_prob`, `pd_entropy`, etc.), but our analysis focuses on **`internal_confidence`**.

We provided your method with a custom prompt (different from the prompt used for evaluation which choice the models consider most correct), so that your method does not get confused by examples in few-shot prompts (see `nodes/query_level_uncertainty.py`). The prompt template we used to generate prompts we assessed your methods with:
> Below you will see a multiple choice question with multiple answers options. Your task is to select the answer choice you believe is correct by responding with the corresponding label: A, B, C, or D.
> {question}
> A) {option_A}
> B) {option_B}
> C) {option_C}
> D) {option_D}

---

## Benchmark Structure

We reused previously generated data and split the evaluation into two benchmarks:

1. **Label Prediction Evaluation** (`run.py`)

   * Evaluates which choice the model considers most correct for each item.
   * Outputs `LabelProbExtractor.csv` for each model/dataset in `data_query_level`.
   * Logic for data extraction is in `nodes/mc_label_prob.py` and `nodes/label_prob_extraction.py`.

2. **Query-Level Uncertainty Evaluation** (`run_query_level.py`)

   * Evaluates internal confidence using your proposed methods.
   * Outputs `QueryLevelUncertainty.csv` per model/dataset in `data_query_level`.
   * Method is implemented in `nodes/query_level_uncertainty.py` using your package directly.

All generated data is included, so you can perform analysis on it without rerunning the benchmark.

---

## Analysis and Results

* Analysis code: `analysis_query_level.ipynb`
* Generated plots: `resources_query_level/figures`

---

## Implementation Observations and Recommendations

The current implementation requires selecting the method at the **instance level** (`QLUncertainty.__init__`). Users must create separate instances for each method, forcing repeated model inference, which is **computationally inefficient**. We recommend adjusting the code so that the desired method is provided to the `estimate` function instead. You can then cache model inference, so that a single forward pass can be reused for evaluating multiple methods.