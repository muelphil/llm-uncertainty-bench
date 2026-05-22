# Semantic Calibration Prevails Where Token Confidence Fails: Benchmarking Long-Form Scientific QA

This repository contains the code, figures, and analysis scripts accompanying the paper  
**"Semantic Calibration Prevails Where Token Confidence Fails: Benchmarking Long-Form Scientific QA"**.

Uncertainty estimation is critical for reliable deployment of LLMs in scientific applications. This work benchmarks different methods to assess calibration of confidence scores in natural science question answering.

The repository is designed to ensure **full reproducibility** of the experiments and benchmarks conducted in the paper. It provides:
- Benchmark code and configuration files
- Figures and tables generated for the publication
- Scripts and notebooks to re-run benchmarks and reproduce analysis

---

## Repository Structure 📁
```bash
├── async_graph_bench/            # Local benchmarking framework
├── llm-unc-bench/                # Main benchmark project
    ├── label_prob_calibration    # Benchmark 1: Label Probability Calibration
    ├── seq_uc_calibration        # Benchmark 2: Sequence-level Uncertainty Calibration
    └── requirements.txt          # Requirements for running the benchmarks
├── container.def                 # Container definition file
└── pip_freeze.txt                # Exact Python package versions for reference
````

Each of the experiment folders `label\_prob\_calibration` and `seq\_uc\_calibration` contains:
```bash
├── run.py            # main entry point for running the benchmark
├── data_sources/     # datasource implementations containing exact dataset versions used
├── models.py         # model definitions and hardware-specific parameters
├── data/             # created when you run new benchmarks (initially empty)
├── _data/            # contains reference outputs used in the paper, included for reproducibility
├── analysis.ipynb    # notebook for analyzing data and generating plots and tables
└── resources/        # figures and tables generated from analysis
    ├── figures
    └── tables
```

The `data` directory will contain the resulting data after benchmark runs, and is initially left empty. In `_data/` leaf-node outputs from the benchmark run seen in the paper have been included for reference and reproducibility of plots. Rename it to `data` and run the `analysis.ipynb` to reproduce the figures and tables in `resources`.

---

## Environment and Versions 🏷️

Experiments were performed on **NVIDIA H100 GPUs (80GB HBM3)**.  
Different hardware may require adjustments. Consult the `kwargs` properties in `models.py`, which will be supplied to the vllm LLM instances, based on the `run.py --gpu` parameter.

- **Operating System:** Ubuntu 22.04 LTS
- **Python:** 3.12.4
- **CUDA Toolkit:** 12.8 (`nvcc --version`)
- **NVIDIA Driver:** 570.158.01
- **Benchmarking framework:** `async_graph_bench` (installed locally, see below)

For exact python package versions, see [`pip_freeze.txt`](./pip_freeze.txt). These versions reflect the exact setup used in the paper experiments. Newer CUDA or driver versions may work, but adjustments in models.py may be required.

---

## Containerized Setup 🛳️

Alternatively, you can build a container with the required software using the provided definition file (container.def). The file is compatible with Apptainer (recommended) and Singularity (used in some HPC environments).

### Build container

```bash
apptainer build vllm_container.sif container.def
```

### Run container

```bash
apptainer shell --writable --nv \
  --bind /usr/bin/nvidia-smi:/usr/bin/nvidia-smi,/path/to/hf-cache:/hf_home \
  vllm_container.sif
```

Inside the container, set Hugging Face cache:

```bash
export HF_HOME=/hf_home/
```

**Notes:**

* Ensure `--nv` is enabled for GPU passthrough.
* `nvidia-smi` must be accessible inside the container.

---

## Installation 🛠️

### Prerequisites
- NVIDIA GPU with sufficient VRAM (H100 80GB used in experiments)  
  - If using a different GPU, add a corresponding entry in `models.py` under each experiment, including `kwargs` parameters passed to vLLM.  
- CUDA drivers installed (`nvidia-smi` and `nvcc --version` should be available).
- Python 3.12+
- (Optional but recommended) [virtualenv](https://virtualenv.pypa.io/) or [conda](https://docs.conda.io/).


### Step 1: Clone the repository
```bash
git clone <REPO_URL>.git
cd llm-unc-bench
````

### Step 2: Install the benchmarking framework

```bash
cd async_graph_bench
pip install -e .
```

### Step 3: Install benchmark dependencies

```bash
cd ../llm-unc-bench
pip install -r requirements.txt
```

## Running the Benchmarks 🚀

Two main experiments are provided in the `llm-unc-bench` project:

1. **`label_prob_calibration`**
   Investigates the effect of instruction tuning on calibration of label probabilities as confidence scores.

2. **`seq_uc_calibration`**
   Benchmarks calibration of multiple sequence-level uncertainty methods.

### Execution

```bash
cd seq_uc_calibration     # or label_prob_calibration
python run.py --gpu h100 --models Ministral,Llama-3.3 --datasets MMLU,GPQA
```

Use `--help` to inspect available parameters.

**Important:**

* Ensure GPU configuration in `models.py` in the experiments folder matches your hardware.
* Adjust batch size, tensor parallelism, and memory parameters for GPUs with <80GB VRAM.
* Benchmark results are written to the `data/` subfolder of each experiment.

<!-- **Runtime expectations:**

* On H100 (80GB), individual runs typically complete within TODO: \[X–Y hours]. -->

---

## Analysis and Reproducing Figures 📈

Benchmark outputs are analyzed in `analysis.ipynb` notebooks located in each experiment subfolder.

Run:

```bash
jupyter notebook analysis.ipynb
```

By default, figures and tables are written to each experiment’s `resources/` subfolder.

<!-- ---

## Citation

If you use this code in your research, please cite:

```
TODO: BibTeX entry for the paper
```
 -->
---

## License 📜
<!-- TODO -->
This repository is licensed under the MIT License.

---

## Reproducibility Checklist ✅

### Full Benchmark Reproduction
* [ ] Verify GPU accessibility (`nvidia-smi`, `nvcc --version`)
* [ ] Install dependencies (or use container)
* [ ] Install `async-graph-bench` and benchmark requirements
* [ ] Run experiments (`run.py`)
* [ ] Check output in `data/`
* [ ] Run `analysis.ipynb` to generate figures/tables

### Figures Only
* [ ] Install `async-graph-bench` and benchmark requirements (or use container)
* [ ] Rename `_data/` → `data/` in each experiment
* [ ] Run `analysis.ipynb` to regenerate figures/tables