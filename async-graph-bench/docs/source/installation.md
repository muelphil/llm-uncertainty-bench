# Installation Guide

This framework can be installed either directly from GitHub or via PyPI.

## Installation Options

### 1. From GitHub

```bash
pip install git+https://github.com/USERNAME/REPO_NAME.git TODO
```

### 2. From PyPI

```bash
pip install PACKAGE_NAME TODO
```
---

## Requirements

* **Python 3.11+** is required.

Before installation, ensure you have a compatible Python environment (e.g., using `venv` or `conda`).

---

## Optional Dependencies

Some features require additional packages depending on the models or visualization tools you plan to use:

| Feature                 | Required Packages | Purpose                                     |
| ----------------------- | ----------------- |---------------------------------------------|
| **vLLM model support**  | `vllm`, `torch`   | Required for running vLLM-based models      |
| **OpenAI API support**  | `openai`          | Required for querying OpenAI API endpoints  |
| **Graph visualization** | `graphviz`        | Required for rendering execution graphs     |

You can install these manually as needed, for example:

```bash
pip install vllm torch
pip install openai
pip install graphviz
```

[//]: # (Alternatively, install all optional dependencies with:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install PACKAGE_NAME[all])

[//]: # (```)
