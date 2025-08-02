
# Encoder Models Performance Analysis ⏱️📈

> **Measure batch-inference latency of BERT-class encoder models across CPU, NVIDIA CUDA, Apple-Silicon MPS, and AWS Inferentia / Trainium devices – with one command.**

---

## 1 · Clone

```bash
https://github.com/Suranjit/encoder_perf_analysis.git
cd latency-bench
```

## 2 · Python & Poetry

| Host                | Python version | Poetry install                                               |
|---------------------|---------------|--------------------------------------------------------------|
| **Mac (M-series)**  | 3.11 + (brew / pyenv) | `curl -sSL https://install.python-poetry.org | python3 -` |
| **Linux CPU / GPU** | 3.10 or 3.11  | same                                                         |
| **AWS Inf2 / Trn1** | 3.9 (included in Neuron DLAMI) | same                                                         |

```bash
# make Poetry visible
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

## 3 · Install dependencies

Choose an **extra** that matches your hardware (combine with `-E dev` for linters etc.).

| Extra  | Command | Notes |
|--------|---------|-------|
| CPU-only            | `poetry install -E cpu` | installs onnxruntime-cpu |
| Mac MPS             | `poetry install -E mac` | universal PyTorch wheel |
| CUDA GPU            | `poetry install -E gpu` <br>then add matching CUDA wheel:<br>`poetry run pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 -f https://download.pytorch.org/whl/cu121` |
| Inferentia v1       | `poetry install -E inf1` | run on **Inf1** DLAMI only |
| Inferentia v2 / Trainium | *(separate venv – Torch 1.13)*<br>`python3 -m venv neuron-env && source neuron-env/bin/activate`<br>`pip install torch-neuronx==2.13.* optimum-neuron[neuronx]==0.0.19 wandb` |

## 4 · Optional: Weights & Biases

```bash
poetry run wandb login      # or export WANDB_API_KEY=...
```

## 5 · Run the benchmark

### Quick smoke test (CPU)

```bash
poetry run python -m latency_bench.benchmark   --suite base --devices cpu
```

### Full Mac sweep with W&B logging

```bash
poetry run python -m latency_bench.benchmark   --suite base --devices cpu mps   --seq-lens 64 128 256 512   --batch-sizes 1 2 4 8 16 32   --dtype bf16   --wandb-project encoder-perf   --wandb-run mac-mps-sweep
```

### GPU example

```bash
poetry run python -m latency_bench.benchmark   --suite base --devices cpu cuda   --batch-sizes 1 2 4 8 16 32 64   --wandb-project encoder-perf   --wandb-run g6-l4
```

### AWS Inf2 – after compiling

```bash
source neuron-env/bin/activate
python -m latency_bench.compile --suite base --out compiled_models
python -m latency_bench.benchmark   --models compiled_models/bert-base-uncased   --devices inf2 --batch-sizes 1 2 4 8 16 32   --seq-lens 128 256 512   --wandb-project encoder-perf   --wandb-run inf2-bench
```

## 6 · Results

* **CSV** → `runs/bench-<timestamp>.csv`
* **Weights & Biases** → a run containing `latency_table` with columns  
  `model · device_type · batch_size · seq_len · latency_ms` – plot any axis you need.

## 7 · Troubleshooting

| Symptom                                         | Fix |
|-------------------------------------------------|-----|
| `wandb not installed`                           | `poetry add -E tracking wandb` or install in venv |
| `decoder_input_ids` error on T5                 | Benchmark encoder-only models or pass dummy decoder inputs |
| CUDA `OSError: libcublas.so`                    | Host driver must match CUDA wheel version |
| Torch 2 vs Neuron conflict                      | Keep Neuron deps in a **separate venv** |

---
