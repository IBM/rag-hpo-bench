# Installation Guide for RAG-HPO-Bench

This guide provides instructions for setting up the `rag-hpo-bench` project using the `uv` package manager.

## Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installing uv

If you don't have `uv` installed, you can install it using one of these methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

## Installation Steps

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/IBM/rag-hpo-bench.git
cd rag-hpo-bench
```

### 2. Create a Virtual Environment

Create a virtual environment using `uv`:

```bash
uv venv .venv --python 3.11
```

### 3. Activate the Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

### 4. Install the Package

Install the package in editable mode:

```bash
uv pip install -e .
```

## Running Examples

After installation, you can run the example scripts:

```bash
cd examples
python find_best_configs.py
```

## Development Setup

If you plan to contribute or modify the code, install with development dependencies:

```bash
uv pip install -e ".[dev]"
```

### Setting up Pre-commit Hooks (Optional)

```bash
pre-commit install