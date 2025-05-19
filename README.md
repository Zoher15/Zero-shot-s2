# Zero-shot-s2: Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models

## Overview

[Briefly describe your project: What problem does it solve? What is the main contribution? What are the key findings if it's a research project? (e.g., This repository contains the code and experimental results for the paper "Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models". We explore methods to enhance the ability of Vision-Language Models (VLMs) to distinguish between real and AI-generated images in a zero-shot setting.)]

## Table of Contents

- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Download NLTK Resources](#4-download-nltk-resources)
  - [5. Data Preparation](#5-data-preparation)
- [Usage](#usage)
  - [Configuration](https://www.google.com/search?q=%23configuration)
  - [Running Experiments](#running-experiments)
    - [Example: Evaluating Qwen 2.5 7B on GenImage](#example-evaluating-qwen-25-7b-on-genimage)
  - [Generating Result Tables and Plots](#generating-result-tables-and-plots)
    - [Example: Generating the scaling consistency plot](#example-generating-the-scaling-consistency-plot)
  - [Downloading and Preprocessing D3 Dataset Images](#downloading-and-preprocessing-d3-dataset-images)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Repository Structure

Provide a brief overview of the main directories and their purpose:
```

Zero-shot-s2/
├── experiments/         \# Scripts for running evaluations and data processing
│   ├── evaluate\_AI\_llama.py
│   ├── evaluate\_AI\_qwen.py
│   └── load\_d3.py
├── results/             \# Scripts for generating tables and plots from experiment outputs
│   ├── combine\_tables.py
│   ├── distinct\_words.py
│   ├── find\_images.py
│   ├── macro\_f1\_bars.py
│   ├── model\_size\_table.py
│   ├── prompt\_table.py
│   ├── recall\_subsets\_table.py
│   └── scaling\_consistency.py
├── data/                \# Placeholder for input data (not committed, managed via .gitignore)
│   ├── D3/
│   ├── DF40/
│   └── genimage/
├── outputs/             \# Placeholder for generated results, plots, tables (some may be cached here)
│   ├── responses/
│   ├── scores/
│   ├── plots/
│   └── tables/
├── utils/               \# (Suggested) Utility scripts and shared functions
├── config/              \# (Suggested) Configuration files
├── .gitignore
├── LICENSE.md           \# (To be added)
├── README.md
└── requirements.txt     \# (To be added)

````

## Prerequisites

* Python (specify version, e.g., 3.8+)
* pip (Python package installer)
* (Optional) Conda for environment management
* (Optional) NVIDIA GPU with CUDA installed for model inference (specify CUDA version if critical)

## Setup

### 1. Clone the Repository
```bash
git clone https://your-repo-url/Zero-shot-s2.git
cd Zero-shot-s2
````

### 2\. Create a Virtual Environment

It is highly recommended to use a virtual environment.

**Using `venv`:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Using `conda`:**

```bash
conda create -n zeroshot_s2 python=3.9  # Or your preferred Python version
conda activate zeroshot_s2
```

### 3\. Install Dependencies

[We will create a `requirements.txt` file in the next step. Once created, the instruction will be:]

```bash
pip install -r requirements.txt
```

### 4\. Download NLTK Resources

Some scripts (e.g., `results/distinct_words.py`) require NLTK resources. The script attempts to download them if not found, but you can also pre-download them:

```bash
python -m nltk.downloader wordnet stopwords punkt
```

Or run the `results/distinct_words.py` script once, which has a built-in checker.

### 5\. Data Preparation

This project requires several datasets. Due to their size, they are not included in the repository. You need to download them and organize them as follows (or update paths in the configuration, see [Configuration](https://www.google.com/search?q=%23configuration) section):

```
Zero-shot-s2/
└── data/
    ├── D3/                      # Directory for D3 images (see experiments/load_d3.py)
    │   └── D3_2k_sample.csv     # Example CSV for D3 image URLs
    ├── DF40/
    │   ├── 10k_sample_df40.csv
    │   └── 2k_sample_df40.csv
    ├── genimage/
    │   ├── 10k_random_sample.csv
    │   └── 2k_random_sample.csv
    └── FACES/                   # For FACES dataset if used by evaluate_AI_qwen.py/llama.py
        ├── LD_raw_512Size/
        ├── StyleGAN_raw_512size/
        └── Real_512Size/

```

  * **D3 Dataset:** The `experiments/load_d3.py` script is used to download images based on a CSV file (e.g., `D3_2k_sample.csv`) containing image URLs.
  * **Other Datasets (DF40, GenImage, FACES):** Place the respective CSV files and image directories as shown.
  * **Large data files/directories within `data/` should be added to your `.gitignore` file.**

[Add specific download links or instructions for each dataset if available/permissible.]

## Usage

### Configuration

Many scripts use hardcoded paths and parameters. In future versions, these will be moved to a central configuration system. For now, you might need to adjust paths at the top of individual scripts if your data is not in the default locations mentioned under "Data Preparation".

Key scripts and their primary data/output locations:

  * Evaluation scripts (`experiments/evaluate_AI_*.py`) save rationales and scores to paths like `/data3/zkachwal/visual-reasoning/data/ai-generation/responses/` and `/data3/zkachwal/visual-reasoning/data/ai-generation/scores/`. These should ideally be configured to save under the project's `outputs/` directory.
  * Result generation scripts (`results/*.py`) often read from these locations and may save plots/tables to the `results/` directory itself or subdirectories (e.g., `results/f1_cache_plot/`).

### Running Experiments

The `experiments` directory contains scripts to run evaluations.

**Example: Evaluating Qwen 2.5 7B on the GenImage dataset (2k sample)**

```bash
python experiments/evaluate_AI_qwen.py \
    -llm qwen25-7b \
    -c 0 \
    -d genimage2k \
    -b 20 \
    -n 1 \
    -m zeroshot-2-artifacts
```

  * `-llm`: Model name (e.g., `qwen25-7b`, `llama3-11b`).
  * `-c`: CUDA device ID.
  * `-d`: Dataset (e.g., `genimage`, `genimage2k`, `d3`, `d32k`, `df40`, `df402k`, `faces`).
  * `-b`: Batch size.
  * `-n`: Number of sequences for the model.
  * `-m`: Mode of reasoning (e.g., `zeroshot`, `zeroshot-cot`, `zeroshot-2-artifacts`).

Refer to the `argparse` section within each evaluation script for all available options.

### Generating Result Tables and Plots

The `results` directory contains scripts to process the output of experiments and generate tables/plots. Ensure that the experiment outputs (rationales, scores) are available in the locations these scripts expect (currently often hardcoded to `/data3/zkachwal/...`).

**Example: Generating the scaling consistency plot**
(Assumes `TARGET_LLAMA_MODEL_NAME = "llama3-11b"` and relevant rationale files are present)

```bash
python results/scaling_consistency.py
```

This will generate `self_consistency_scaling.png`.

### Downloading and Preprocessing D3 Dataset Images

The `experiments/load_d3.py` script downloads and saves images.

```bash
python experiments/load_d3.py \
    --csv_filepath path/to/your/D3_sample.csv \
    --save_directory data/D3/images \
    --timeout 15 \
    --force  # Optional: to overwrite existing images
    --verbose # Optional: for debug logging
```

  * Update `--csv_filepath` and `--save_directory` as needed. The script will create the save directory if it doesn't exist.
  * A log file `processing_log.log` will be created in the directory where the script is run.

## Expected Outputs

  * **Experiment Scripts:** Generate JSONL files with detailed rationales and JSON/CSV files with scores, typically saved to specified output directories (currently hardcoded).
  * **Result Scripts:** Generate `.png` plots, `.tex` table files, or print tables to the console. These are usually saved in the `results/` directory or a subdirectory.

## Troubleshooting

  * **`FileNotFoundError`:**
      * Double-check hardcoded paths in the scripts if you haven't configured a central path management system.
      * Ensure your datasets are organized as specified in "Data Preparation".
  * **`ModuleNotFoundError`:**
      * Make sure your virtual environment is activated.
      * Verify that all dependencies from `requirements.txt` are installed.
  * **CUDA Errors / GPU Issues:**
      * Ensure PyTorch is installed with the correct CUDA version matching your system's CUDA drivers.
      * Check if the correct GPU is visible and selected (`-c` argument in evaluation scripts).
  * **NLTK `LookupError`:**
      * Run the NLTK resource download command (see Setup) or let the `distinct_words.py` script attempt the download.

## License

[To be added. Choose a license, e.g., MIT, Apache 2.0, and add a `LICENSE.md` file.]
Example: This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this code or these findings in your research, please cite:

```
[Your BibTeX entry here]
```

## Contributing

[Optional: Add guidelines for contributing if you welcome contributions.]
Examples:

  * Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
  * Please make sure to update tests as appropriate.

## Acknowledgements

[Optional: Acknowledge any libraries, datasets, or individuals that helped your project.]