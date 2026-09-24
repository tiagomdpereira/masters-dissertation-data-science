# Masters Dissertation - Data Science

## Overview

This repository contains the code, data, and documentation for my **Master's Dissertation in Data Science**. The project is built using the **Kedro** framework, which helps structure data pipelines, manage dependencies, and ensure reproducibility.

---

## Project Structure

- **`src/`**: Source code for the dissertation, including data processing, modeling, and analysis.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and experiments.
- **`data/`**: 
  - **`03_primary/`**: Must contain all **IMAD-DS dataset files**, including the existing `dataset_splitted.csv`.  
  The folder for Parquet files must be named **`time_series_data`**.
- **`conf/`**: Configuration files for Kedro pipelines and project settings.
- **`tests/`**: Unit and integration tests for the project. It is not used.
- **`pyproject.toml`**: Project dependencies, metadata and additional configurations.
- **`run_experiments.py`**: Script to automatically run training and evaluation pipelines.

---

## How to Run

1. **Install Dependencies**
  ```bash
   uv sync
  ```
2. **Run the Kedro Pipeline**
  ```bash
   make experiments
  ```

---

## How to Use Notebooks

  ```bash
  make jupyter
  ```

---

## About

This project is part of my **Master's Dissertation in Data Science**. It follows best practices for reproducibility, modularity, and clean code.