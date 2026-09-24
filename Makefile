install:
	uv sync

lint:
# 	uv run pylint src/
	uv run nbqa pylint --disable=C,redefined-outer-name --additional-builtins=catalog,display notebooks/

format:
	uv run black notebooks/

jupyter:
	uv run kedro jupyter lab

train:
	kedro run --pipelines=training

experiments:
	uv run run_experiments.py

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db