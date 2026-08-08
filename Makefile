install:
	uv sync

lint:
# 	uv run pylint src/
	uv run nbqa pylint --disable=C notebooks/

format:
	uv run black notebooks/

jupyter:
	uv run kedro jupyter lab


