install:
	uv sync

lint:
# 	uv run pylint src/
	uv run nbqa pylint --disable=C,redefined-outer-name --additional-builtins=catalog,display notebooks/

format:
	uv run black notebooks/

jupyter:
	uv run kedro jupyter lab


