PKG_SRC_DIR=src/openai_lib

build: clean
	python -m build
	pip install dist/openai_lib-0.0.0-py3-none-any.whl
clean:
	rm -rf $(PKG_SRC_DIR)/__pycache__ $(PKG_SRC_DIR)/*~ $(PKG_SRC_DIR)/*.egg_info/
	pip uninstall -r requirements.txt
	pip uninstall openai_lib
	rm -rf dist/
