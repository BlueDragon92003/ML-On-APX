define readme
Please see the README for project information.

----------------------------------------------------------------
Functions available in this project:

install-dependencies	Installs relevant dependencies via pip

run-tests				Runs the test suite to ensure non-model-training code
						behaves as expected.


endef
export readme

_readme:
	@echo "$$readme"

install-dependencies:
	@echo -e "import sys\nif not sys.version_info >= (3,11):\n\tprint('Python version 3.11 or greater is required.')\n\texit(1)" | python3 -
	python3 -m ensurepip
	python3 -m pip freeze | xargs python3 -m pip uninstall
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

run-tests:
	python3 -m tests
