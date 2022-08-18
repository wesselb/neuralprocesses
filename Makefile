.PHONY: install test

PACKAGE := neuralprocesses

install:
	pip install -r requirements.txt -e .

test:
	python setup.py --version
	pytest -v --cov=$(PACKAGE) --cov-report html:cover --cov-report term-missing
