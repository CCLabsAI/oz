dev:
	python setup.py build --debug develop

install:
	python setup.py

test:
	python -m unittest

.PHONY: dev install test
