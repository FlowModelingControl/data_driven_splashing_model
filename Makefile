.PHONY: test coverage

test:
	python -m unittest

coverage:
	coverage run -m unittest
	coverage report -m $(shell find . -name "*.py" | grep -v "example" | grep -v "tests")
