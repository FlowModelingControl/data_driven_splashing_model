.PHONY: test coverage

test:
	python -m unittest discover tests/

coverage:
	coverage run -m unittest discover tests/
	coverage report -m $(shell find . -name "*.py" | grep -v "example" | grep -v "tests")
