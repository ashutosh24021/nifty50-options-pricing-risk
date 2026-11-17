.PHONY: install demo test check clean
install:
	python -m pip install -e ".[dev]"
demo:
	optionlab demo --n-paths 100000 --seed 42 --out results/demo
test:
	pytest -q
check:
	python -m compileall -q src
	pytest -q
clean:
	rm -rf results/demo results/ci-demo
