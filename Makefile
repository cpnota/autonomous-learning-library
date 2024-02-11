install:
	pip install -e .[dev]
	AutoROM -y --quiet

test: unit-test integration-test

unit-test:
	python -m unittest discover -s all -p "*test.py" -t .

integration-test:
	python -m unittest discover -s integration -p "*test.py"

lint:
	black --check all benchmarks examples integration setup.py
	isort --profile black --check all benchmarks examples integration setup.py
	flake8 --select "F401" all benchmarks examples integration setup.py

format:
	black all benchmarks examples integration setup.py
	isort --profile black all benchmarks examples integration setup.py

tensorboard:
	tensorboard --logdir runs

benchmark:
	tensorboard --logdir benchmarks/runs --port=6007

clean:
	rm -rf dist
	rm -rf build

build: clean
	python setup.py sdist bdist_wheel

deploy: lint test build
	twine upload dist/*
