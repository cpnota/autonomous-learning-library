install:
	pip install -e .[dev]
	AutoROM -y --quiet

test: unit-test integration-test

unit-test:
	python -m unittest discover -s all -p "*test.py" -t .

integration-test:
	python -m unittest discover -s integration -p "*test.py"

lint:
	flake8 --ignore "E501,E731,E74,E402,F401,W503,E128" all

format:
	autopep8 --in-place --aggressive --aggressive --ignore "E501,E731,E74,E402,F401,W503,E128" -r all

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
