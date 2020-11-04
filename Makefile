install:
	pip install -e .[dev]

lint:
	pylint all --rcfile=.pylintrc

test:
	python -m unittest discover -s all -p "*test.py"

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
