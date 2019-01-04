install:
	pip install -e .

lint:
	pylint all --rcfile=.pylintrc

test:
	python -m unittest discover -s all -p "*test.py"
