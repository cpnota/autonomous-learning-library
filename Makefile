lint:
	pylint all --rcfile=.pylintrc

test:
	python -m unittest discover -s all -p "*_test.py"
