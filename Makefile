install:
	pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
	pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
	pip install tensorflow
	pip install -e .

lint:
	pylint all --rcfile=.pylintrc

test:
	python -m unittest discover -s all -p "*test.py"

tensorboard:
	tensorboard --logdir runs

benchmark:
	tensorboard --logdir benchmarks/runs --port=6007
