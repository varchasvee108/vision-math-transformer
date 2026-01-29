.PHONY: train setup infer docker-cpu docker-cuda run-cpu run-cuda

train:
	python scripts/train.py --config configs/base.toml

infer:
	python scripts/infer.py --image_path assets/test.png

docker-cpu:
	docker build -f Dockerfile.cpu -t vision-math-transformer:cpu .

docker-cuda:
	docker build -f Dockerfile.cuda -t vision-math-transformer:cuda .

run-cpu:
	docker run --rm -it \
		-v $(PWD):/app \
		vision-math-transformer:cpu

run-cuda:
	docker run --rm -it \
		--gpus all \
		-v $(PWD):/app \
		vision-math-transformer:cuda

setup:
	pip install -e .