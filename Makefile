.PHONY train infer docker-build

train:
	python scripts/train.py --config configs/base.toml

infer:
	python scripts/infer.py --image_path assets/test.png

docker-build:
	docker build -t vision-math-transformer .

setup:
	pip install -e