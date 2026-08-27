.PHONY: install data extract train eval

install:
	pip install -e .
	pip install -r requirements.txt

data:
	python scripts/download_data.py \
		--out data/manifests/train_230k.jsonl \
		--streaming \
		--caption-count 100000 \
		--qa-count 100000 \
		--mcq-count 30000

extract:
	bash scripts/run_all_extract.sh data/manifests/train_230k.jsonl features/train_230k

train:
	bash scripts/run_all_train.sh data/manifests/train_230k.jsonl features/train_230k checkpoints/projectors

eval:
	bash scripts/run_all_eval.sh data/benchmarks/mcq_all.jsonl features/benchmarks checkpoints/projectors outputs/eval
