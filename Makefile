.PHONY: install data diagnose extract train eval

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

diagnose:
	bash scripts/run_all_no_train_analysis.sh data/manifests/train_debug.jsonl outputs/no_train_diagnostics
	python scripts/aggregate_diagnostics.py \
		--diagnostics-root outputs/no_train_diagnostics \
		--out outputs/no_train_diagnostics_table.csv

train:
	bash scripts/run_all_train.sh data/manifests/train_230k.jsonl features/train_230k checkpoints/projectors

eval:
	bash scripts/run_all_eval.sh data/benchmarks/mcq_all.jsonl features/benchmarks checkpoints/projectors outputs/eval
