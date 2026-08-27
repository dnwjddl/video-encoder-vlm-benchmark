STORAGE_ROOT ?= /mnt/disks/data/vlm_encoder_benchmark
HF_HOME ?= /mnt/disks/data/hf_cache
export HF_HOME

.PHONY: install storage data activitynet-debug diagnose perturb-mcq extract train eval

install:
	pip install -e .
	pip install -r requirements.txt

storage:
	bash scripts/setup_storage.sh $(STORAGE_ROOT)

data:
	python scripts/download_data.py \
		--out data/manifests/train_230k.jsonl \
		--streaming \
		--caption-count 100000 \
		--qa-count 100000 \
		--mcq-count 30000 \
		--video-root $(STORAGE_ROOT)/videos

activitynet-debug:
	python scripts/download_activitynet_subset.py \
		--video-dir $(STORAGE_ROOT)/videos/activitynet \
		--out data/manifests/activitynet_debug.jsonl \
		--max-samples 1000 \
		--max-duration 180 \
		--skip-existing

extract:
	bash scripts/run_all_extract.sh data/manifests/train_230k.jsonl features/train_230k

diagnose:
	bash scripts/run_all_no_train_analysis.sh data/manifests/train_debug.jsonl outputs/no_train_diagnostics
	python scripts/aggregate_diagnostics.py \
		--diagnostics-root outputs/no_train_diagnostics \
		--out outputs/no_train_diagnostics_table.csv

perturb-mcq:
	python scripts/make_label_mcq_manifest.py \
		--input data/manifests/activitynet_debug.jsonl \
		--out data/benchmarks/mcq_all.jsonl \
		--num-choices 5
	bash scripts/run_text_aligned_perturbation_mcq.sh data/benchmarks/mcq_all.jsonl outputs/zeroshot_perturbation_mcq
	python scripts/aggregate_perturbation_mcq.py \
		--root outputs/zeroshot_perturbation_mcq \
		--out outputs/zeroshot_perturbation_mcq_table.csv

train:
	bash scripts/run_all_train.sh data/manifests/train_230k.jsonl features/train_230k checkpoints/projectors

eval:
	bash scripts/run_all_eval.sh data/benchmarks/mcq_all.jsonl features/benchmarks checkpoints/projectors outputs/eval
