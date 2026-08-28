STORAGE_ROOT ?= /mnt/disks/data/vlm_encoder_benchmark
HF_HOME ?= /mnt/disks/data/hf_cache
PILOT_MCQ ?= data/benchmarks/hf_video_debug_mcq.jsonl
export HF_HOME

.PHONY: install storage doctor data activitynet-debug video-debug hf-video-debug kinetics700-debug diagnose diagnose-parallel perturb-mcq pilot-mcq train-pilot extract train eval

install:
	pip install -e .
	pip install -r requirements.txt

storage:
	bash scripts/setup_storage.sh $(STORAGE_ROOT)

doctor:
	python scripts/check_runtime.py

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

video-debug: hf-video-debug

hf-video-debug:
	python scripts/download_hf_video_dataset.py \
		--dataset-id VLM2Vec/mvbench-FunQA_test \
		--split test \
		--source-mode video-column \
		--video-column video \
		--label-column label \
		--video-dir $(STORAGE_ROOT)/videos/mvbench_funqa_debug \
		--out data/manifests/hf_video_debug.jsonl \
		--max-samples 358 \
		--validate

kinetics700-debug:
	@echo "VLM2Vec/Kinetics-700 is metadata-only in HF; using HF rawvideo debug dataset instead."
	$(MAKE) hf-video-debug

extract:
	bash scripts/run_all_extract.sh data/manifests/train_230k.jsonl features/train_230k

diagnose:
	test -s data/manifests/hf_video_debug.jsonl || $(MAKE) hf-video-debug
	bash scripts/run_all_no_train_analysis.sh data/manifests/hf_video_debug.jsonl outputs/no_train_diagnostics
	python scripts/aggregate_diagnostics.py \
		--diagnostics-root outputs/no_train_diagnostics \
		--out outputs/no_train_diagnostics_table.csv

diagnose-parallel:
	test -s data/manifests/hf_video_debug.jsonl || $(MAKE) hf-video-debug
	bash scripts/run_parallel_no_train_analysis.sh \
		data/manifests/hf_video_debug.jsonl \
		outputs/no_train_diagnostics \
		runs/no_train_diagnostics

perturb-mcq:
	test -s data/manifests/hf_video_debug.jsonl || $(MAKE) hf-video-debug
	python scripts/make_label_mcq_manifest.py \
		--input data/manifests/hf_video_debug.jsonl \
		--out data/benchmarks/mcq_all.jsonl \
		--num-choices 3 \
		--benchmark-name hf_video_label_mcq
	bash scripts/run_text_aligned_perturbation_mcq.sh data/benchmarks/mcq_all.jsonl outputs/zeroshot_perturbation_mcq
	python scripts/aggregate_perturbation_mcq.py \
		--root outputs/zeroshot_perturbation_mcq \
		--out outputs/zeroshot_perturbation_mcq_table.csv

pilot-mcq:
	test -s data/manifests/hf_video_debug.jsonl || $(MAKE) hf-video-debug
	python scripts/make_label_mcq_manifest.py \
		--input data/manifests/hf_video_debug.jsonl \
		--out $(PILOT_MCQ) \
		--num-choices 3 \
		--benchmark-name hf_video_label_mcq

train-pilot: pilot-mcq
	bash scripts/run_parallel_pilot_train.sh \
		$(PILOT_MCQ) \
		features/pilot_train \
		checkpoints/pilot_projectors \
		runs/pilot_train

train:
	bash scripts/run_all_train.sh data/manifests/train_230k.jsonl features/train_230k checkpoints/projectors

eval:
	bash scripts/run_all_eval.sh data/benchmarks/mcq_all.jsonl features/benchmarks checkpoints/projectors outputs/eval
