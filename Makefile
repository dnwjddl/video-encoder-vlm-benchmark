STORAGE_ROOT ?= /mnt/disks/data/vlm_encoder_benchmark
HF_HOME ?= /mnt/disks/data/hf_cache
LLM_ID ?= Qwen/Qwen2.5-7B-Instruct
PILOT_MCQ ?= data/benchmarks/hf_video_debug_mcq.jsonl
TRAIN5K_MANIFEST ?= data/manifests/train_5k_msrvtt.jsonl
TRAIN20K_MANIFEST ?= data/manifests/train_20k_msrvtt.jsonl
TRAIN20K_EPOCHS ?= $(or $(EPOCHS),2)
TRAIN_GPUS ?= 0,1,2,3
ALL_ENCODERS ?= clip-vit-l-14-336,siglip-so400m,siglip2-so400m,dinov2-vitl14,internvit-300m,videomaev2-base,vjepa2-vith-256,internvideo2-clip-s
MVBENCH_ROOT ?= /home/woojunghan_google_com/hf_cache/mvbench_video
MVBENCH_MANIFEST ?= data/benchmarks/mvbench_all.jsonl
MVBENCH_VALIDATE_MEDIA ?= 0
MVBENCH_SKIP_MISSING_MEDIA ?= 1
MVBENCH_ANALYZE_SKIP_INCOMPLETE ?= 1
ENCODER ?= internvit-300m
GPU ?= 0
DIAGNOSTICS_TABLE ?= outputs/no_train_diagnostics_table.csv
DIAGNOSTICS_FIGURE ?= outputs/figures/no_train_diagnostics_overview
export HF_HOME

.PHONY: install storage doctor doctor-model check-encoder smoke-encoder check-problem-encoders check-flash-attn check-llm download-llm download-internvideo2-clip-s data train-manifest-5k train-manifest-20k mvbench-manifest mvbench-eval mvbench-analyze mvbench-overall activitynet-debug video-debug hf-video-debug kinetics700-debug diagnose diagnose-parallel figure-diagnostics perturb-mcq pilot-mcq train-pilot train-5k train-20k train-20k-all extract train eval

install:
	pip install -e .
	pip install -r requirements.txt

storage:
	bash scripts/setup_storage.sh $(STORAGE_ROOT)

doctor:
	python scripts/check_runtime.py

doctor-model:
	python scripts/check_runtime.py --load-model

check-encoder:
	CUDA_VISIBLE_DEVICES=$(GPU) python scripts/check_encoder_runtime.py --encoder $(ENCODER)

smoke-encoder:
	CUDA_VISIBLE_DEVICES=$(GPU) python scripts/check_encoder_runtime.py --encoder $(ENCODER) --forward

check-problem-encoders:
	$(MAKE) check-encoder ENCODER=internvit-300m GPU=$(GPU)
	$(MAKE) check-encoder ENCODER=videomaev2-base GPU=$(GPU)
	$(MAKE) check-encoder ENCODER=internvideo2-clip-s GPU=$(GPU)

check-flash-attn:
	python scripts/check_runtime.py \
		--model-id OpenGVLab/InternVideo2_CLIP_S \
		--load-model \
		--allow-missing-processor \
		--required-module open_clip

check-llm:
	python scripts/check_hf_model_cache.py --model-id $(LLM_ID) --trust-remote-code

download-llm:
	VLMEB_LOCAL_FILES_ONLY=0 python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("$(LLM_ID)", repo_type="model"))'

download-internvideo2-clip-s:
	VLMEB_LOCAL_FILES_ONLY=0 python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("OpenGVLab/InternVideo2_CLIP_S", repo_type="model"))'

data:
	python scripts/download_data.py \
		--out data/manifests/train_230k.jsonl \
		--streaming \
		--caption-count 100000 \
		--qa-count 100000 \
		--mcq-count 30000 \
		--video-root $(STORAGE_ROOT)/videos

train-manifest-5k:
	python scripts/check_jsonl_rows.py --path $(TRAIN5K_MANIFEST) --min-rows 5000 || python scripts/download_hf_video_dataset.py \
		--dataset-id VLM2Vec/MSR-VTT \
		--config-name train_7k \
		--split train \
		--source-mode path-column \
		--video-path-column video \
		--path-prefix raw_videos \
		--id-column id \
		--label-column category \
		--caption-column caption \
		--video-dir $(STORAGE_ROOT)/videos/msrvtt_train_5k \
		--out $(TRAIN5K_MANIFEST) \
		--max-samples 5000 \
		--validate

train-manifest-20k:
	python scripts/check_jsonl_rows.py --path $(TRAIN20K_MANIFEST) --min-rows 20000 || python scripts/download_hf_video_dataset.py \
		--dataset-id VLM2Vec/MSR-VTT \
		--config-name train_9k \
		--split train \
		--source-mode path-column \
		--video-path-column video \
		--path-prefix raw_videos \
		--id-column id \
		--label-column category \
		--caption-column caption \
		--caption-expand-count 3 \
		--video-dir $(STORAGE_ROOT)/videos/msrvtt_train_20k \
		--out $(TRAIN20K_MANIFEST) \
		--max-samples 20000 \
		--validate

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

figure-diagnostics:
	test -s $(DIAGNOSTICS_TABLE) || python scripts/aggregate_diagnostics.py \
		--diagnostics-root outputs/no_train_diagnostics \
		--out $(DIAGNOSTICS_TABLE)
	python scripts/plot_no_train_diagnostics.py \
		--input $(DIAGNOSTICS_TABLE) \
		--out-prefix $(DIAGNOSTICS_FIGURE)

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

train-5k: train-manifest-5k
	bash scripts/run_parallel_pilot_train.sh \
		$(TRAIN5K_MANIFEST) \
		features/train_5k \
		checkpoints/projectors_5k \
		runs/train_5k

train-20k: train-manifest-20k
	EPOCHS=$(TRAIN20K_EPOCHS) bash scripts/run_parallel_pilot_train.sh \
		$(TRAIN20K_MANIFEST) \
		features/train_20k \
		checkpoints/projectors_20k \
		runs/train_20k

train-20k-all: train-manifest-20k
	EPOCHS=$(TRAIN20K_EPOCHS) \
	GPUS=$(if $(GPUS),$(GPUS),$(TRAIN_GPUS)) \
	ENCODERS=$(ALL_ENCODERS) \
	SCHEDULER=dynamic \
	bash scripts/run_parallel_pilot_train.sh \
		$(TRAIN20K_MANIFEST) \
		features/train_20k \
		checkpoints/projectors_20k \
		runs/train_20k

mvbench-manifest:
	python scripts/build_mvbench_manifest.py --mvbench-root $(MVBENCH_ROOT) --out $(MVBENCH_MANIFEST) $(if $(filter 1,$(MVBENCH_SKIP_MISSING_MEDIA)),--skip-missing-media,) $(if $(filter 1,$(MVBENCH_VALIDATE_MEDIA)),--validate-media,)

mvbench-eval: mvbench-manifest
	bash scripts/run_parallel_mvbench_eval.sh \
		$(MVBENCH_MANIFEST) \
		features/mvbench \
		checkpoints/projectors_20k \
		outputs/mvbench \
		runs/mvbench_eval

mvbench-analyze:
	python scripts/aggregate_mvbench_filters.py \
		--manifest $(MVBENCH_MANIFEST) \
		--text-predictions outputs/mvbench/text_only/predictions.jsonl \
		--eval-root outputs/mvbench/projector_eval \
		--out-dir outputs/mvbench/analysis \
		--encoders $(ALL_ENCODERS) \
		$(if $(filter 1,$(MVBENCH_ANALYZE_SKIP_INCOMPLETE)),--skip-incomplete,)

mvbench-overall:
	python scripts/aggregate_mvbench_overall.py \
		--manifest $(MVBENCH_MANIFEST) \
		--eval-root outputs/mvbench/projector_eval \
		--out outputs/mvbench/analysis/overall_accuracy.csv \
		--encoders $(ALL_ENCODERS)

train:
	bash scripts/run_all_train.sh data/manifests/train_230k.jsonl features/train_230k checkpoints/projectors

eval:
	bash scripts/run_all_eval.sh data/benchmarks/mcq_all.jsonl features/benchmarks checkpoints/projectors outputs/eval
