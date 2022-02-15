
# Server options
SERVER_IP=narval
SERVER_PATH=~/scratch/open-set
USER=mboudiaf

# Simu options
DATADIR=data
SRC_DATASET=mini_imagenet
TGT_DATASETS=$(SRC_DATASET)
DETECTORS=knn
PREPOOL=trivial
POSTPOOL=debiased_centering l2_norm
LAYERS=4_4
COMBIN=1
EXP=default
RESOLUTION=84
BACKBONE=resnet12
MODEL_SRC=feat
TRAINING=standard
DEBUG=False
GPUS=0
SIMU_PARAMS=current_sequence
OVERRIDE=False
MODE=benchmark

# Tasks
N_TASKS=10000
SHOTS=1 5
BALANCED=True
ABLATION_ARG=alpha
ABLATION_VAL=1.0

train:
	SHOTS=1
	for dataset in $(DATASETS); do \
		for shot in $(SHOTS); do \
		    python3 -m src.pretrain \
		        --exp_name $${dataset}'-'$(EXP) \
		        --data_dir $(DATADIR) \
		        --inference_method SimpleShot \
		        --n_tasks 500 \
		        --n_shot $${shot} \
		        --prepool_transform  $(PREPOOL) \
		        --postpool_transform  $(POSTPOOL) \
		        --pool \
		        --backbone $(BACKBONE) \
		        --dataset $${dataset} \
		        --debug $(DEBUG) \
		        --gpus $(GPUS) ;\
	    done ;\
	done ;\

extract:
		for dataset in $(TGT_DATASETS); do \
		    for split in train  test; do \
				python -m src.compute_features \
					--backbone $(BACKBONE) \
					--src_dataset $(SRC_DATASET) \
					--tgt_dataset $${dataset} \
					--data_dir $(DATADIR) \
			        --model_source $(MODEL_SRC) \
			        --training $(TRAINING) \
					--split $${split} \
					--layers $(LAYERS) ;\
		    done \
		done \

run:
	for dataset in $(TGT_DATASETS); do \
		for detector in $(DETECTORS); do \
			for shot in $(SHOTS); do \
			    python3 -m src.inference_features \
			        --exp_name $(EXP)'-'$${shot}'-'$${dataset} \
			        --mode 'tune' \
			        --inference_method SimpleShot \
			        --n_tasks $(N_TASKS) \
			        --n_shot $${shot} \
			        --layers $(LAYERS) \
			        --outlier_detectors $${detector} \
			        --prepool_transform  $(PREPOOL) \
			        --postpool_transform  $(POSTPOOL) \
			        --pool \
			        --aggreg l2_bar \
			        --backbone $(BACKBONE) \
			        --model_source $(MODEL_SRC) \
			        --balanced $(BALANCED) \
			        --training $(TRAINING) \
					--src_dataset $(SRC_DATASET) \
					--tgt_dataset $${dataset} \
			        --simu_hparams $(SIMU_PARAMS) \
			        --combination_size $(COMBIN) \
			        --$(ABLATION_ARG) $(ABLATION_VAL) \
			        --override $(OVERRIDE) \
			        --mode $(MODE) ;\
		    done ;\
		done ;\
	done ;\

run_scratch:
	for dataset in $(DATASETS); do \
		for detector in $(DETECTORS); do \
			for shot in $(SHOTS); do \
			    python3 -m src.inference \
			        --exp_name $(EXP)'-'$${shot}'-'$${dataset} \
			        --mode 'tune' \
			        --inference_method SimpleShot \
			        --n_tasks 500 \
			        --n_shot $${shot} \
			        --layers $(LAYERS) \
			        --image_size $(RESOLUTION) \
			        --outlier_detectors $${detector} \
			        --prepool_transform  $(PREPOOL) \
			        --postpool_transform  $(POSTPOOL) \
			        --pool \
			        --aggreg l2_bar \
			        --backbone $(BACKBONE) \
			        --model_source feat \
			        --dataset $${dataset} \
			        --simu_hparams 'current_sequence' \
			        --combination_size $(COMBIN) \
			        --override $(OVERRIDE);\
		    done ;\
		done ;\
	done ;\

# ========== Extraction pipelines ===========

extract_standard:
	make SRC_DATASET=mini_imagenet TGT_DATASETS=cub extract ;\
	make SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet extract ;\
	make SRC_DATASET=tiered_imagenet TGT_DATASETS=tiered_imagenet extract ;\
	make SRC_DATASET=tiered_imagenet TGT_DATASETS=cub extract ;\

extract_snatcher:
	make TRAINING='feat' MODEL_SRC='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=cub extract ;\
# 	make TRAINING='feat' MODEL_SRC='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet extract ;\
# 	make TRAINING='feat' MODEL_SRC='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=tiered_imagenet extract ;\
# 	make TRAINING='feat' MODEL_SRC='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=cub extract ;\

snatcher_run:
	make PREPOOL=trivial POSTPOOL=trivial DETECTORS='snatcher_f' TRAINING='feat' MODEL_SRC='feat' run ;\

# ========== Benchmarking ===========

baseline:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=cub POSTPOOL="debiased_centering l2_norm" run ;\
# 	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=tiered_imagenet TGT_DATASETS=cub POSTPOOL="debiased_centering l2_norm" run ;\
# 	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="debiased_centering l2_norm" run ;\
# 	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=tiered_imagenet TGT_DATASETS=tiered_imagenet POSTPOOL="debiased_centering l2_norm" run ;\

oracle:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="oracle_centering l2_norm" run ;\

debiased:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="debiased_centering l2_norm" run ;\

kcenter:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="kcenter_centering l2_norm" run ;\

t_center:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="transductive_centering l2_norm" run ;\

tarjan:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="tarjan_centering l2_norm" run ;\

protorect:
	make EXP=debiased_centering PREPOOL=trivial SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet POSTPOOL="protorect_centering l2_norm" run ;\

snatcher:
	make EXP=snatcher SRC_DATASET=mini_imagenet TGT_DATASETS=cub snatcher_run ;\


layer_mixing:
	make EXP=layer_mixing COMBIN=3 run ;\

# ========== Ablations ===========

alpha_influence:
	for alpha in 0.5 1.0 2.0 3.0 4.0 5.0; do \
		make SHOTS=1 BALANCED=False EXP=influence_alpha_debiased SIMU_PARAMS="current_sequence alpha" POSTPOOL="debiased_centering l2_norm" ABLATION_ARG=alpha ABLATION_VAL=$${alpha} run; \
		make SHOTS=1 BALANCED=False EXP=influence_alpha_biased SIMU_PARAMS="current_sequence alpha" POSTPOOL="transductive_centering l2_norm" ABLATION_ARG=alpha ABLATION_VAL=$${alpha} run; \
		make SHOTS=1 BALANCED=False EXP=influence_alpha_oracle SIMU_PARAMS="current_sequence alpha" POSTPOOL="oracle_centering l2_norm" ABLATION_ARG=alpha ABLATION_VAL=$${alpha} run; \
	done ;\
# 		make SHOTS=1 BALANCED=False EXP=influence_alpha_sota SIMU_PARAMS="current_sequence alpha" ABLATION_ARG=alpha ABLATION_VAL=$${alpha} snatcher; \

plot_alpha:
	python -m src.plots.csv_plotter --folder results --exp influence_alpha --param_plot alpha

nquery_influence:
	for n_query in 1 5 10 15 20; do \
		make SHOTS=1 EXP=influence_query_sota SIMU_PARAMS="current_sequence n_query" ABLATION_ARG=n_query ABLATION_VAL=$${n_query} snatcher; \
		make SHOTS=1 EXP=influence_query_debiased SIMU_PARAMS="current_sequence n_query" POSTPOOL="debiased_centering l2_norm" ABLATION_ARG=n_query ABLATION_VAL=$${n_query} run; \
		make SHOTS=1 EXP=influence_query_biased SIMU_PARAMS="current_sequence n_query" POSTPOOL="transductive_centering l2_norm" ABLATION_ARG=n_query ABLATION_VAL=$${n_query} run; \
		make SHOTS=1 EXP=influence_query_bn SIMU_PARAMS="current_sequence n_query" POSTPOOL="transductive_batch_norm l2_norm" ABLATION_ARG=n_query ABLATION_VAL=$${n_query} run; \
	done ;\

plot_nquery:
	python -m src.plots.csv_plotter --folder results --exp influence_query --param_plot n_query


# ================= Deployment / Imports ==================

deploy:
	rsync -avm Makefile $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' src $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' scripts $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' configs $(SERVER_IP):${SERVER_PATH}/ ;\

data/mini_imagenet.tar.gz:
	tar -czvf  data/mini_imagenet.tar.gz -C data/ mini_imagenet

data/tiered_imagenet.tar.gz:
	tar -czvf  data/tiered_imagenet.tar.gz -C data/ tiered_imagenet

data/cub.tar.gz:
	tar -czvf  data/cub.tar.gz -C data/ cub


deploy_data:
	for dataset in $(DATASETS); do \
		rsync -avm data/$${dataset}.tar.gz $(SERVER_IP):${SERVER_PATH}/data/ ;\
	done \

kill_all: ## Kill all my python and tee processes on the server
	ps -u $(USER) | grep "python" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill
	ps -u $(USER) | grep "tee" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill