# Server options
SERVER_IP=shannon
SERVER_PATH=/ssd/repos/Few-Shot-Classification/Open-Set-Test
USER=malik

# Simu options
DATADIR=/ssd/repos/Few-Shot-Classification/Open-Set/open-query-set/data
SRC_DATASET=mini_imagenet
TGT_DATASETS=$(SRC_DATASET)


# Modules
CLS_TRANSFORMS=Pool L2norm
DET_TRANSFORMS=Pool
FEATURE_DETECTOR=kNNDetector
PROBA_DETECTOR=EntropyDetector
CLASSIFIER=SimpleShot
FILTERING=False


# Model
LAYERS=1
BACKBONE=resnet12
MODEL_SRC=feat
TRAINING=standard

# Misc
EXP=default
DEBUG=False
GPUS=0
SIMU_PARAMS=
OVERRIDE=True
TUNE=""

# Tasks
RESOLUTION=84
OOD_QUERY=10
N_TASKS=1000
SHOTS=1 5
BALANCED=True
MISC_ARG=alpha
MISC_VAL=1.0

train:
	SHOTS=1 ;\
	for dataset in $(DATASETS); do \
		for shot in $(SHOTS); do \
		    python3 -m src.pretrain \
		        --exp_name $${dataset}'-'$(EXP) \
		        --data_dir $(DATADIR) \
		        --classifier SimpleShot \
		        --n_tasks 500 \
		        --n_shot $${shot} \
		        --feature_transforms  $(TRANSFORMS) \
		        --backbone $(BACKBONE) \
		        --dataset $${dataset} \
		        --debug $(DEBUG) \
		        --gpus $(GPUS) ;\
	    done ;\
	done ;\

extract:
		for dataset in $(TGT_DATASETS); do \
		    for split in train test; do \
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
		for shot in $(SHOTS); do \
		    python3 -m src.inference_features \
		        --exp_name $(EXP)/$(SRC_DATASET)'-->'$${dataset}/$(BACKBONE)'($(MODEL_SRC))'/$${shot} \
		        --data_dir $(DATADIR) \
		        --classifier $(CLASSIFIER) \
		        --n_tasks $(N_TASKS) \
		        --n_shot $${shot} \
		        --layers $(LAYERS) \
		        --feature_detector $(FEATURE_DETECTOR) \
		        --use_filtering $(FILTERING) \
		        --proba_detector $(PROBA_DETECTOR) \
		        --detector_transforms  $(DET_TRANSFORMS) \
		        --classifier_transforms  $(CLS_TRANSFORMS) \
		        --backbone $(BACKBONE) \
		        --model_source $(MODEL_SRC) \
		        --balanced $(BALANCED) \
		        --training $(TRAINING) \
				--src_dataset $(SRC_DATASET) \
				--tgt_dataset $${dataset} \
		        --simu_hparams $(SIMU_PARAMS) \
		        --$(MISC_ARG) $(MISC_VAL) \
		        --override $(OVERRIDE) \
		        --tune $(TUNE) \
		        --debug $(DEBUG) ;\
	    done ;\
	done ;\

# ========== Extraction pipelines ===========

extract_standard:
	# Extract for RN and WRN
	for tgt_dataset in mini_imagenet tiered_imagenet; do \
		for backbone in resnet12 wrn2810; do \
			make BACKBONE=$${backbone} LAYERS='all' MODEL_SRC='feat' TGT_DATASETS=$${tgt_dataset} extract ;\
		done ;\
	done ;\

	# Extract for cross-domain
	for tgt_dataset in cub aircraft; do \
		for backbone in deit_tiny_patch16_224 ssl_resnext101_32x16d vit_base_patch16_224_in21k; do \
			make BACKBONE=$${backbone} LAYERS='all' SRC_DATASET=imagenet MODEL_SRC='url' TGT_DATASETS=$${tgt_dataset} extract ;\
		done ;\
	done ;\


extract_snatcher:
	make TRAINING='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=tiered_imagenet extract ;\
	make TRAINING='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=cub extract ;\
	make TRAINING='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=cub extract ;\
# 	make TRAINING='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet extract ;\


run_feature_detectors:
	for feature_detector in FinetuneDetector; do \
		make FEATURE_DETECTOR=$${feature_detector} run ;\
	done ;\

run_ssl_detectors:
	for feature_detector in MTC; do \
		make FEATURE_DETECTOR=$${feature_detector} run ;\
	done ;\

# ========== Experiments ===========

run_classifiers:
	for dataset in mini_imagenet; do \
		for backbone in resnet12; do \
			for classifier in TIM_GD ; do \
				make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} BACKBONE=$${backbone} CLASSIFIER=$${classifier} run ;\
			done ;\
		done ;\
		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} \
			CLS_TRANSFORMS="Pool Power QRreduction L2norm MeanCentering"  BACKBONE=$${backbone} CLASSIFIER=MAP run ;\
	done ;\
# 			make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} BACKBONE=$${backbone} \
# 					CLASSIFIER=SemiFEAT TRAINING=feat run_proba_detectors ;\

ideal_case:
	make EXP=ideal_case run_classifiers

no_filtering:
	for ood_query in 1 5 10 15 20 25 30 40 50 75 100; do \
		make EXP=filtering SIMU_PARAMS=n_ood_query FEATURE_DETECTOR=none MISC_ARG=n_ood_query MISC_VAL=$${ood_query} run_classifiers ;\
	done ;\

filtering_knn:
	for ood_query in 1 5 10 15 20 25 30 40 50 75 100; do \
		make EXP=filtering_test SIMU_PARAMS=n_ood_query MISC_ARG=n_ood_query MISC_VAL=$${ood_query} \
			DET_TRANSFORMS="Pool BaseCentering L2norm" FILTERING=True FEATURE_DETECTOR=kNNDetector run_classifiers ;\
	done ;\

filtering_repri:
	for ood_query in 1 5 10 15 20 25 30 40 50 75 100; do \
		for detector in RepriDetector; do \
			make EXP=filtering_test SIMU_PARAMS=n_ood_query MISC_ARG=n_ood_query MISC_VAL=$${ood_query} \
				FILTERING=True FEATURE_DETECTOR=$${detector} run_classifiers ;\
		done ;\
	done ;\



cross_domain:
	# Tiered -> CUB, Aircraft
	for backbone in resnet12 wrn2810; do \
		for tgt_dataset in cub; do \
			make EXP=cross_domain BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet TGT_DATASETS=$${tgt_dataset} run_feature_detectors ;\
		done ; \
	done ;\

	# ImageNet -> CUB Aircraft with all kinds of models
	for tgt_dataset in aircraft; do \
		for backbone in deit_tiny_patch16_224 efficientnet_b4 ssl_resnext101_32x16d vit_base_patch16_224_in21k; do \
			make EXP=cross_domain BACKBONE=$${backbone} MODEL_SRC='url' \
				SRC_DATASET=imagenet TGT_DATASETS=$${tgt_dataset} run_feature_detectors ;\
		done ;\
	done ;\

layers:
	for layers in 2 3 4; do \
		make EXP=layers LAYERS=$${layers} SIMU_PARAMS="layers" benchmark ;\
	done ;\

imbalance:
	for alpha in 0.5 1.0 2.0 3.0 4.0 5.0; do \
		for backbone in resnet12; do \
			make BALANCED=False EXP=imbalance SIMU_PARAMS="alpha" MISC_ARG=alpha MISC_VAL=$${alpha} BACKBONE=$${backbone} run_feature_detectors; \
		done ;\
	done ;\


# ========== Ablations ===========

plot_filtering:
	for backbone in resnet12; do \
		for shot in 1 5; do \
			for tgt_dataset in mini_imagenet tiered_imagenet; do \
				python -m src.plots.csv_plotter --exp filtering --groupby feature_detector --plot_versus n_ood_query \
					--filters n_shot=$${shot} backbone=$${backbone} tgt_dataset=$${tgt_dataset} ;\
			done ;\
		done ;\
	done ;\

# plot_cross_domain:
# 	for backbone in wrn2810 resnet12 vitb16; do \
# 		for shot in 1 5; do \
# 			python -m src.plots.csv_plotter --exp cross_domain --groupby transformss --plot_versus tgt_dataset \
# 				--filters n_shot=$${shot} feature_detector=knn backbone=$${backbone} ;\
# 		done ;\
# 	done ;\

# plot_imbalance:
# 	for backbone in wrn2810 resnet12; do \
# 		for  shot in 1 5; do \
# 			python -m src.plots.csv_plotter --exp imbalance --groupby transformss --plot_versus alpha \
# 				--filters n_shot=$${shot} feature_detector=knn backbone=$${backbone} ;\
# 		done ;\
# 	done ;\

# ================= Deployment / Imports ==================

deploy:
	rsync -avm Makefile $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' src $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' scripts $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' configs $(SERVER_IP):${SERVER_PATH}/ ;\

import:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/results ./ ;\


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


# ============= Archive results =============


store: # Archive experiments
	python src/utils/list_files.py results/ archive/ tmp.txt
	{ read -r out_files; read -r archive_dir; } < tmp.txt ; \
	for file in $${out_files}; do \
		cp -Rv $${file} $${archive_dir}/ ; \
	done
	rm tmp.txt


restore: # Restore experiments to output/
	python src/utils/list_files.py archive/ results/ tmp.txt ; \
	read -r out_files < tmp.txt ; \
	mkdir -p results/$${folder[1]} ; \
	for file in $${out_files}; do \
		cp -Rv $${file} results/$${folder[1]}/ ; \
	done
	rm tmp.txt

# =============== Models ====================

models/vit_b16:
	mkdir -p data/models/standard/
	wget https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth -O data/models/standard/vitb16_imagenet_luke.pth

models/vit_l16:
	mkdir -p data/models/standard/
	wget https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16.pth -O data/models/standard/vitl16_imagenet_luke.pth