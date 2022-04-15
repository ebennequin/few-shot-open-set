# Server options
# SERVER_IP=narval
# SERVER_PATH=~/scratch/open-set
# USER=mboudiaf
# DATADIR=data

# SERVER_IP=shannon
# SERVER_PATH=/ssd/repos/Few-Shot-Classification/Open-Set-Test
# DATADIR=../Open-Set/open-query-set/data/
# USER=malik


SERVER_IP=shannon
SERVER_PATH=/ssd/repos/Few-Shot-Classification/Open-Set/open-query-set
DATADIR=data
USER=malik



# Simu options
SRC_DATASET=mini_imagenet
TGT_DATASETS=$(SRC_DATASET)


# Modules
CLS_TRANSFORMS=Pool L2norm  # Feature transformations used before feeding to the classifier
DET_TRANSFORMS=Pool  # Feature transformations used before feeding to the OOD detector
FEATURE_DETECTOR=kNNDetector
PROBA_DETECTOR=EntropyDetector # may be removed, was just curious to see how detection on proba was working --> very bad
CLASSIFIER=SimpleShot
FILTERING=False # whether to use $(FEATURE_DETECTOR) in order to filter out outliers before feeding to classifier


# Model
LAYERS=1 # Numbers of layers (starting from the end) to use. If 2 layers, OOD detection will be made on 2 last layers, and then aggregated
BACKBONE=resnet12
MODEL_SRC=feat# Origin of the model. For all timm models, use MODEL_SRC=url
TRAINING=standard# To differentiate between episodic and standard models

# Misc
EXP=default# name of the folder in which results will be stored.
DEBUG=False
GPUS=0
SIMU_PARAMS=  # just in case you need to track some particular args in out.csv
OVERRIDE=True # used to override existing entries in out.csv
TUNE=""
VISU=False
THRESHOLD=otsu

# Tasks
OOD_QUERY=10
N_TASKS=1000
SHOTS=1 5 # will iterate over these values
BALANCED=True
MISC_ARG=alpha
MISC_VAL=1.0

# === Base recipes ===

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
		    python3 -m src.inference \
		        --exp_name $(EXP)/$(SRC_DATASET)'-->'$${dataset}/$(BACKBONE)/$(MODEL_SRC)/$${shot} \
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
		        --visu_episode $(VISU) \
		        --backbone $(BACKBONE) \
		        --model_source $(MODEL_SRC) \
		        --balanced $(BALANCED) \
		        --training $(TRAINING) \
		        --threshold $(THRESHOLD) \
				--src_dataset $(SRC_DATASET) \
				--n_ood_query $(OOD_QUERY) \
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
#	for tgt_dataset in mini_imagenet tiered_imagenet; do \
#		for backbone in resnet12 wrn2810; do \
#			make BACKBONE=$${backbone} SRC_DATASET=$${tgt_dataset} MODEL_SRC='feat' TGT_DATASETS=$${tgt_dataset} extract ;\
#		done ;\
#	done ;\

	# Extract for cross-domain
	for tgt_dataset in cub aircraft; do \
		for backbone in deit_tiny_patch16_224 ssl_resnext101_32x16d vit_base_patch16_224_in21k; do \
			make BACKBONE=$${backbone} SRC_DATASET=imagenet MODEL_SRC='url' TGT_DATASETS=$${tgt_dataset} extract ;\
		done ;\
	done ;\


extract_snatcher:
	make TRAINING='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=tiered_imagenet extract ;\
	make TRAINING='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=cub extract ;\
	make TRAINING='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=cub extract ;\
# 	make TRAINING='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet extract ;\


extract_bis:
	for backbone in resnet12 wrn2810; do \
		make BACKBONE=$${backbone} SRC_DATASET=mini_imagenet MODEL_SRC='feat' TGT_DATASETS=mini_imagenet_bis extract ;\
	done ;\

# ========== Evaluating OOD detectors in isolation ===========

run_transductive_detectors:
	for feature_detector in FinetuneDetector; do \
		make FEATURE_DETECTOR=$${feature_detector} run ;\
	done ;\

run_pyod_detectors:
	for feature_detector in kNNDetector; do \
		make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=$${feature_detector} run ;\
	done ;\

# ========== Evaluating transductive methods ===========

run_transductive_methods:
	for dataset in mini_imagenet; do \
		for backbone in resnet12; do \
			make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} \
				CLS_TRANSFORMS="Pool Power QRreduction L2norm MeanCentering"  BACKBONE=$${backbone} CLASSIFIER=MAP run ;\
			for classifier in TIM_GD; do \
				make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} BACKBONE=$${backbone} CLASSIFIER=$${classifier} run ;\
			done ;\
		done ;\
	done ;\

run_w_knn_filtering:
	for ood_query in 1 3 5 7 10 12 15 17 20 22 25 27 30 35 40 45 50 60 75 90 100; do \
		make SIMU_PARAMS=n_ood_query OOD_QUERY=$${ood_query} \
			DET_TRANSFORMS="Pool BaseCentering L2norm" FILTERING=True run ;\
	done ;\

run_wo_filtering:
	for ood_query in 1 3 5 7 10 12 15 17 20 22 25 27 30 35 40 45 50 60 75 90 100; do \
		make SIMU_PARAMS=n_ood_query OOD_QUERY=$${ood_query} run ;\
	done ;\

run_fixed_thresholding:
	for thresh in 0.1 0.12 0.15 0.17 0.2 0.22 0.25 0.27 0.3 0.32 0.35 0.37 0.4; do \
		make DET_TRANSFORMS="Pool BaseCentering L2norm" \
			 SIMU_PARAMS=threshold FILTERING=True THRESHOLD=$${thresh} run ;\
	done ;\

run_svm_thresholding:
	make EXP=svm_thresholding THRESHOLD=svm run_w_knn_filtering ;\

tune_ood_tim:
	make DET_TRANSFORMS="Pool" N_TASKS=500 EXP=tune_ood_tim FEATURE_DETECTOR=OOD_TIM TUNE=feature_detector run ;\

run_ood_tim:
	make DET_TRANSFORMS="Pool" EXP=ood_tim FEATURE_DETECTOR=OOD_TIM run_wo_filtering ;\

# ========== Feature Investigation ==========

clustering_metrics:
	for dataset in mini_imagenet; do \
		for split in train test; do \
			python -m src.investigate_features \
				data/features/$${dataset}/$${dataset}_bis/$${split}/standard/resnet12_$${dataset}_feat_4_4.pickle ;\
			python -m src.investigate_features \
				data/features/$${dataset}/$${dataset}_bis/$${split}/standard/wrn2810_$${dataset}_feat_last.pickle ;\
		done ;\
	done ;\

# ========== Evaluating SSL methods ===========

run_ssl_detectors:
	for feature_detector in FixMatch; do \
		make FEATURE_DETECTOR=$${feature_detector} run ;\
	done ;\


# ========== Cross-domain experiments ===========

cross_domain:
	# Tiered -> CUB
	for backbone in resnet12 wrn2810; do \
		for tgt_dataset in cub; do \
			make EXP=cross_domain BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet TGT_DATASETS=$${tgt_dataset} run_pyod_detectors ;\
		done ; \
	done ;\

	# ImageNet -> Aircraft with all kinds of models
	for tgt_dataset in aircraft; do \
		for backbone in deit_tiny_patch16_224 efficientnet_b4 ssl_resnext101_32x16d vit_base_patch16_224_in21k; do \
			make EXP=cross_domain BACKBONE=$${backbone} MODEL_SRC='url' \
				SRC_DATASET=imagenet TGT_DATASETS=$${tgt_dataset} run_pyod_detectors ;\
		done ;\
	done ;\

# ========== Plots ===========

plot_acc_vs_n_ood:
	for backbone in resnet12; do \
		for shot in 1 5; do \
			for tgt_dataset in mini_imagenet; do \
				python -m src.plots.csv_plotter --exp $(EXP) --groupby classifier \
					 --metrics mean_acc mean_features_rocauc \
					 --plot_versus n_ood_query --filters n_shot=$${shot} backbone=$${backbone} tgt_dataset=$${tgt_dataset} ;\
			done ;\
		done ;\
	done ;\


plot_acc_vs_threshold:
	for backbone in resnet12; do \
		for shot in 1 5; do \
			for tgt_dataset in mini_imagenet; do \
				python -m src.plots.csv_plotter --exp thresholding --groupby classifier \
				     --metrics mean_acc mean_features_rocauc mean_believed_inliers mean_thresholding_accuracy \
					 --plot_versus threshold --filters n_shot=$${shot} backbone=$${backbone} tgt_dataset=$${tgt_dataset} ;\
			done ;\
		done ;\
	done ;\

# ================= Deployment / Imports ==================

deploy:
	rsync -avm Makefile $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' src $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' scripts $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' configs $(SERVER_IP):${SERVER_PATH}/ ;\

import/results:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/results ./ ;\

import/archive:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/archive ./ ;\

tar_data:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		tar -czvf  data/$${dataset}.tar.gz -C data/ $${dataset} ;\
	done ;\


deploy_data:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm data/$${dataset}.tar.gz $(SERVER_IP):${SERVER_PATH}/data/ ;\
	done ;\

deploy_models:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm data/models $(SERVER_IP):${SERVER_PATH}/ ;\
	done ;\

deploy_features:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm data/features $(SERVER_IP):${SERVER_PATH}/ ;\
	done ;\


kill_all: ## Kill all my python and tee processes on the server
	ps -u $(USER) | grep "python" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill
	ps -u $(USER) | grep "tee" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill


# ============= Downlooad/Prepare data ============

aircraft:
	wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
	tar -xvf  fgvc-aircraft-2013b.tar.gz -C data ;\
	rm fgvc-aircraft-2013b.tar.gz ;\

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
