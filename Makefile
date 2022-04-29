# Server options
SERVER_IP=narval
SERVER_PATH=~/scratch/open-set
USER=mboudiaf
DATADIR=data

# SERVER_IP=shannon
# SERVER_PATH=/ssd/repos/Few-Shot-Classification/Open-Set-Test
# DATADIR=../Open-Set/open-query-set/data/
# USER=malik


# SERVER_IP=shannon
# SERVER_PATH=/ssd/repos/Few-Shot-Classification/Open-Set/open-query-set
# DATADIR=data
# USER=malik



# Simu options
SRC_DATASET=mini_imagenet
TGT_DATASET=$(SRC_DATASET)


# Modules
CLS_TRANSFORMS=Pool  # Feature transformations used before feeding to the classifier
DET_TRANSFORMS=Pool  # Feature transformations used before feeding to the OOD detector
FEATURE_DETECTOR=none
PROBA_DETECTOR=none # may be removed, was just curious to see how detection on proba was working --> very bad
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
OVERRIDE=False # used to override existing entries in out.csv
TUNE=""
VISU=False
THRESHOLD=otsu

# Tasks
SPLIT=test
OOD_QUERY=15
N_TASKS=1000
SHOTS=1 5 # will iterate over these values
BALANCED=True
MISC_ARG=alpha
MISC_VAL=1.0

# === Base recipes ===

extract:
	    for split in train test; do \
			python -m src.compute_features \
				--backbone $(BACKBONE) \
				--src_dataset $(SRC_DATASET) \
				--tgt_dataset $(TGT_DATASET) \
				--data_dir $(DATADIR) \
		        --model_source $(MODEL_SRC) \
		        --training $(TRAINING) \
				--split $${split} \
				--layers $(LAYERS) ;\
	    done ;\


run:
	for shot in $(SHOTS); do \
	    python3 -m src.inference \
	        --exp_name $(EXP)/$(SRC_DATASET)'-->'$(TGT_DATASET)/$(BACKBONE)/$(MODEL_SRC)/$${shot} \
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
	        --split $(SPLIT) \
	        --threshold $(THRESHOLD) \
			--src_dataset $(SRC_DATASET) \
			--n_ood_query $(OOD_QUERY) \
			--tgt_dataset $(TGT_DATASET) \
	        --simu_hparams $(SIMU_PARAMS) \
	        --$(MISC_ARG) $(MISC_VAL) \
	        --override $(OVERRIDE) \
	        --tune $(TUNE) \
	        --debug $(DEBUG) ;\
    done ;\

# ========== Extraction pipelines ===========

extract_all:
# 	# Extract for RN and WRN
# 	for backbone in resnet12 wrn2810; do \
# 		for dataset in mini_imagenet tiered_imagenet; do \
# 			make BACKBONE=$${backbone} SRC_DATASET=$(TGT_DATASET) MODEL_SRC='feat' TGT_DATASET=$(TGT_DATASET) extract ;\
# 			make BACKBONE=$${backbone} TRAINING='feat' SRC_DATASET=$(TGT_DATASET) MODEL_SRC='feat' TGT_DATASET=$${dataset} extract ;\
# 		done ;\
# 	done ;\

# 	# Tiered-Imagenet -> *
# 	for backbone in resnet12 wrn2810; do \
# 		for dataset in fungi; do \
# 			make BACKBONE=$${backbone} TRAINING='feat' SRC_DATASET=tiered_imagenet MODEL_SRC='feat' TGT_DATASET=$(TGT_DATASET) extract ;\
# 			make BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet MODEL_SRC='feat' TGT_DATASET=$${dataset} extract ;\
# 		done ;\
# 	done ;\

	# Imagenet -> *
	for dataset in fungi imagenet; do \
# 		for backbone in clip_vit_base_patch16 vit_base_patch16_224 vit_base_patch16_224_dino vit_base_patch16_224_sam resnet50 dino_resnet50 ssl_resnet50 swsl_resnet50 mixer_b16_224_in21k mixer_b16_224_miil_in21k; do \
		for backbone in resnet50 ssl_resnet50 swsl_resnet50; do \
			make BACKBONE=$${backbone} SRC_DATASET=imagenet MODEL_SRC='url' TGT_DATASET=$${dataset} extract ;\
		done ;\
	done ;\


extract_bis:
	for backbone in resnet12 wrn2810; do \
			for split in train val test; do \
					python -m src.compute_features \
							--backbone $${backbone} \
							--src_dataset tiered_imagenet \
							--tgt_dataset tiered_imagenet_bis \
							--data_dir $(DATADIR) \
							--model_source feat \
							--training $(TRAINING) \
							--split $${split} \
							--layers $(LAYERS) \
							--keep_all_train_features True ;\
			done \
	done ;\


# ========== Feature Investigation ==========

clustering_metrics:
	for dataset in mini_imagenet tiered_imagenet; do \
			for split in train test; do \
					python -m src.investigate_features \
							data/features/$${dataset}/$${dataset}_bis/$${split}/standard/resnet12_$${dataset}_feat_4_4.pickle ;\
					python -m src.investigate_features \
							data/features/$${dataset}/$${dataset}_bis/$${split}/standard/wrn2810_$${dataset}_feat_last.pickle ;\
			done ;\
	done ;\

	for dataset in aircraft imagenet_val; do \
			for feature in ssl_resnext101_32x16d_imagenet_url_4_3 vit_base_patch16_224_in21k_imagenet_url_last_cls deit_tiny_patch16_224_imagenet_url_last_cls; do \
					python -m src.investigate_features \
							data/features/imagenet/$${dataset}/test/standard/$${feature}.pickle ;\
			done ;\
	done ;\


# ========== Running pipelines ===========

run_pyod:
	for method in HBOS KNN PCA OCSVM IForest COPOD; do \
		make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=$${method} run ;\
	done ;\

run_best:
	make run_ottim ;\
# 	make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=SimpleShot FEATURE_DETECTOR=KNN run ;\
# 	make CLS_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=TIM_GD PROBA_DETECTOR=MaxProbDetector run ;\
# 	make DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=OpenMax run ;\
# 	make run_snatcher ;\

run_finalists:
	make run_ottim ;\
	make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=SimpleShot FEATURE_DETECTOR=KNN run ;\

run_classifiers:
	for classifier in TIM_GD BDCSPN Finetune LaplacianShot SimpleShot; do \
		make PROBA_DETECTOR=MaxProbDetector CLS_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=$${classifier} run ;\
	done ;\
	make PROBA_DETECTOR=MaxProbDetector MODEL_SRC=feat TRAINING=feat CLASSIFIER=FEAT run ;\
	make CLS_TRANSFORMS="Pool Power QRreduction L2norm MeanCentering"  PROBA_DETECTOR=MaxProbDetector CLASSIFIER=MAP run ;\

run_snatcher:
	make MODEL_SRC=feat TRAINING=feat FEATURE_DETECTOR=SnatcherF run ;\

run_ottim:
	make FEATURE_DETECTOR=OTTIM run ;\

run_open_set:
	for method in RPL PROSER OpenMax; do \
		make DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=$${method} run ;\
	done \


# ========== 1) Tuning + Running pipelines ===========

tuning:
	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_ottim ;\
# 	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_open_set ;\
# 	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_pyod ;\
# 	make EXP=tuning TUNE=classifier SPLIT=val N_TASKS=500 run_classifiers ;\
# 	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_snatcher ;\

log_best_pyod:
	for shot in 1 5; do \
		for exp in HBOS KNN PCA OCSVM IForest COPOD; do \
			python -m src.plots.csv_plotter \
				 --exp $${exp} \
				 --groupby feature_detector \
				 --metrics mean_rocauc \
				 --use_pretty False \
				 --plot_versus backbone \
				 --action log_best \
				 --filters n_shot=$${shot} ;\
		done ;\
	done ;\

log_best_ottim:
	for shot in 1 5; do \
		python -m src.plots.csv_plotter \
			 --exp tuning \
			 --groupby feature_detector \
			 --metrics mean_acc mean_rocauc \
			 --use_pretty False \
			 --plot_versus backbone \
			 --action log_best \
			 --filters n_shot=$${shot} ;\
	done ;\

log_best_classif:
	for shot in 1 5; do \
		for exp in Finetune LaplacianShot BDCSPN TIM_GD MAP; do \
			python -m src.plots.csv_plotter \
				 --exp $${exp} \
				 --groupby classifier \
				 --metrics mean_acc \
				 --use_pretty False \
				 --plot_versus backbone \
				 --action log_best \
				 --filters n_shot=$${shot} ;\
		done ;\
	done ;\

benchmark:
	for dataset in mini_imagenet tiered_imagenet; do \
		make EXP=benchmark SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_ottim ;\
	done ;\
# 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_pyod ;\
# 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_classifiers ;\
# 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_snatcher ;\
# 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_open_set ;\

log_latex:
	for dataset in mini_imagenet tiered_imagenet; do \
		for shot in 1 5 ; do \
			python -m src.plots.csv_plotter \
				 --exp . \
				 --groupby classifier feature_detector \
				 --metrics mean_acc std_acc mean_rocauc std_rocauc mean_aupr std_aupr mean_prec_at_90 std_prec_at_90 \
				 --use_pretty True \
				 --plot_versus backbone \
				 --action log_latex \
				 --filters n_shot=$${shot} src_dataset=$${dataset} ;\
		done \
	done \


# ========== 2) Cross-domain experiments ===========

exhaustive_benchmark:
	# Tiered -> CUB
	for backbone in resnet12; do \
		make EXP=spider SHOTS=1 BACKBONE=$${backbone} run_best ;\
		for dataset in tiered_imagenet fungi aircraft cub; do \
			make EXP=spider SHOTS=1 BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet TGT_DATASET=$${dataset} run_best ;\
		done ; \
	done ;\

spider_chart:
	for backbone in resnet12; do \
		python -m src.plots.spider_plotter \
			 --exp spider \
			 --groupby classifier feature_detector \
			 --use_pretty True \
			 --metrics mean_acc mean_rocauc mean_aupr mean_prec_at_90 \
			 --plot_versus src_dataset tgt_dataset \
			 --filters n_shot=1 \
			 backbone=$${backbone} ;\
	done ;\


# ========== 3) Scaling up ==========


run_archs:
	# Imagenet -> *
	for backbone in vit_base_patch16_224 clip_vit_base_patch16 vit_base_patch16_224_dino vit_base_patch16_224_sam resnet50 dino_resnet50 ssl_resnet50 swsl_resnet50 mixer_b16_224_in21k mixer_b16_224_miil_in21k; do \
		for dataset in fungi; do \
			make EXP=barplots SHOTS=1 MODEL_SRC='url' BACKBONE=$${backbone} SRC_DATASET=imagenet TGT_DATASET=$${dataset} run_finalists ;\
		done ; \
	done ;\


barplots:
	python -m src.plots.bar_plotter \
		 --exp barplots \
		 --groupby classifier feature_detector \
		 --metrics mean_acc mean_rocauc mean_rec_at_90 mean_prec_at_90 \
		 --latex True \
		 --plot_versus backbone \
		 --filters n_shot=1 ;\

hbarplots:
	python -m src.plots.bar_plotter \
		 --exp barplots \
		 --groupby classifier feature_detector \
		 --metrics mean_acc mean_rocauc \
		 --latex True \
		 --plot_versus backbone \
		 --filters n_shot=1 ;\


# ========== 4) Ablation study ==========


ablate_ottim:
	# Imagenet -> *
	make TUNE=feature_detector run_ottim ;\

# ========== Plots ===========

plot_acc_vs_n_ood:
	for backbone in resnet12; do \
		for shot in 1 5; do \
			for tgt_dataset in mini_imagenet tiered_imagenet; do \
				python -m src.plots.csv_plotter --exp $(EXP) --groupby classifier \
					 --metrics mean_acc mean_rocauc \
					 --plot_versus n_ood_query --filters n_shot=$${shot} backbone=$${backbone} tgt_dataset=$(TGT_DATASET) ;\
			done ;\
		done ;\
	done ;\


plot_acc_vs_threshold:
	for backbone in resnet12; do \
		for shot in 1 5; do \
			for tgt_dataset in mini_imagenet; do \
				python -m src.plots.csv_plotter --exp thresholding --groupby classifier \
				     --metrics mean_acc mean_rocauc mean_believed_inliers mean_thresholding_accuracy \
					 --plot_versus threshold --filters n_shot=$${shot} backbone=$${backbone} tgt_dataset=$(TGT_DATASET) ;\
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

import/tiered:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/data/tiered_imagenet.tar.gz ./data/ ;\

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
		rsync -avm data/features/$${dataset} $(SERVER_IP):${SERVER_PATH}/data/features/ ;\
	done ;\


kill_all: ## Kill all my python and tee processes on the server
	ps -u $(USER) | grep "python" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill
	ps -u $(USER) | grep "tee" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill


# ============= Downlooad/Prepare data ============

aircraft:
	wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
	tar -xvf  fgvc-aircraft-2013b.tar.gz -C data ;\
	rm fgvc-aircraft-2013b.tar.gz ;\

fungi:
	mkdir -p data/fungi ;\
	wget https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz ;\
	wget https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz ;\
	tar -xvf fungi_train_val.tgz -C data/fungi ;\
	tar -xvf train_val_annotations.tgz -C data/fungi ;\
	rm fungi_train_val.tgz; rm train_val_annotations.tgz ;


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
	folder=`echo ${out_files} | cut -d'/' -f2-` ;\
	mkdir -p results/$${folder} ; \
	for file in $${out_files}; do \
		cp -Rv $${file} results/$${folder}/ ; \
	done
	rm tmp.txt
