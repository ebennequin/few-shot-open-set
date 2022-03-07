
# Server options
SERVER_IP=narval
SERVER_PATH=~/scratch/open-set
USER=mboudiaf

# Simu options
DATADIR=data
SRC_DATASET=mini_imagenet
TGT_DATASETS=$(SRC_DATASET)
DETECTORS=knn
TRANSFORMS="Pool L2norm"
LAYERS=1
COMBIN=1
EXP=default
RESOLUTION=84
BACKBONE=resnet12
MODEL_SRC=feat
TRAINING=standard
DEBUG=False
GPUS=0
SIMU_PARAMS=current_sequence
OVERRIDE=True
MODE=tune

# Tasks
N_TASKS=1000
SHOTS=1 5
BALANCED=True
MISC_ARG=alpha
MISC_VAL=1.0

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
		        --feature_transforms  $(TRANSFORMS) \
		        --pool \
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
		for detector in $(DETECTORS); do \
			for shot in $(SHOTS); do \
			    python3 -m src.inference_features \
			        --exp_name $(EXP)/$(SRC_DATASET)'-->'$${dataset}/$(BACKBONE)'($(MODEL_SRC))'/$${shot} \
			        --mode 'tune' \
			        --inference_method SimpleShot \
			        --n_tasks $(N_TASKS) \
			        --n_shot $${shot} \
			        --layers $(LAYERS) \
			        --outlier_detectors $${detector} \
			        --feature_transforms  $(TRANSFORMS) \
			        --backbone $(BACKBONE) \
			        --model_source $(MODEL_SRC) \
			        --balanced $(BALANCED) \
			        --training $(TRAINING) \
					--src_dataset $(SRC_DATASET) \
					--tgt_dataset $${dataset} \
			        --simu_hparams $(SIMU_PARAMS) \
			        --combination_size $(COMBIN) \
			        --$(MISC_ARG) $(MISC_VAL) \
			        --override $(OVERRIDE) \
			        --mode $(MODE) \
			        --debug $(DEBUG) ;\
		    done ;\
		done ;\
	done ;\

run_scratch:
	for dataset in $(DATASETS); do \
		for detector in $(DETECTORS); do \
			for shot in $(SHOTS); do \
			    python3 -m src.inference \
			        --exp_name $(EXP)'-'$${shot}'-'$${dataset}'-'$(BACKBONE) \
			        --mode 'tune' \
			        --inference_method SimpleShot \
			        --n_tasks 500 \
			        --n_shot $${shot} \
			        --layers $(LAYERS) \
			        --image_size $(RESOLUTION) \
			        --outlier_detectors $${detector} \
			        --transforms  $(TRANSFORMS) \
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
	# Extract for RN and WRN
# 	for tgt_dataset in mini_imagenet tiered_imagenet; do \
# 		for backbone in resnet12; do \
# 			make BACKBONE=$${backbone} LAYERS='all' MODEL_SRC='feat' TGT_DATASETS=$${tgt_dataset} extract ;\
# 		done ;\
# 	done ;\

	# Extract for cross-domain
	for tgt_dataset in cub aircraft; do \
		for backbone in resnet12 efficientnet_b4 deit_tiny_patch16_224 ssl_resnext101_32x16d vit_base_patch16_224_in21k; do \
			make BACKBONE=$${backbone} LAYERS='all' SRC_DATASET=imagenet MODEL_SRC='url' TGT_DATASETS=$${tgt_dataset} extract ;\
		done ;\
	done ;\



extract_snatcher:
	make TRAINING='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=tiered_imagenet extract ;\
	make TRAINING='feat' SRC_DATASET=tiered_imagenet TGT_DATASETS=cub extract ;\
	make TRAINING='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=cub extract ;\
# 	make TRAINING='feat' SRC_DATASET=mini_imagenet TGT_DATASETS=mini_imagenet extract ;\

run_snatcher:
	make TRANSFORMS=trivial DETECTORS='snatcher_f' TRAINING='feat' run ;\

run_centering:
	for centering in Alternate; do \
		make TRANSFORMS="Pool $${centering}Centering L2norm" run ;\
	done ;\
# 	make TRANSFORMS="l2_norm" run ;\
# 	for centering in base debiased tarjan transductive kcenter; do \

# ========== Experiments ===========

benchmark:
	for dataset in mini_imagenet tiered_imagenet; do \
		for backbone in resnet12; do \
			make EXP=benchmark SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} BACKBONE=$${backbone} run_centering ;\
# 			make EXP=benchmark SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} BACKBONE=$${backbone} run_snatcher ;\
		done ;\
	done ;\

cross_domain:
	# Tiered -> CUB, Aircraft
	for backbone in resnet12 wrn2810; do \
		for tgt_dataset in cub aircraft; do \
			make EXP=cross_domain BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet TGT_DATASETS=$${tgt_dataset} run_centering ;\
		done ; \
	done ;\
	# ImageNet -> CUB Aircraft with ViT
	for tgt_dataset in cub aircraft; do \
		make EXP=cross_domain BACKBONE=vitb16 MODEL_SRC='luke' \
			SRC_DATASET=imagenet TGT_DATASETS=$${tgt_dataset} run_centering ;\
	done ;\

imbalance:
	for alpha in 0.5 1.0 2.0 3.0 4.0 5.0; do \
		for backbone in resnet12 wrn2810; do \
			make BALANCED=False EXP=imbalance SIMU_PARAMS="current_sequence alpha" MISC_ARG=alpha MISC_VAL=$${alpha} BACKBONE=$${backbone} run_centering; \
		done ;\
	done ;\

nquery_influence:
	for n_query in 1 5 10 15 20; do \
		for backbone in resnet12; do \
			make EXP=n_query SIMU_PARAMS="current_sequence n_query" MISC_ARG=n_query MISC_VAL=$${n_query} BACKBONE=$${backbone} run_centering; \
		done ;\
	done ;\

# ========== Ablations ===========

plot_benchmark:
	for backbone in wrn2810 resnet12; do \
		for tgt_dataset in mini_imagenet tiered_imagenet; do \
			python -m src.plots.csv_plotter --exp benchmark --groupby transformss --plot_versus n_shot \
				--filters outlier_detectors=knn backbone=$${backbone} tgt_dataset=$${tgt_dataset} ;\
		done ;\
	done ;\

plot_cross_domain:
	for backbone in wrn2810 resnet12 vitb16; do \
		for shot in 1 5; do \
			python -m src.plots.csv_plotter --exp cross_domain --groupby transformss --plot_versus tgt_dataset \
				--filters n_shot=$${shot} outlier_detectors=knn backbone=$${backbone} ;\
		done ;\
	done ;\

plot_imbalance:
	for backbone in wrn2810 resnet12; do \
		for  shot in 1 5; do \
			python -m src.plots.csv_plotter --exp imbalance --groupby transformss --plot_versus alpha \
				--filters n_shot=$${shot} outlier_detectors=knn backbone=$${backbone} ;\
		done ;\
	done ;\

plot_nquery:
	for backbone in wrn2810 resnet12; do \
		for  shot in 1 5; do \
			python -m src.plots.csv_plotter --exp n_query --groupby transformss --plot_versus n_query \
				--filters n_shot=$${shot} outlier_detectors=knn backbone=$${backbone} ;\
		done ;\
	done ;\


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