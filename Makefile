
# Server options
SERVER_IP=narval
SERVER_PATH=~/scratch/open-set
USER=mboudiaf

# Simu options
DATADIR=data
DATASETS=mini_imagenet
DETECTORS=knn
PREPOOL=base_centering
POSTPOOL=l2_norm
LAYERS=4_4
COMBIN=1
EXP=default
RESOLUTION=84
BACKBONE=resnet12
MODEL_SRC=feat
TRAINING=standard
DEBUG=False
GPUS=0

# Tasks
N_TASKS=10000
SHOTS=1 5
BALANCED=True


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
		for dataset in $(DATASETS); do \
		    for split in train test; do \
				python -m src.compute_features \
					--backbone $(BACKBONE) \
					--dataset $${dataset} \
					--data_dir $(DATADIR) \
			        --model_source $(MODEL_SRC) \
			        --training $(TRAINING) \
					--split $${split} \
					--layers 4_0 4_1 4_2 4_3 4_4 ;\
		    done \
		done \

run:
	for dataset in $(DATASETS); do \
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
			        --dataset $${dataset} \
			        --simu_hparams 'current_sequence' \
			        --combination_size $(COMBIN) \
			        --override ;\
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
			        --override ;\
		    done ;\
		done ;\
	done ;\

baseline:
	make EXP=baseline run ;\

layer_mixing:
	make EXP=layer_mixing COMBIN=3 run ;\

multi_layers:
	make LAYERS="4_0 4_1 4_2 4_3" run ;\

extract_snatcher:
	make TRAINING='feat' MODEL_SRC='feat' extract ;\

snatcher:
	make PREPOOL=trivial POSTPOOL=trivial DETECTORS='snatcher_f' TRAINING='feat' MODEL_SRC='feat' run ;\

experimental_training:
	make PREPOOL=trivial SHOTS=1 train ;\

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