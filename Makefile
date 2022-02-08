DATASETS=mini_imagenet
DETECTORS=knn
SHOTS=1 5
PREPOOL=base_centering
POSTPOOL=l2_norm
LAYERS=4_4
COMBIN=1
EXP=default
RESOLUTION=84
AUGMENTATIONS=trivial
BACKBONE=resnet12
TRAINING="feat"

lint:
		pylint easyfsl

test:
		pytest easyfsl

dev-install:
	pip install -r dev_requirements.txt


extract:
	for layer in $(LAYERS); do \
	    for dataset in $(DATASETS); do \
	        for split in train test; do \
				python -m src.compute_features \
					--backbone $(BACKBONE) \
					--dataset $${dataset} \
					--training $(TRAINING) \
					--split $${split} \
					--layer $${layer} ;\
	        done \
	    done \
	done \
wdawd

run:
	for dataset in $(DATASETS); do \
		for detector in $(DETECTORS); do \
			for shot in $(SHOTS); do \
			    python3 -m src.main_features \
			        --exp_name $(EXP)'-'$${shot}'-'$${dataset} \
			        --mode 'tune' \
			        --inference_method SimpleShot \
			        --n_tasks 500 \
			        --n_shot $${shot} \
			        --layers $(LAYERS) \
			        --outlier_detectors $${detector} \
			        --prepool_transform  $(PREPOOL) \
			        --postpool_transform  $(POSTPOOL) \
			        --pool \
			        --aggreg l2_bar \
			        --backbone $(BACKBONE) \
			        --training feat \
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
			    python3 -m src.main \
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
			        --augmentations $(AUGMENTATIONS) \
			        --pool \
			        --aggreg l2_bar \
			        --backbone $(BACKBONE) \
			        --training feat \
			        --dataset $${dataset} \
			        --simu_hparams 'current_sequence' \
			        --combination_size $(COMBIN) \
			        --override ;\
		    done ;\
		done ;\
	done ;\

baseline:
	make EXP=baseline run_scratch ;\


coupling:
	make COMBIN=3 SHOTS="1 5" EXP=coupling run ;\

layer_mixing:
	make EXP=layer_mixing PREPOOL=layer_norm LAYERS='4_1' run_scratch ;\


resolution:
	make PREPOOL=trivial EXP=resolution SHOTS=1 LAYERS='4_0' RESOLUTION=224 run_scratch ;\

cutmix:
	make EXP=cutmix SHOTS=5 PREPOOL='inductive_batch_norm' AUGMENTATIONS='mixup' run_scratch ;\
# 	make EXP=cutmix SHOTS=5 PREPOOL='base_bn' AUGMENTATIONS='cutmix' run_scratch ;\

