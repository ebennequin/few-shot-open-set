DATASETS=mini_imagenet
DETECTORS=knn
SHOTS=1 5
N_TASKS=1000
PREPOOL=base_centering
POSTPOOL=l2_norm
LAYERS=4_4
COMBIN=1
EXP=default
RESOLUTION=84
AUGMENTATIONS=trivial
BACKBONE=resnet12
TRAINING="feat"


extract:
		for dataset in $(DATASETS); do \
		    for split in train test; do \
				python -m src.compute_features \
					--backbone $(BACKBONE) \
					--dataset $${dataset} \
					--training $(TRAINING) \
					--split $${split} \
					--layers $(LAYERS) ;\
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
	make EXP=baseline run ;\


coupling:
	make COMBIN=3 SHOTS="1 5" EXP=coupling run ;\

layer_mixing:
	make EXP=layer_mixing PREPOOL=base_centering run ;\


resolution:
	make PREPOOL=trivial EXP=resolution SHOTS=1 LAYERS='4_0' RESOLUTION=224 run_scratch ;\

cutmix:
	make EXP=cutmix SHOTS=5 PREPOOL='inductive_batch_norm' AUGMENTATIONS='mixup' run_scratch ;\
# 	make EXP=cutmix SHOTS=5 PREPOOL='base_bn' AUGMENTATIONS='cutmix' run_scratch ;\

