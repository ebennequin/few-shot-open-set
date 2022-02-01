DATASETS=mini_imagenet
DETECTORS=knn
SHOTS=1 5
PREPOOL=base_centering
POSTPOOL=l2_norm
LAYERS=4_2
COMBIN=1
EXP=default

lint:
		pylint easyfsl

test:
		pytest easyfsl

dev-install:
		pip install -r dev_requirements.txt


extract:
		training="feat.pth" ;\
		for layer in 4_2; do \
		    for dataset in mini_imagenet; do \
		        for split in train test; do \
		            for arch in resnet12; do \
		                python -m stages.compute_features $${arch} $${dataset} data/models/$${arch}_$${dataset}_$${training} --split $${split} --layer $${layer} ;\
		            done \
		        done \
		    done \
		done \


run:
		for dataset in $(DATASETS); do \
			for detector in $(DETECTORS); do \
				for shot in $(SHOTS); do \
				    python3 -m notebooks.experiments_open_query_detection_on_features \
				        --exp_name '$(EXP) $${shot}-$${dataset}' \
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
				        --backbone resnet12 \
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
		make EXP=baseline PREPOOL='base_bn' run ;\