method=SimpleShot
dataset="tiered_imagenet"
backbone=resnet12
training=feat

# ====== Search best way to normalize ========

# for backbone in resnet12; do
# 	for prepool in trivial inductive_batch_norm transductive_batch_norm layer_norm; do
# 		for postpool in trivial l2_norm; do
# 			python3 -m notebooks.experiments_open_query_detection_on_features \
# 			  --exp_name "normalizations_${dataset}_${backbone}" \
# 			  --inference_method ${method} \
# 			  --outlier_detectors knn_10 \
# 			  --prepool_transform base_centering ${prepool} \
# 			  --postpool_transform ${postpool} \
# 			  --training ${training} \
# 			  --backbone ${backbone} \
# 			  --dataset ${dataset}
# 		done
# 	done
# done

# ====== Trying median detectors =========

# for combin in knn_med_1 knn_med_3 knn_med_5 knn_med_10 knn_med_1 knn_med_1-knn_med_3-knn_med_5-knn_med_10-knn_med_15; do
# 	python3 -m notebooks.experiments_open_query_detection_on_features \
# 			  --exp_name "median_detector_${dataset}_${backbone}" \
# 			  --inference_method ${method} \
# 			  --outlier_detectors ${combin} \
# 			  --prepool_transform  base_centering \
# 			  --postpool_transform  l2_norm \
# 			  --backbone ${backbone} \
# 			  --simu_hparams 'training' \
# 			  --training ${training} \
# 			  --dataset ${dataset} \
# 			  --override
# done


# ====== Trying multidetectors =========

# for combin in knn_med_1-knn_med_3-knn_med_5-knn_med_10; do
# 	python3 -m notebooks.experiments_open_query_detection_on_features \
# 			  --exp_name "multi_detectors_${dataset}_${backbone}" \
# 			  --inference_method ${method} \
# 			  --outlier_detectors ${combin} \
# 			  --prepool_transform  base_centering \
# 			  --postpool_transform  l2_norm \
# 			  --backbone ${backbone} \
# 			  --simu_hparams 'training' \
# 			  --training ${training} \
# 			  --dataset ${dataset} \
# 			  --override
# done

# ========= Benchmarking properly ==========

for dataset in mini_imagenet tiered_imagenet; do
	for backbone in resnet12; do
		python3 -m notebooks.experiments_open_query_detection_on_features \
			--exp_name 'benchmark' \
			--inference_method ${method} \
			--n_shot 1 \
			--outlier_detectors knn_med_1-knn_med_3 \
			--prepool_transform  base_centering \
			--postpool_transform  l2_norm \
		    --backbone ${backbone} \
		    --simu_hparams 'training' \
		    --training ${training} \
		    --dataset ${dataset} \
		    --override

		python3 -m notebooks.experiments_open_query_detection_on_features \
			--exp_name 'benchmark' \
			--inference_method ${method} \
			--n_shot 5 \
			--outlier_detectors knn_med_1-knn_med_3-knn_med_5-knn_med_7-knn_med_9-knn_med_11 \
			--prepool_transform  base_centering \
			--postpool_transform  l2_norm \
		    --backbone ${backbone} \
		    --simu_hparams 'training' \
		    --training ${training} \
		    --dataset ${dataset} \
		    --override
	done
done

# # ====== Trying iterative version =========

# python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'going_iterative' \
# 																  --inference_method ${method} \
# 																  --outlier_detector 'iterative_knn' \
# 																  --prepool_transform  base_centering layer_norm \
# 																  --postpool_transform  l2_norm \
# 																  --nn 10 \
# 																  --backbone resnet18 \
# 																  --simu_hparams 'nn' \
# 																  --override