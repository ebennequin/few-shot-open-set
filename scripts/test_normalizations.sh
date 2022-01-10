method="SimpleShot"

# ====== Search best way to normalize ========

for training in malik classic; do
	for prepool in trivial inductive_batch_norm transductive_batch_norm layer_norm; do
		for postpool in trivial l2_norm; do
			for detector in knn; do 
				python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'normalizations' \
																				  --inference_method ${method} \
																				  --outlier_detector ${detector} \
																				  --prepool_transform ${prepool} \
																				  --postpool_transform ${postpool} \
																				  --training ${training} \
																				  --simu_hparams 'nn' 'training'
			done
		done
	done
done

# ====== Increasing k in kNN =========

# for nn in 1 3 5 7 10 12 15 17 20; do
# 			python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'tuning_knn' \
# 																			  --inference_method ${method} \
# 																			  --outlier_detector 'knn' \
# 																			  --prepool_transform  base_centering layer_norm \
# 																			  --postpool_transform  l2_norm \
# 																			  --nn ${nn} \
# 																			  --backbone resnet18 \
# 																			  --simu_hparams 'nn' \
# 																			  --override
# done

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