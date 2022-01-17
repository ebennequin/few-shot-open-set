method="SimpleShot"

# ====== Sanity check with my training ========

# python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'sanity_check' \
# 																  --inference_method ${method} \
# 																  --outlier_detector 'knn' \
# 																  --prepool_transform trivial \
# 																  --postpool_transform base_centering l2_norm \
# 																  --training malik \
# 																  --simu_hparams 'nn' 'training'


# ====== Search best way to normalize ========

# for training in feat; do
# 	for prepool in trivial inductive_batch_norm transductive_batch_norm layer_norm; do
# 		for postpool in trivial l2_norm; do
# 			for detector in knn; do 
# 				python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'normalizations' \
# 																				  --inference_method ${method} \
# 																				  --outlier_detector ${detector} \
# 																				  --prepool_transform ${prepool} \
# 																				  --postpool_transform base_centering ${postpool} \
# 																				  --training ${training} \
# 																				  --backbone resnet12 \
# 																				  --simu_hparams 'nn' 'training'
# 			done
# 		done
# 	done
# done

# ====== Increasing k in kNN =========

# for nn in 1 3 5 7 10 12 15 17 20; do
# 			python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'tuning_knn' \
# 																			  --inference_method ${method} \
# 																			  --outlier_detector 'knn' \
# 																			  --prepool_transform  trivial \
# 																			  --postpool_transform  base_centering l2_norm \
# 																			  --nn ${nn} \
# 																			  --backbone resnet12 \
# 																			  --simu_hparams 'nn' 'training' \
# 																			  --training feat \
# 																			  --override
# done

# ========= Benchmarking properly ==========

python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'benchmark' \
																  --inference_method ${method} \
																  --n_shot 1 \
																  --outlier_detector 'knn' \
																  --prepool_transform  base_centering \
																  --postpool_transform  l2_norm \
																  --nn 4 \
																  --backbone resnet12 \
																  --simu_hparams 'nn' 'training' \
																  --training feat \
																  --override
																  
python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'benchmark' \
													  --inference_method ${method} \
													  --n_shot 5 \
													  --outlier_detector 'knn' \
													  --prepool_transform  base_centering \
													  --postpool_transform  l2_norm \
													  --nn 15 \
													  --backbone resnet12 \
													  --simu_hparams 'nn' 'training' \
													  --training feat \
													  --override

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