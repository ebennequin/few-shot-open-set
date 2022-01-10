
for prepool in trivial inductive_batch_norm transductive_batch_norm layer_norm; do
	for postpool in l2_norm; do
		for detector in knn; do 
			python3 -m notebooks.experiments_open_query_detection_on_features --exp_name 'normalizations' \
																			  --inference_method ${method} \
																			  --outlier_detector ${detector} \
																			  --prepool_transform ${prepool} \
																			  --postpool_transform ${postpool}
		done
	done
done
