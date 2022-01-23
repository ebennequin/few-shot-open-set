
shot=5

# Trying detectors individually
for detector in knn ; do
    python3 -m notebooks.experiments_open_query_detection_on_features \
        --exp_name "individual detectors - ${shot}" \
        --mode 'tune' \
        --inference_method SimpleShot \
        --n_tasks 500 \
        --n_shot ${shot} \
        --outlier_detectors ${detector} \
        --prepool_transform  base_centering \
        --postpool_transform  l2_norm \
        --backbone resnet12 \
        --training feat \
        --dataset mini_imagenet \
        --simu_hparams 'current_sequence' \
        --combination_size 1 \
        --override
done



# Test the promising ones

# for shot in 5; do \
#     python3 -m notebooks.experiments_open_query_detection_on_features \
#         --exp_name 'bagging' \
#         --mode 'tune' \
#         --inference_method SimpleShot \
#         --n_tasks 500 \
#         --n_shot ${shot} \
#         --outlier_detectors knn \
#         --prepool_transform  base_centering \
#         --postpool_transform  l2_norm \
#         --backbone resnet12 \
#         --training feat \
#         --dataset mini_imagenet \
#         --simu_hparams 'current_sequence' \
#         --combination_size 3 \
#         --override
# done \