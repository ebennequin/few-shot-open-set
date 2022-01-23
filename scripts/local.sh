
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
        --pool False \
        --backbone resnet12 \
        --training feat \
        --dataset mini_imagenet \
        --simu_hparams 'current_sequence' \
        --combination_size 1 \
        --override
done