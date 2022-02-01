
for shot in 5; do \
    for dataset in mini_imagenet; do \
        # for layers in 4_0-4_1 4_1-4_2 4_0-4_2 4_0-4_1-4_2; do
        for layers in 4_2; do
            python3 -m notebooks.experiments_open_query_detection_on_features \
                --exp_name "aggreg-${dataset}-${shot}-shot" \
                --mode 'tune' \
                --inference_method SimpleShot \
                --n_tasks 500 \
                --n_shot ${shot} \
                --outlier_detectors knn \
                --layers ${layers} \
                --aggreg l2_bar \
                --prepool_transform base_centering \
                --pool \
                --postpool_transform  l2_norm \
                --backbone resnet12 \
                --training feat \
                --dataset ${dataset} \
                --simu_hparams layers current_sequence aggreg \
                --combination_size 1 \
                --override
        done
    done
done