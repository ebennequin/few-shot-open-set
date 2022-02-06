training="feat.pth"
for layer in 4_0 4_1 4_2 4_3; do
    for dataset in mini_imagenet; do
        for split in train test; do
            for arch in resnet12; do
                python -m stages.compute_features ${arch} ${dataset} data/models/${arch}_${dataset}_${training} --split ${split} --layer ${layer}
            done
        done
    done
done
