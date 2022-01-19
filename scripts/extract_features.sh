training="feat.pth"
for dataset in tiered_imagenet; do
    for split in train test; do
        for arch in wrn2810; do
            python -m stages.compute_features ${arch} ${dataset} data/models/${arch}_${dataset}_${training} --split ${split}
        done
    done
done
