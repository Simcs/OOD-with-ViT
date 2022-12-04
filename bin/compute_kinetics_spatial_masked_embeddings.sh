for threshold in 0.02
do
    for dataset in k400 k600 k700-2020
    do
        echo python -m bin.compute_kinetics_masked_embeddings --dataset ${dataset} --split val \
            --mask_mode 'average' --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
        python -m bin.compute_kinetics_masked_embeddings \
            --dataset ${dataset} \
            --mask_mode 'average' \
            --split val \
            --spatial_masking \
            --spatial_mask_method lt_threshold \
            --spatial_mask_threshold ${threshold}
    done
done