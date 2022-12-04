for threshold in 0.1
do
    for dataset in k400
    do
        echo python -m bin.compute_kinetics_masked_embeddings --dataset ${dataset} --split train \
            --mask_mode 'zero' --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
        python -m bin.compute_kinetics_masked_embeddings \
            --dataset ${dataset} \
            --mask_mode 'zero' \
            --split train \
            --spatial_masking \
            --spatial_mask_method lt_threshold \
            --spatial_mask_threshold ${threshold}
    done
done