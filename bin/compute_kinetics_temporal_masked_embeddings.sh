for threshold in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009
do
    for dataset in k400 k600 k700-2020
    do
        echo python -m bin.compute_kinetics_masked_embeddings --dataset ${dataset} --split val \
            --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${threshold}
        python -m bin.compute_kinetics_masked_embeddings \
            --dataset ${dataset} \
            --split val \
            --temporal_masking \
            --temporal_mask_method lt_threshold \
            --temporal_mask_threshold ${threshold}
    done
done