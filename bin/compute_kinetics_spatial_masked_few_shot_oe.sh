for threshold in 0.009 0.01
do
    for shot in 20
    do
        echo python -m bin.compute_kinetics_masked_few_shot_oe \
            --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold} --n_shot ${shot}
        python -m bin.compute_kinetics_masked_few_shot_oe \
            --spatial_masking \
            --spatial_mask_method lt_threshold \
            --spatial_mask_threshold ${threshold} \
            --n_shot ${shot}
    done
done