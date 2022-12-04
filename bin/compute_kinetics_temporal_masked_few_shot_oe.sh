for threshold in 0.009 0.01
do
    for shot in 1 2 3 5 10 20 50
    do
        echo python -m bin.compute_kinetics_masked_few_shot_oe \
            --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${threshold} --n_shot ${shot}
        python -m bin.compute_kinetics_masked_few_shot_oe \
            --temporal_masking \
            --temporal_mask_method lt_threshold \
            --temporal_mask_threshold ${threshold} \
            --n_shot ${shot}
    done
done