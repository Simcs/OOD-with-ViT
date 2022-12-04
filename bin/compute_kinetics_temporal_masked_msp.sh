for threshold in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
do
    echo python -m bin.compute_kinetics_masked_msp \
        --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${threshold}
    python -m bin.compute_kinetics_masked_msp \
        --temporal_masking \
        --temporal_mask_method lt_threshold \
        --temporal_mask_threshold ${threshold}
done