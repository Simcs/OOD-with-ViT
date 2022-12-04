for spatial_threshold in 0.01 0.02 0.03
do
    for temporal_threhold in 0.01 0.02 0.03
    do
        echo python -m bin.compute_kinetics_masked_msp \
            --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
            --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold}
        python -m bin.compute_kinetics_masked_msp \
            --spatial_masking \
            --spatial_mask_method lt_threshold \
            --spatial_mask_threshold ${spatial_threshold} \
            --temporal_masking \
            --temporal_mask_method lt_threshold \
            --temporal_mask_threshold ${temporal_threhold}
    done
done