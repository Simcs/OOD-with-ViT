for threshold in 0.01
do
    echo python -m bin.compute_kinetics_masked_mahalanobis \
        --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
    python -m bin.compute_kinetics_masked_mahalanobis \
        --spatial_masking \
        --spatial_mask_method lt_threshold \
        --spatial_mask_threshold ${threshold}
done