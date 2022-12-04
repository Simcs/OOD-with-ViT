for threshold in 0.012 0.014 0.016 0.018
do
    echo python -m bin.compute_kinetics_masked_classwise_mahalanobis \
        --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
    python -m bin.compute_kinetics_masked_classwise_mahalanobis \
        --spatial_masking \
        --spatial_mask_method lt_threshold \
        --spatial_mask_threshold ${threshold}
    
    echo python -m bin.compute_kinetics_masked_mahalanobis \
        --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
    python -m bin.compute_kinetics_masked_mahalanobis \
        --spatial_masking \
        --spatial_mask_method lt_threshold \
        --spatial_mask_threshold ${threshold}
    
    echo python -m bin.compute_kinetics_masked_msp \
        --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
    python -m bin.compute_kinetics_masked_msp \
        --spatial_masking \
        --spatial_mask_method lt_threshold \
        --spatial_mask_threshold ${threshold}
done