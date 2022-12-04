for shot in 1 2 3 5 10 20 50
do
    echo python -m bin.compute_kinetics_few_shot_oe --n_shot ${shot}
    python -m bin.compute_kinetics_few_shot_oe --n_shot ${shot}
done