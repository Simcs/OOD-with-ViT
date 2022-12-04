root="./data/kinetics/embeddings/masked/k600_val"
dataset="k600"
echo python -m bin.deduplicate_kinetics_embeddings --root ${root} --dataset ${dataset} --split val
python -m bin.deduplicate_kinetics_embeddings --root ${root} --dataset ${dataset} --split val
root="./data/kinetics/embeddings/masked/k700-2020_val"
dataset="k700-2020"
echo python -m bin.deduplicate_kinetics_embeddings --root ${root} --dataset ${dataset} --split val
python -m bin.deduplicate_kinetics_embeddings --root ${root} --dataset ${dataset} --split val