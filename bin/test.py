import json
from tqdm import tqdm

lines = []
fn = '/home/simc/workspace/OOD-with-ViT/data/kinetics/embeddings/masked/k400_val/spatial_max_0.5_lt_threshold_0.01.jsonl'
with open(fn, 'r') as f:
    for i, line in tqdm(enumerate(f)):
        emb_js = json.loads(line)
        if i != int(emb_js['id']):
            lines.append(line)

tmp_fn = '/home/simc/workspace/OOD-with-ViT/data/kinetics/embeddings/masked/k400_val/tmp_spatial_max_0.5_lt_threshold_0.01.jsonl'
# for line in lines:
#     print(line + '\n')
with open(tmp_fn, 'w') as f:
    for line in lines:
        f.write(line)