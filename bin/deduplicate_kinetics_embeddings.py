from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import argparse
import os


def deduplicate_embeddings(args):
    root = args.root
    dataset, split, force = args.dataset, args.split, args.force
    print(f'removing {dataset} split {split}...')
    
    root_path = Path(root)
    for embeddings_filename in sorted(root_path.glob('*.jsonl')):
        embeddings_filename = embeddings_filename.absolute()
        name, ext = embeddings_filename.stem, embeddings_filename.suffix
        deduplicated_embeddings_filename = embeddings_filename.parent / (name + '_deduplicated' + ext)
        print('current filename:', embeddings_filename)
        print('target filename:', deduplicated_embeddings_filename)

        # original_embeddings_filename = f'./data/kinetics/{dataset}_original_{split}_embeddings.jsonl'
        # original_embeddings_filename = f'./data/kinetics/embeddings/masked/{dataset}_original_{split}_embeddings.jsonl'
        
        if 'deduplicated' in str(embeddings_filename):
            continue
        if not force and os.path.exists(deduplicated_embeddings_filename):
            continue
        
        classes = pd.read_csv(f'./data/kinetics/k400_{dataset}_classes.tsv', sep='\t')
        classes = classes[classes['k400_id'] == -1][f'{dataset}_class']
        kinetics_original = classes.to_list()
        kinetics_original_embeddings = []
        
        id = 0
        print('extracting original embeddings...')
        with open(embeddings_filename, 'r') as f:
            for line in tqdm(f):
                emb_js = json.loads(line)
                gt_label, pred_label = emb_js['gt'], emb_js['pred']
                pre_logit, logit = emb_js['penultimate'], emb_js['logit']
                if gt_label in kinetics_original:
                    kinetics_original_embeddings.append({
                        'id': id, 
                        'gt': gt_label, 
                        'pred': pred_label, 
                        'penultimate': pre_logit,
                        'logit': logit,   
                    })
                    id += 1
                    
        print('writing original embeddings...')       
        with open(deduplicated_embeddings_filename, 'w') as f:
            for emb in tqdm(kinetics_original_embeddings):
                f.write(json.dumps(emb) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--dataset', choices=['k600', 'k700-2020'])
    parser.add_argument('--split', choices=['val'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    deduplicate_embeddings(args)