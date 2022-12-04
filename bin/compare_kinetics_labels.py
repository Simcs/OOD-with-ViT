from ood_with_vit.datasets.kinetics import VideoOnlyKinetics
import os
import pickle
import pandas as pd

# load metadata
with open('./data/kinetics/k400_val_metadata.pkl', 'rb') as f:
    k400_val_metadata = pickle.load(f)
with open('./data/kinetics/k600_val_metadata.pkl', 'rb') as f:
    k600_val_metadata = pickle.load(f)
with open('./data/kinetics/k700-2020_val_metadata.pkl', 'rb') as f:
    k700_val_metadata = pickle.load(f)


# initialize datasets
k400_root = f'~/workspace/dataset/kinetics/k400'
k400_root = os.path.expanduser(k400_root)
k400 = VideoOnlyKinetics(
    root=k400_root,
    frames_per_clip=16,
    split='val',
    num_workers=16,
    frame_rate=2,
    _precomputed_metadata=k400_val_metadata,
)
print('k400:')
with open('./data/kinetics/k400_classes.txt', 'w') as f:
    for i, c in enumerate(k400.classes):
        print(i, c, file=f)


k600_root = f'~/workspace/dataset/kinetics/k600'
k600_root = os.path.expanduser(k600_root)
k600 = VideoOnlyKinetics(
    root=k600_root,
    frames_per_clip=16,
    split='val',
    num_workers=16,
    frame_rate=2,
    _precomputed_metadata=k600_val_metadata,
)
print('k600:')
with open('./data/kinetics/k600_classes.txt', 'w') as f:
    for i, c in enumerate(k600.classes):
        print(c, file=f)
        
        
k700_root = f'~/workspace/dataset/kinetics/k700-2020'
k700_root = os.path.expanduser(k700_root)
k700 = VideoOnlyKinetics(
    root=k700_root,
    frames_per_clip=16,
    split='val',
    num_workers=16,
    frame_rate=2,
    _precomputed_metadata=k700_val_metadata,
)
print('k700:')
with open('./data/kinetics/k700_classes.txt', 'w') as f:
    for i, c in enumerate(k700.classes):
        print(c, file=f)
        

# refine k400_classes
k400_classes = []
for k400_cls in k400.classes:
    if k400_cls.count('-') == 1:
        # massaging_person-s_head -> massaging_person's_head
        k400_cls = k400_cls.replace('-', '\'')
    else:
        # petting_animal_-not_cat- -> petting_animal_(not_cat)
        k400_cls = k400_cls.replace('-', '(', 1)
        k400_cls = k400_cls.replace('-', ')', 1)
    k400_classes.append(k400_cls)
    
        
# compare k400 vs. k600
existing_k400 = []
existing_labels = []
for j, k600_cls in enumerate(k600.classes):
    is_exist = False
    for i, k400_cls in enumerate(k400_classes):
        if k400_cls == k600_cls:
            existing_labels.append([i, k400_cls, j, k600_cls])
            existing_k400.append(k400_cls)
            is_exist = True
            break
    if not is_exist:
        existing_labels.append([-1, None, j, k600_cls])

non_existing_k400 = []
for i, k400_cls in enumerate(k400_classes):
    if k400_cls not in existing_k400:
        non_existing_k400.append([i, k400_cls])

df = pd.DataFrame(existing_labels, columns=['k400_id', 'k400_class', 'k600_id', 'k600_class'])
df.to_csv('./data/kinetics/k400_k600_classes.tsv', sep='\t')

df = pd.DataFrame(non_existing_k400, columns=['k400_id', 'k400_class'])
df.to_csv('./data/kinetics/missing_k400_classes_(k600).tsv', sep='\t')
print(len(non_existing_k400))


# compare k400 vs. k700-2020
existing_k400 = []
existing_labels = []
for j, k700_cls in enumerate(k700.classes):
    is_exist = False
    for i, k400_cls in enumerate(k400_classes):
        if k400_cls == k700_cls:
            existing_labels.append([i, k400_cls, j, k700_cls])
            existing_k400.append(k400_cls)
            is_exist = True
            break
    if not is_exist:
        existing_labels.append([-1, None, j, k700_cls])

non_existing_k400 = []
for i, k400_cls in enumerate(k400_classes):        
    if k400_cls not in existing_k400:
        non_existing_k400.append([i, k400_cls])

df = pd.DataFrame(existing_labels, columns=['k400_id', 'k400_class', 'k700-2020_id', 'k700-2020_class'])
df.to_csv('./data/kinetics/k400_k700-2020_classes.tsv', sep='\t')

df = pd.DataFrame(non_existing_k400, columns=['k400_id', 'k400_class'])
df.to_csv('./data/kinetics/missing_k400_classes_(k700-2020).tsv', sep='\t')
print(len(non_existing_k400))