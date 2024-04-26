import pickle as pkl
import torch
import h5py
import csv

language_features_file = '/mnt/welles/scratch/adrienne/MAD/features/CLIP_B32_language_tokens_features.h5'
annotations_file = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/MAD_train_annotations.pickle'

output_file = '/home/adrienne/AutoMAD/datasets/1_clip_language_features2.tsv'

print('Loading cached annotations...')
annotations = pkl.load(open(annotations_file, 'rb'))
movies = {a['movie']:a['movie_duration'] for a in annotations}

print('Loading language features...')
with h5py.File(language_features_file, 'r') as f:
    language_feats = {m: torch.from_numpy(f[m][:]) for m in movies}

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    
    for annotation in annotations:
        annotation_ID = annotation['id']
        movie_ID = annotation['movie']
        features = language_feats[movie_ID]

        row = int(annotation_ID), features.numpy().tolist()
        writer.writerow(row)

        break           #break for 1 clip

