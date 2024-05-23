import csv
import json

annotations_file = '/mnt/welles/scratch/adrienne/MAD/annotations/MAD-v1/MAD_val.json'
output_file = '/home/adrienne/AutoMAD/datasets/test/val.img.tsv'
ignore_clips_file = '/home/adrienne/AutoMAD/datasets/archive-clips-under-8-frames/val-clips-under-8frames.txt'

print('Loading cached annotations...')
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

ignore_IDs = []
print('Loading clip IDs to exclude from dataset...')
with open(ignore_clips_file, 'r') as g:
    for line in g:
        ignore_IDs.append(int(line))

# Convert annotations to a list of dictionaries for easier processing
annotations_list = [{**{"id": key}, **value} for key, value in annotations.items()]
movies = {a['movie']: a['movie_duration'] for a in annotations_list}

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations_list:
        annotation_ID = annotation['id']

        if int(annotation_ID) in ignore_IDs:
            continue
        else:

            row = [annotation_ID] + [annotation_ID]
            writer.writerow(row)