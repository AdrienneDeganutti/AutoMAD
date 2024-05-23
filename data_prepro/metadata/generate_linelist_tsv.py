import json
import csv


annotations_file = '/mnt/welles/scratch/adrienne/MAD/annotations/MAD-v1/MAD_test.json'
output_file = '/home/adrienne/AutoMAD/datasets/test/test.caption.linelist.tsv'
ignore_clips_file = '/home/adrienne/AutoMAD/datasets/archive-clips-under-8-frames/test-clips-under-8frames.txt'

print('Loading cached annotations...')
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Convert annotations to a list of dictionaries for easier processing
annotations_list = [{**{"id": key}, **value} for key, value in annotations.items()]
movies = {a['movie']: a['movie_duration'] for a in annotations_list}

ignore_IDs = []
print('Loading clip IDs to exclude from dataset...')
with open(ignore_clips_file, 'r') as g:
    for line in g:
        ignore_IDs.append(int(line))

incr = 0

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations_list:
        annotation_ID = annotation['id']

        if int(annotation_ID) in ignore_IDs:
            continue
        else:

            writer.writerow([incr, 0])

            incr += 1