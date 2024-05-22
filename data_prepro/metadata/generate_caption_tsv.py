import csv
import json

annotations_file = '/mnt/welles/scratch/adrienne/MAD/annotations/MAD-v1/MAD_test.json'
output_file = '/home/adrienne/AutoMAD/datasets/test/test.label.tsv'

print('Loading cached annotations...')
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Convert annotations to a list of dictionaries for easier processing
annotations_list = [{**{"id": key}, **value} for key, value in annotations.items()]
movies = {a['movie']: a['movie_duration'] for a in annotations_list}

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations_list:
        annotation_ID = annotation['id']
        caption = annotation['sentence']
        caption_dict = [{'caption': caption}]

        writer.writerow([annotation_ID, caption_dict])