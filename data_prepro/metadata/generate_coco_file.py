import pickle as pkl
import json


annotations_file = '/mnt/welles/scratch/adrienne/MAD/annotations/MAD-v1/MAD_test.json'
output_file = '/home/adrienne/AutoMAD/datasets/test/test.caption_coco_format.json'

print('Loading cached annotations...')
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Convert annotations to a list of dictionaries for easier processing
annotations_list = [{**{"id": key}, **value} for key, value in annotations.items()]
movies = {a['movie']: a['movie_duration'] for a in annotations_list}


id_counter = 0
previous_movie_id = 0
annotations_aggregate = []
images_list = []

 
for annotation in annotations_list:
    annotation_ID = annotation['id']
    caption = annotation.get('sentence') or annotation.get('caption', '')

    if 'movie' in annotation:
        movie_ID = annotation['movie']
        if not movie_ID == previous_movie_id:
            print(f'Processing movie ID: {movie_ID}...')
    
    annotations_aggregate.append({
        "image_id": annotation_ID,
        "caption": caption,
        "id": id_counter
    })

    images_list.append({
        "id": annotation_ID,
        "file_name": annotation_ID
    })

    id_counter += 1
    previous_movie_id = movie_ID

final_json = {"annotations": annotations_aggregate, "images": images_list}

with open(output_file, 'w') as f:
    json.dump(final_json, f, indent=4)