import json

# INPUT DIRECTORIES: #
annotations_file = '/mnt/welles/scratch/adrienne/MAD/annotations/MAD-v1/MAD_val.json'
output_file = '/home/adrienne/AutoMAD/datasets/test/val.caption_coco_format.json'
ignore_clips_file = '/home/adrienne/AutoMAD/datasets/archive-clips-under-8-frames/val-clips-under-8frames.txt'

print('Loading cached annotations...')
with open(annotations_file, 'r') as f:
    json_annotations = json.load(f)

ignore_IDs = []
print('Loading clip IDs to exclude from dataset...')
with open(ignore_clips_file, 'r') as g:
    for line in g:
        ignore_IDs.append(int(line))

# Convert annotations to a list of dictionaries for easier processing
annotations_list = [{**{"id": key}, **value} for key, value in json_annotations.items()]
movies = {a['movie']: a['movie_duration'] for a in annotations_list}


id_counter = 0
previous_movie_id = 0
final_annotations = []
images_list = []
end_metadata = []

 
for annotation in annotations_list:
    annotation_ID = annotation['id']

    if int(annotation_ID) in ignore_IDs:
        continue
    else:

        caption = annotation.get('sentence') or annotation.get('caption', '')

        if 'movie' in annotation:
            movie_ID = annotation['movie']
            if not movie_ID == previous_movie_id:
                print(f'Processing movie ID: {movie_ID}...')
    
        final_annotations.append({
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

final_json = {"annotations": final_annotations, "images": images_list}

with open(output_file, 'w') as f:
    json.dump(final_json, f, indent=4)