import h5py
import torch
import random
import csv
import json

def uniform_subsample(tensor, num_samples):
    """
    Uniformly subsample 'num_samples' frames from the tensor.
    If the total_frames is less than num_samples, pad with zeros to match the desired size.
    """
    total_frames = tensor.size(0)
    if total_frames == 0:
        raise ValueError("Tensor has no frames.")
    if total_frames < num_samples:
        padded_tensor = torch.zeros((num_samples, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:total_frames] = tensor
        return padded_tensor
    indices = torch.linspace(0, total_frames - 1, num_samples).long()
    return tensor[indices]

video_features_file = '/mnt/welles/scratch/adrienne/MAD/features/CLIP_B32_frames_features_5fps.h5'
annotations_file = '/mnt/welles/scratch/adrienne/MAD/annotations/MAD-v1/MAD_test.json'
output_file = '/home/adrienne/AutoMAD/datasets/test/8_frames_test_features.tsv'

frames_sampling = 1

print('Loading cached annotations...')
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Convert annotations to a list of dictionaries for easier processing
annotations_list = [{**{"id": key}, **value} for key, value in annotations.items()]
movies = {a['movie']: a['movie_duration'] for a in annotations_list}

print('Loading video features...')
with h5py.File(video_features_file, 'r') as f:
    video_feats = {m: torch.from_numpy(f[m][:]) for m in movies}

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    previous_movie_id = 0

    for annotation in annotations_list:
        annotation_ID = annotation['id']
        movie_ID = annotation['movie']
        start_timestamp, end_timestamp = annotation['timestamps']
        movie_duration = annotation['movie_duration']

        # Convert timestamps to frame indices
        start_frame = int((start_timestamp / movie_duration) * len(video_feats[movie_ID]))
        end_frame = int((end_timestamp / movie_duration) * len(video_feats[movie_ID]))

        try:
            numeric_id = int(annotation_ID)
        except ValueError:
            print("Invalid ID: not a number!")
        

        else:
            if not movie_ID == previous_movie_id:
                print(f'Processing movie ID: {movie_ID}...')
            full_feature_tensor = video_feats[movie_ID]

            segmented_feature_tensor = full_feature_tensor[start_frame:end_frame + 1]
            subsampled_feature_tensor = uniform_subsample(segmented_feature_tensor, frames_sampling)

            # Write annotation ID and frame tensors in one line
            row = [annotation_ID] + [frame_tensor.numpy().tolist() for frame_tensor in subsampled_feature_tensor]
            writer.writerow(row)

            previous_movie_id = movie_ID
