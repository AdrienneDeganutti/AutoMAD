import sys
import torch
import pickle as pkl
import csv
from itertools import islice

MAD_features = '/home/adrienne/AutoMAD/datasets/8_frames_tsv/test_segmented_features.tsv'

csv.field_size_limit(sys.maxsize)

print('Loading MAD video features...')
with open(MAD_features, 'r', encoding='utf-8') as MAD_file:
    # Create a CSV reader object specifying the delimiter as a tab
    MAD_reader = csv.reader(MAD_file, delimiter='\t')

    #MAD_data = [row for row in islice(MAD_reader, 10)]          #load only first 10 lines
    MAD_data = [row for row in (MAD_reader)]                   #load full file

    #for row in MAD_reader:
    #    if row[1] == None:
    #        print('Error at: ')
    #        print(row[0])
    #        continue
    #    else:
    #        MAD_data = row
    print('completed MAD load in')
